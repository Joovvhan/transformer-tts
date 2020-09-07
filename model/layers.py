import torch
from torch.nn import Conv1d, Conv2d
from torch.nn import BatchNorm1d, ReLU, Dropout
from torch.nn import ModuleList, Sequential
from torch import nn
from utils import ENCODING_SIZE
import numpy as np
import math

import logging

log = logging.getLogger('log')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)s] (%(filename)s:%(lineno)d) > %(message)s')
fileHandler = logging.FileHandler('./log.txt')
fileHandler.setFormatter(formatter)

log.addHandler(fileHandler)


class Linear(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, weight_init='linear'):
        super(Conv, self).__init__()
        self.conv = torch.Conv1d(in_channels, out_channels,
                                 kernel_size=kernel_size)
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(weight_init))

    def forward(self, input_tensor):
        return self.conv(input_tensor)


class DummyEmbedding(torch.nn.Module):

    '''
     Input: (B, L)
    Output: (B, L, H)
    '''

    def __init__(self, configs):
        super(DummyEmbedding, self).__init__()

        self.embedding = torch.nn.Embedding(configs['n_mel_channels'], configs['embedding_dim'], padding_idx=0)

    def forward(self, input_tensor):

        output_tensor = self.embedding(input_tensor) # (B, L) -> (B, L, H)
        output_tensor = torch.transpose(output_tensor, 1, 2) # (B, L, H) -> (B, H, L)

        return output_tensor # (B, H, L)



class DummyPrenet(torch.nn.Module):
    def __init__(self, configs):
        super(DummyPrenet, self).__init__()
        self.configs = configs
        '''
        From Transformer-TTS:
        we add a linear projection after the final ReLU activation
        '''
        self.prenet_layers = ModuleList([DummyPrenetModule(configs) for i in range(3)])
        self.pernet_linear = Linear(512, 512)

    def forward(self, input_tensor):

        tensor = input_tensor

        for layer in self.prenet_layers:
            tensor = layer(tensor) # (B, H, L) -> (B, H, L)
        tensor = tensor.permute(0, 2, 1) # (B, H, L) -> (B, L, H)

        tensor = self.prenet_linear(tensor) # (B, L, H) -> (B, L, H)

        output_tensor = tensor
        log.info("prenet forward ")
        
        return output_tensor


class EncoderPrenet(torch.nn.Module):
    def __init__(self, configs):
        super(EncoderPrenet, self).__init__()
        self.configs = configs
        self.embedding_dim = configs['embedding_dim']
        self.embed = torch.nn.Embedding(ENCODING_SIZE, configs['embedding_dim'])
        self.conv = Conv1d(configs['embedding_dim'], configs['num_hidden'], kernel_size=5,
                                    padding=int(np.floor(5/2)))
        self.batch_norm = torch.nn.BatchNorm1d(configs['num_hidden'])
        self.dropout = torch.nn.Dropout(p=0.2)
        self.linear = Linear(configs['num_hidden'], configs['num_hidden'])

    def forward(self, input_tensor):
        tensor = self.embed(input_tensor)
        tensor = tensor.transpose(1, 2)
        #log.debug("transpose : ", tensor.shape)
        #for layer in self.prenet_layers:
        tensor = self.conv(tensor)  # (B, H, L) -> (B, H, L)
        tensor = self.batch_norm(tensor)
        tensor = self.dropout(tensor)
        #log.debug("prenet layer : ", tensor.shape)
        tensor = tensor.permute(0, 2, 1)  # (B, H, L) -> (B, L, H)
        #log.debug("transpose : ", tensor.shape)
        tensor = self.linear(tensor)  # (B, L, H) -> (B, L, H)
        #log.debug("prenet linear : ", tensor.shape)
        output_tensor = tensor
        return output_tensor


class DecoderPrenet(torch.nn.Module):
    def __init__(self, configs):
        super(DecoderPrenet, self).__init__()
        self.configs = configs
        self.num_mels = self.configs['n_mel_channels']
        self.num_hidden = self.configs['num_hidden']
        self.layers = ModuleList([
            Linear(self.num_mels, self.num_hidden * 2),
            ReLU(),
            Dropout(p=0.2),
            Linear(self.num_hidden * 2, self.num_hidden),
            ReLU(),
            Dropout(p=0.2)
        ])

    def forward(self, input_tensor):
        tensor = input_tensor
        for layer in self.layers:
            tensor = layer(tensor)
        log.info("Decoder prenet ")
        return tensor


class FFN(torch.nn.Module):
    def __init__(self, configs):
        super(FFN, self).__init__()
        self.configs = configs
        self.num_hidden = self.configs['num_hidden']
        self.layers = ModuleList([
            Conv1d(self.num_hidden,
                   self.num_hidden * 4,
                   kernel_size=1),
            ReLU(),
            Conv1d(self.num_hidden * 4,
                   self.num_hidden,
                   kernel_size=1)
        ])
        self.layer_norm = torch.nn.LayerNorm(self.num_hidden)

    def forward(self, input_tensor):
        tensor = input_tensor
        tensor = tensor.transpose(1, 2)
        for layer in self.layers:
            tensor = layer(tensor)
        tensor = tensor.transpose(1, 2)

        # Residual
        tensor = tensor + input_tensor

        tensor = self.layer_norm(tensor)
        return tensor


class MultiheadAttention(torch.nn.Module):
    def __init__(self, num_hidden):
        super(MultiheadAttention, self).__init__()

        self.num_hidden = num_hidden
        self.attention_dropout = torch.nn.Dropout(p=0.1)

    def forward(self, key, value, query, mask, query_mask):
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden)

        result = torch.bmm(attn, value)
        return result, attn


class DummyAttention(torch.nn.Module):
    def __init__(self, configs):
        super(DummyAttention, self).__init__()
        self.configs = configs
        self.num_hidden = self.configs['num_hidden']
        self.multihead_num = self.configs['multihead_num']
        self.embedding_dim = configs['embedding_dim']
        self.multihead = MultiheadAttention(self.num_hidden // self.multihead_num)
        self.key = Linear(self.num_hidden, self.num_hidden, bias=False)
        self.value = Linear(self.num_hidden, self.num_hidden, bias=False)
        self.query = Linear(self.num_hidden, self.num_hidden, bias=False)
        self.normalization = torch.nn.LayerNorm(self.num_hidden)
        self.linear = Linear(self.num_hidden * 2, self.num_hidden)

    def forward(self, input_tensor, input_tensor2, mask, query_mask):
        batch_size = input_tensor.size(0)
        input_key = input_tensor.size(1)
        input_query = input_tensor2.size(1)
        if query_mask is not None:
            query_mask = query_mask.unsqueeze(-1).repeat(1, 1, input_key).repeat(self.num_hidden, 1, 1)
        if mask is not None:
            mask = mask.repeat(self.num_hidden, 1, 1)

        key = self.key(input_tensor).view(batch_size,
                                          input_key,
                                          self.multihead_num,
                                          self.num_hidden // self.multihead_num)
        '''value = self.value(input_tensor).view(input_tensor_0,
                                          input_tensor_1,
                                          self.num_hidden)'''
        value = self.value(input_tensor).view(batch_size,
                                          input_key,
                                          self.multihead_num,
                                          self.num_hidden // self.multihead_num)
        query = self.query(input_tensor).view(batch_size,
                                              input_query,
                                              self.multihead_num,
                                              self.num_hidden // self.multihead_num)
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, input_key,
                                                        self.num_hidden // self.multihead_num)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, input_key,
                                                            self.num_hidden // self.multihead_num)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, input_query,
                                                            self.num_hidden // self.multihead_num)

        result, attns = self.multihead(key, value, query, mask, query_mask)
        result = result.view(self.multihead_num, batch_size, input_query,
                             self.num_hidden // self.multihead_num)
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, input_query, -1)

        result = torch.cat([input_tensor, result], dim=-1)
        result = self.linear(result)

        # residual
        result = result + input_tensor2

        result = self.normalization(result)
        return result, attns

        # return self.multihead(key, value, query)


class Encoder(torch.nn.Module):
    def __init__(self, configs):
        super(Encoder, self).__init__()
        self.configs = configs
        self.encoder_prenet = EncoderPrenet(configs)
        self.attention = DummyAttention(configs)
        self.ffn = FFN(configs)

    def forward(self, input_tensor):
        c_mask = input_tensor.ne(0).type(torch.float)
        mask = input_tensor.eq(0).unsqueeze(1).repeat(1, input_tensor.size(1), 1)
        tensor = self.encoder_prenet(input_tensor)
        attns = list()

        tensor, attn = self.attention(tensor, tensor, mask, c_mask)
        tensor = self.ffn(tensor)
        attns.append(attn)
        '''for layer, ffn in zip(self.attention, self.ffn):
            tensor, attn = layer(tensor, mask, c_mask)
            tensor = ffn(tensor)
            attns.append(attn)'''
        return tensor, c_mask, attns


class Decoder(torch.nn.Module):
    def __init__(self, configs):
        super(Decoder, self).__init__()
        self.configs = configs
        self.decoder_prenet = DecoderPrenet(configs)
        self.attention = DummyAttention(configs)
        self.ffn = FFN(configs)
        self.num_hidden = configs['num_hidden']
        self.num_mels = configs['n_mel_channels']
        self.mel_linear = Linear(self.num_hidden, self.num_mels)
        self.stop_linear = Linear(self.num_hidden, 1)

    def forward(self, input_tensor, input_tensor2, c_mask, mel_input):
        batch_size = input_tensor.size(0)
        input_tensor_1 = input_tensor2.size(1)
        tensor = self.decoder_prenet(mel_input.type(torch.cuda.FloatTensor))

        '''m_mask = mel_input.ne(0).type(torch.float)
        #print(m_mask.eq(0).unsqueeze(1))
        #mask = m_mask.eq(0).unsqueeze(1).repeat(1, input_tensor_1, 1)
        mask = m_mask.eq(0).unsqueeze(1)
        if next(self.parameters()).is_cuda:
            mask = mask + torch.triu(torch.ones(input_tensor_1, input_tensor_1).cuda(),
                                     diagonal=1).byte()
        else:
            mask = mask + torch.triu(t.ones(input_tensor_1, input_tensor_1),
                                     diagonal=1).repeat(batch_size, 1, 1).byte()
        mask = mask.gt(0)
        zero_mask = c_mask.eq(0).unsqueeze(-1).repeat(1, 1, input_tensor_1)
        zero_mask = zero_mask.transpose(1, 2)'''

        attn_dot_list = list()
        attn_dec_list = list()

        '''for i,(selfattn, dotattn, ffn) in enumerate(self.attention, self.attention, self.ffn):
            tensor, attn_dec = selfattn(tensor)
            tensor, attn_dot = dotattn(input_tensor)
            decoder_input = ffn(tensor)
            attn_dot_list.append(attn_dot)
            attn_dec_list.append(attn_dec)'''
        output_tensor, attn_dec = self.attention(tensor, tensor, None, None)
        tensor, attn_dot = self.attention(tensor, output_tensor, None, None)
        tensor = self.ffn(tensor)
        attn_dot_list.append(attn_dot)
        attn_dec_list.append(attn_dec)

        mel_out = self.mel_linear(tensor)

        stop_tokens = self.stop_linear(tensor)

        return mel_out, attn_dot_list, stop_tokens, attn_dec_list


class DummyModel(torch.nn.Module):
    def __init__(self, configs):
        super(DummyModel, self).__init__()
        self.configs = configs
        #self.embedding = DummyEmbedding(configs)
        #self.encoder_prenet = DummyPrenet(configs)

        self.encoder = Encoder(configs)
        self.decoder = Decoder(configs)

    def forward(self, input_tensor, mel_input):
        '''print("before embedding : ", input_tensor.shape)
        tensor = self.embedding(input_tensor)
        print("before encoder : ", tensor.shape)
        tensor = self.encoder_prenet(tensor)'''

        tensor, c_mask, attn_enc = self.encoder.forward(input_tensor.cuda())
        #log.debug("before decoder : ", tensor.shape)
        return self.decoder.forward(tensor, tensor, c_mask, mel_input.cuda())


'''if __name__ == '__main__':
    DummyModel()
    DummyAttention()
    DummyEmbedding()'''