import torch
from torch.nn import Conv1d, Conv2d, Linear
from torch.nn import BatchNorm1d, ReLU, Dropout
from torch.nn import ModuleList, Sequential
from torch import nn
from utils import ENCODING_SIZE
import numpy as np

import logging

log = logging.getLogger('log')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)s] (%(filename)s:%(lineno)d) > %(message)s')
fileHandler = logging.FileHandler('./log.txt')
fileHandler.setFormatter(formatter)

log.addHandler(fileHandler)


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


class DummyPrenetModule(torch.nn.Module):

    '''
     Input: (B, H, L)
    Output: (B, H, L)
    '''

    def __init__(self, configs, dropout_rate=0.2):
        super(DummyPrenetModule, self).__init__()
        self.configs = configs

        '''
        From Tacotron2:
        The network is composed of an encoder and a decoder with atten-
        tion. The encoder converts a character sequence into a hidden feature 
        representation which the decoder consumes to predict a spectrogram. 
        Input characters are represented using a learned 512-dimensional character embedding, 
        which are passed through a stack of 3 convolutional layers each containing 512 filters 
        with shape 5 × 1, i.e., where each filter spans 5 characters, followed by batch normalization [18] 
        and ReLU activations.
        '''

        self.layers = ModuleList([
            Conv1d(self.configs['embedding_dim'],
                   self.configs['embedding_dim'], 5, padding=2),
            BatchNorm1d(self.configs['embedding_dim']),
            ReLU(),  
            Dropout(dropout_rate),         
        ])

    def forward(self, input_tensor):

        tensor = input_tensor

        for layer in self.layers:
            tensor = layer(tensor)
        
        output_tensor = tensor

        return output_tensor


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

        tensor = self.pernet_linear(tensor) # (B, L, H) -> (B, L, H)

        output_tensor = tensor
        log.info("prenet forward ")
        
        return output_tensor


class EncoderPrenet(torch.nn.Module):
    def __init__(self, configs):
        super(EncoderPrenet, self).__init__()
        self.configs = configs
        self.embedding_dim = configs['embedding_dim']
        self.embed = torch.nn.Embedding(ENCODING_SIZE, configs['embedding_dim'])
        self.prenet_layers = Conv1d(configs['embedding_dim'], configs['num_hidden'], kernel_size=5,
                                    padding=int(np.floor(5/2)))
        self.batch_norm = torch.nn.BatchNorm1d(configs['num_hidden'])
        self.dropout = torch.nn.Dropout(p=0.2)
        self.prenet_linear = Linear(configs['num_hidden'], configs['num_hidden'])

    def forward(self, input_tensor):
        tensor = self.embed(input_tensor)
        tensor = tensor.transpose(1, 2)
        #log.debug("transpose : ", tensor.shape)
        #for layer in self.prenet_layers:
        tensor = self.prenet_layers(tensor)  # (B, H, L) -> (B, H, L)
        tensor = self.batch_norm(tensor)
        tensor = self.dropout(tensor)
        #log.debug("prenet layer : ", tensor.shape)
        tensor = tensor.permute(0, 2, 1)  # (B, H, L) -> (B, L, H)
        #log.debug("transpose : ", tensor.shape)
        tensor = self.prenet_linear(tensor)  # (B, L, H) -> (B, L, H)
        #log.debug("prenet linear : ", tensor.shape)
        output_tensor = tensor
        log.info('encoder prenet ')
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
            #패딩 디폴트 0이다 넣어fk
            # https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html
            Conv1d(self.num_hidden,
                   self.num_hidden * 4, 5),
            ReLU(),
            Conv1d(self.num_hidden * 4,
                   self.num_hidden, 5)
        ])
        self.layer_norm = torch.nn.LayerNorm(self.num_hidden)

    def forward(self, input_tensor):
        tensor = input_tensor
        tensor = tensor.transpose(1,2)
        for layer in self.layers:
            tensor = layer(tensor)
        tensor = tensor.transpose(1,2)
        tensor = self.layer_norm(tensor)
        log.info("FFN tensor ")
        return tensor


class DummyAttention(torch.nn.Module):
    def __init__(self, configs):
        super(DummyAttention, self).__init__()
        self.configs = configs
        self.num_hidden = self.configs['num_hidden']
        self.multihead_num = self.configs['multihead_num']
        self.embedding_dim = configs['embedding_dim']
        self.multihead = torch.nn.MultiheadAttention(configs['num_hidden'], self.multihead_num)
        self.key = Linear(self.num_hidden, self.num_hidden, bias=False)
        self.value = Linear(self.num_hidden, self.num_hidden, bias=False)
        self.query = Linear(self.num_hidden, self.num_hidden, bias=False)

    def forward(self, input_tensor):
        input_tensor_0 = input_tensor.size(0)
        input_tensor_1 = input_tensor.size(1)

        key = self.key(input_tensor).view(input_tensor_0,
                                          input_tensor_1,
                                          self.num_hidden)
        value = self.value(input_tensor).view(input_tensor_0,
                                          input_tensor_1,
                                          self.num_hidden)
        '''value = self.value(input_tensor).view(input_tensor_0,
                                          input_tensor_1,
                                          self.multihead_num,
                                          self.num_hidden // self.multihead_num)'''
        query = self.query(input_tensor).view(input_tensor_0,
                                          input_tensor_1,
                                          self.num_hidden)
        return self.multihead(key, value, query)


class Encoder(torch.nn.Module):
    def __init__(self, configs):
        super(Encoder, self).__init__()
        self.configs = configs
        self.encoder_prenet = EncoderPrenet(configs)
        self.attention = DummyAttention(configs)
        self.ffn = FFN(configs)

    def forward(self, input_tensor):
        tensor = self.encoder_prenet(input_tensor)
        attns = list()
        tensor, attn = self.attention(tensor)
        tensor = self.ffn(tensor)
        attns.append(attn)
        '''for layer, ffn in zip(self.attention, self.ffn):
            tensor, attn = layer(tensor)
            tensor = ffn(tensor)
            attns.append(tensor)'''
        return tensor, attns


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

    def forward(self, input_tensor, mel_input):
        input_tensor_0 = input_tensor.size(0)
        input_tensor_1 = input_tensor.size(1)
        tensor = self.decoder_prenet(mel_input.type(torch.cuda.FloatTensor))

        attn_dot_list = list()
        attn_dec_list = list()

        '''for i,(selfattn, dotattn, ffn) in enumerate(self.attention, self.attention, self.ffn):
            tensor, attn_dec = selfattn(tensor)
            tensor, attn_dot = dotattn(input_tensor)
            decoder_input = ffn(tensor)
            attn_dot_list.append(attn_dot)
            attn_dec_list.append(attn_dec)'''
        tensor, attn_dec = self.attention(tensor)
        tensor, attn_dot = self.attention(tensor)
        decoder_input = self.ffn(tensor)
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

        tensor, attn_enc = self.encoder.forward(input_tensor.cuda())
        #log.debug("before decoder : ", tensor.shape)
        return self.decoder.forward(tensor, mel_input.cuda())


'''if __name__ == '__main__':
    DummyModel()
    DummyAttention()
    DummyEmbedding()'''