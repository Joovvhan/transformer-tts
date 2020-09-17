import torch
from torch.nn import Conv1d, Conv2d
from torch.nn import BatchNorm1d, ReLU, Dropout
from torch.nn import ModuleList, Sequential
from torch import nn
from utils import ENCODING_SIZE
import numpy as np
import math
import copy

import logging

log = logging.getLogger('log')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)s] (%(filename)s:%(lineno)d) > %(message)s')
fileHandler = logging.FileHandler('./log.txt')
fileHandler.setFormatter(formatter)

log.addHandler(fileHandler)


def t_embedding(max_length, num_hidden):
    emb = np.array([[pos_i / np.power(10000, 2 * (hid_idx // 2) / num_hidden)
                     for hid_idx in range(num_hidden)]
                    for pos_i in range(max_length)])

    emb[:, 0::2] = np.sin(emb[:, 0::2])
    emb[:, 1::2] = np.cos(emb[:, 1::2])
    # return torch.from_numpy(emb).type(torch.FloatTensor)
    return torch.from_numpy(emb).type(torch.FloatTensor).requires_grad_(False)


def pos_emb(text_tensor):

    '''text_embedding = text_tensor.ne(0).type(torch.float)
    text_attention_embedding = text_tensor.eq(0)
    mel_embedding_dimension = mel_tensor.shape[1]
    mel_embedding = np.array([[[position / np.power(10000, 2 * i / mel_embedding_dimension)
            for i in range(mel_embedding_dimension)]
        if pos != 0 else np.zeros(mel_embedding_dimension)
        for pos in range(mel_tensor.shape[0])]
                              for position in range(mel_tensor.shape[2])
    ])
    mel_embedding[1:, 0::2] = np.sin(mel_embedding[1:, 0::2])  # dim 2i
    mel_embedding[1:, 1::2] = np.cos(mel_embedding[1:, 1::2])  # dim 2i+1'''
    text_attention_embedding = text_tensor.eq(0)
    #text_embedding = text_position_embedding(text_tensor)
    return text_attention_embedding


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
        self.layers = ModuleList([
            Conv1d(configs['num_hidden'], configs['num_hidden'],
                   kernel_size=5,
                   padding=int(np.floor(5 / 2))),
            BatchNorm1d(configs['num_hidden']),
            ReLU(),
            Dropout(p=0.2),
            Conv1d(configs['num_hidden'], configs['num_hidden'],
                   kernel_size=5,
                   padding=int(np.floor(5 / 2))),
            BatchNorm1d(configs['num_hidden']),
            ReLU(),
            Dropout(p=0.2)
        ])

    def forward(self, text_tensor):
        tensor = self.embed(text_tensor)
        tensor = tensor.transpose(1, 2)
        tensor = self.conv(tensor)
        tensor = self.batch_norm(tensor)
        tensor = self.dropout(tensor)
        for layer in self.layers:
            tensor = layer(tensor)
        tensor = tensor.permute(0, 2, 1)  # (B, H, L) -> (B, L, H)
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
            ReLU()
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
        # Input:  (N, C, L)
        # Output: (N, C, L)
        self.layer_norm = torch.nn.LayerNorm(self.num_hidden)
        # Input: (N, *)
        # Output: (N, *)

    def forward(self, input_tensor):
        tensor = input_tensor # (B, L, H)
        tensor = tensor.transpose(1, 2) # (B, L, H) -> (B, H, L)
        for layer in self.layers:
            tensor = layer(tensor)
        tensor = tensor.transpose(1, 2) # (B, H, L) -> (B, L, H)
        tensor = tensor + input_tensor
        tensor = self.layer_norm(tensor) 
        return tensor # (B, L, H)


class Attention(torch.nn.Module):


    def __init__(self, configs):
        super(Attention, self).__init__()
        self.configs = configs
        self.num_hidden = self.configs['num_hidden']
        self.multihead_num = self.configs['multihead_num']
        self.embedding_dim = configs['embedding_dim']
        self.multihead = torch.nn.MultiheadAttention(self.num_hidden, self.multihead_num)
        # torch.nn.MultiheadAttention(embed_dim, num_heads, 
        # dropout=0.0, bias=True, 
        # add_bias_kv=False, add_zero_attn=False, 
        # kdim=None, vdim=None)
        # query: (L, N, E)
        # key: (S, N, E)
        # value: (S, N, E)
        # attn_output: (L, N, E)
        # self.key = Linear(self.num_hidden, self.num_hidden, bias=False)
        # self.value = Linear(self.num_hidden, self.num_hidden, bias=False)
        # self.query = Linear(self.num_hidden, self.num_hidden, bias=False)
        self.normalization = torch.nn.LayerNorm(self.num_hidden)
        # Input: (N, *)
        # Output: (N, *)

    def forward(self, key_value_tensor, query_tensor, text_embedding=None, query_embedding=None):
        key, value, query = key_value_tensor, key_value_tensor, query_tensor
        # if text_embedding is not None:
        #     text_embedding = text_embedding.transpose(0, 1)

        # print(f' query: {query.shape} | key: {key.shape} | value: {value.shape} | text_embedding: {text_embedding.shape}')

        output_tensor, _ = self.multihead(query, key, value, key_padding_mask=text_embedding)
        output_tensor = output_tensor + query_tensor

        output_tensor = output_tensor.transpose(0, 1)     # (L, N, E) -> (N, L, E)

        output_tensor = self.normalization(output_tensor) # ('B', L, H)
        
        return output_tensor


class Encoder(torch.nn.Module):
    def __init__(self, configs):
        super(Encoder, self).__init__()
        self.configs = configs
        self.encoder_prenet = EncoderPrenet(configs)
        # self.position_embedding = torch.nn.Embedding.from_pretrained(t_embedding(configs['embedding_dim'],
        #                                                                          configs['num_hidden']))

        self.position_embedding = t_embedding(128, configs['num_hidden']).cuda()

        self.attention = torch.nn.ModuleList([Attention(configs) for _ in range(3)])
        self.ffn = torch.nn.ModuleList([FFN(configs) for i in range(3)])

    def forward(self, text_tensor, text_attention_emb=None):
        tensor = self.encoder_prenet(text_tensor)
        # position_embedding = torch.nn.Embedding.from_pretrained(t_embedding(text_tensor.shape[1], self.configs['num_hidden'])).cuda()
        # text_emb = position_embedding(text_tensor)

        # print(tensor.shape)                  # torch.Size([4, 46, 256])
        # print(self.position_embedding.shape) # torch.Size([512, 256])
        # tensor = tensor + text_emb[:tensor.shape[1], :]
        B, L, _ = tensor.shape

        # print(self.position_embedding[:L, :].repeat(B, 1, 1).shape)         # torch.Size([4, 57, 256])
        # print(self.position_embedding[:L, :].repeat(B, 1, 1).requires_grad) # False

        # print(f"{'Encoder Input':20} | {tensor.shape}") # torch.Size([4, 42, 256]
        # print(f"{'Positional Embedding':20} | {self.position_embedding[:L, :].repeat(B, 1, 1).shape}")
        # torch.Size([4, 42, 256]
        tensor = tensor + self.position_embedding[:L, :].repeat(B, 1, 1)

        # print(f"{'Encoder Input':20} | {tensor.shape}") # torch.Size([4, 42, 256]

        for attention_layer, ffn_layer in zip(self.attention, self.ffn):
            # tensor = attention_layer(tensor, tensor, text_embedding=text_attention_emb)
            tensor = tensor.transpose(0, 1) # (B, L, H) -> (L, B, H)
            tensor = attention_layer(tensor, tensor) # (B, L, H)
            tensor = ffn_layer(tensor) # (B, L, H)
        return tensor # (B, L, H)


class Decoder(torch.nn.Module):
    def __init__(self, configs):
        super(Decoder, self).__init__()
        self.configs = configs
        self.alpha = torch.nn.Parameter(torch.ones(1)).cuda()
        self.decoder_prenet = DecoderPrenet(configs)
        # self.attention = Attention(configs)
        # self.ffn = FFN(configs)
        self.attention_1 = torch.nn.ModuleList([Attention(configs) for _ in range(3)])
        self.attention_2 = torch.nn.ModuleList([Attention(configs) for _ in range(3)])
        self.ffn = torch.nn.ModuleList([FFN(configs) for i in range(3)])

        self.num_hidden = configs['num_hidden']
        self.num_mels = configs['n_mel_channels']
        self.mel_linear = Linear(self.num_hidden, self.num_mels)
        self.stop_linear = Linear(self.num_hidden, 1)

        self.position_embedding = t_embedding(1024, configs['num_hidden']).cuda()

    def forward(self, text_tensor, text_attention_emb, input_tensor, mel_tensor):
        output_tensor = self.decoder_prenet(mel_tensor.type(torch.cuda.FloatTensor))
        # print(output_tensor.shape) # torch.Size([4, 464, 256])

        B, L, _ = output_tensor.shape
        output_tensor = output_tensor + self.position_embedding[:L, :].repeat(B, 1, 1)

        '''position_embedding = torch.nn.Embedding.from_pretrained(t_embedding(mel_tensor.shape[1],
                                                                            self.configs['num_hidden'])).cuda()

        text_emb = position_embedding(mel_tensor[:, :].type(torch.LongTensor).cuda())
        print(mel_tensor.shape, output_tensor.shape, text_emb.shape)
        output_tensor = text_emb + output_tensor'''
        # input_tensor = input_tensor.transpose(0, 1)
        text_tensor = text_tensor.transpose(0, 1) # (B, L, H) -> (L, B, H)

        for attention_layer_1, attention_layer_2, ffn_layer in zip(self.attention_1, self.attention_2, self.ffn):

            output_tensor = output_tensor.transpose(0, 1) # (B, T, 80) -> (T, B, 80)
            masked_tensor = attention_layer_1(output_tensor, output_tensor) # (B, T, 80)
            masked_tensor = masked_tensor.transpose(0, 1) # (B, T, 80) -> # (T, B, 80)
            # output_tensor = self.attention(input_tensor, masked_tensor, text_embedding=None)
            output_tensor = attention_layer_2(text_tensor, masked_tensor, text_embedding=None) # (B, T, 80)

            output_tensor = ffn_layer(output_tensor) # (B, L, H)
            # output_tensor = output_tensor.transpose(0, 1)
            # print(output_tensor.shape) # torch.Size([4, 405, 256])
        mel_out = self.mel_linear(output_tensor)

        stop_tokens = self.stop_linear(output_tensor)

        return mel_out, stop_tokens


class Model(torch.nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.encoder = Encoder(configs)
        self.decoder = Decoder(configs)

    def forward(self, text_tensor, mel_tensor):

        # text_tensor (B, L, H)
        # mel_tensor (B, T, 80)

        # text_attention_emb = pos_emb(text_tensor.cuda())
        # output_tensor = self.encoder.forward(text_tensor.cuda(), text_attention_emb)
        output_tensor = self.encoder.forward(text_tensor.cuda())

        # print(f"{'Encoder Output':20} | {output_tensor.shape}") # torch.Size([4, 42, 256]

        # return self.decoder.forward(text_tensor.cuda(), text_attention_emb, output_tensor,
        #                             mel_tensor.cuda())
        return self.decoder.forward(output_tensor.cuda(), None, None, mel_tensor.cuda())
