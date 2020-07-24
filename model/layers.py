import torch
from torch.nn import Conv1d, Conv2d, Linear
from torch.nn import BatchNorm1d, ReLU, Dropout
# from torch.nn.functional import relu
from torch.nn import ModuleList, Sequential
from torch import nn
from utils import ENCODING_SIZE


class DummyEmbedding(torch.nn.Module):

    '''
     Input: (B, L)
    Output: (B, L, H)
    '''

    def __init__(self, num_embedding, embedding_dim):
        super(DummyEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(num_embedding, embedding_dim, padding_idx=0)

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
            Conv1d(512, 512, 5, padding=2),
            BatchNorm1d(512),
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
        # print(tensor.shape)
        tensor = tensor.permute(0, 2, 1) # (B, H, L) -> (B, L, H)

        tensor = self.pernet_linear(tensor) # (B, L, H) -> (B, L, H)

        output_tensor = tensor
        
        return output_tensor


class DummyAttention(torch.nn.Module):
    def __init__(self, configs):
        super(DummyAttention, self).__init__()

    def forward(self, input_tensor):

        return output_tensor


class DummyModel(torch.nn.Module):
    def __init__(self, configs):
        super(DummyModel, self).__init__()
        self.configs = configs
        self.embedding = DummyEmbedding(ENCODING_SIZE, configs["embbeding_dim"])
        self.encoder_prenet = DummyPrenet(configs)

    def forward(self, input_tensor):
        tensor = self.embedding(input_tensor)
        tensor = self.encoder_prenet(tensor)

        output_tensor = tensor

        return output_tensor

if __name__ == '__main__':
    DummyModel()
    DummyAttention()
    DummyEmbedding()