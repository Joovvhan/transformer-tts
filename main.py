from datetime import datetime
from tqdm import tqdm
import argparse

from settings import configs
from utils import dataset
from utils.dataset import prepare_data_loaders, get_data_loaders, PHONEME_DICT
from model import DummyModel as Model
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


MODEL_FOLDER = "./logs"
MODEL_NAME = "test_model1"
PATH = MODEL_FOLDER + "/" + MODEL_NAME
writer = SummaryWriter(PATH)

def main():

    #prepare_data_loaders(configs)

    cuda = torch.device('cuda')
    step = 0

    for epoch in range(10):

        train_data_loader, valid_data_loader = get_data_loaders(configs)

        t0 = datetime.now()

        model = Model(configs).cuda()

        for i, data in enumerate(train_data_loader):
            step += 1
            path_list, mel_batch, encoded_batch, text_list, mel_length_list, encoded_length_list = data

            mel_out, attn_dot_list, stop_tokens, attn_dec_list = model(torch.tensor(encoded_batch),
                                                                       torch.tensor(mel_batch))
            writer.add_scalar('loss', nn.L1Loss()(mel_out.cuda(), mel_batch.cuda()).item(), step)
            print(nn.L1Loss()(mel_out.cuda(), mel_batch.cuda()).item())

        '''for batch in tqdm(train_data_loader):
            # batch = (path_list, mel_batch, encoded_batch, text_list, mel_length_list, encoded_length_list)
            # Check collate_function in utils/dataset.py
            path_list, mel_batch, encoded_batch, text_list, mel_length_list, encoded_length_list = batch

            mel_out, attn_dot_list, stop_tokens, attn_dec_list = model(torch.tensor(encoded_batch), torch.tensor(mel_batch))
            writer.add_scalars('loss', mel_out.cuda() - mel_batch.cuda())
            #print(mel_out.cuda() - mel_batch.cuda())


        t1 = datetime.now()

        for batch in tqdm(valid_data_loader):

            path_list, mel_batch, encoded_batch, text_list, mel_length_list, encoded_length_list = batch
            mel_out, attn_dot_list, stop_tokens, attn_dec_list = model(torch.tensor(encoded_batch), torch.tensor(mel_batch))

        t2 = datetime.now()

        print(t1 - t0, t2 - t1)'''

    torch.save(model, PATH)

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str,
                        help='Training or Inference Mode')

    args = parser.parse_args()

    # I should overwrite configs with parsed arguments
    
    main()