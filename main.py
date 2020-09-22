from datetime import datetime
from tqdm import tqdm
import argparse

from settings import configs
from utils import dataset
from utils import util, jamo2text
from utils.dataset import prepare_data_loaders, get_data_loaders, PHONEME_DICT
from model import Model as Model
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = 'NanumGothic'
plt.rcParams["font.size"] = 14

import numpy as np

import secrets
hex_token = secrets.token_hex(8)

MODEL_FOLDER = "./logs"
# MODEL_NAME = "xavier_test2"
MODEL_NAME = hex_token
PATH = MODEL_FOLDER + "/" + MODEL_NAME
writer = SummaryWriter(PATH)

LOGGING_STEPS = 40

def matrix_to_plt_image(data_matrix, text):
    fig, ax = plt.subplots()
    im = ax.imshow(data_matrix) #, origin="reversed", aspect="auto")
    im.set_clim(-6, 0)
    plt.colorbar(im, ax=ax)
    plt.xlabel(jamo2text(text))
    plt.tight_layout()
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return img


def train(args):
    #prepare_data_loaders(configs)
    cuda = torch.device('cuda')
    step = 0
    loss_list = list()

    plt.rc('font', family='Malgun Gothic')
    model = Model(configs).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    if args.load != '':
        model, optimizer, step = util.load_model(args.load, model, optimizer)

    util.mkdir(args.save)

    for epoch in range(100):

        train_data_loader, valid_data_loader = get_data_loaders(configs)

        for i, data in tqdm(enumerate(train_data_loader), total=int(len(train_data_loader.dataset) / train_data_loader.batch_size)):
            step += 1
            path_list, mel_batch, encoded_batch, text_list, mel_length_list, encoded_length_list = data

            # mel_out, stop_tokens = model(torch.tensor(encoded_batch), torch.tensor(mel_batch))
            mel_out, stop_tokens, enc_attention, dec_attention = model(encoded_batch, mel_batch)
            loss = nn.L1Loss()(mel_out.cuda(), mel_batch.cuda())
            loss_list.append(loss.item())
            if step % LOGGING_STEPS == 0:
                writer.add_scalar('loss', np.mean(loss_list), step)
                writer.add_text('script', text_list[0], step)
                # writer.add_image('mel_in', torch.transpose(mel_batch[:1], 1, 2), step)  # (1, 80, T)
                # writer.add_image('mel_out', torch.transpose(mel_out[:1], 1, 2), step)  # (1, 80, T)
                #attention_image = matrix_to_plt_image(enc_attention[0].cpu().detach().numpy().T, text_list[0])
                #writer.add_image('attention', attention_image, step, dataformats="HWC")
                for i, prob in enumerate(enc_attention):

                    for j in range(4):
                        x = torchvision.utils.make_grid(prob[j*4] * 255)
                        writer.add_image('ENC_Attention_%d_0' % step, x, step)
                print(dec_attention[0].shape)
                for i, prob in enumerate(dec_attention):
                    for j in range(4):
                        x = torchvision.utils.make_grid(prob[j * 4] * 255)
                        writer.add_image('DEC_Attention_%d_0' % step, x, step)

                image = matrix_to_plt_image(mel_batch[0].cpu().detach().numpy().T, text_list[0])
                writer.add_image('mel_in', image, step, dataformats="HWC")  # (1, 80, T)

                image = matrix_to_plt_image(mel_out[0].cpu().detach().numpy().T, text_list[0])
                writer.add_image('mel_out', image, step, dataformats="HWC")  # (1, 80, T)

                # print(torch.min(mel_batch), torch.max(mel_batch))
                # print(torch.min(mel_out), torch.max(mel_out))

                # AssertionError: size of input tensor and input format are different.
                # tensor shape: (578, 80), input_format: CHW

                # print(mel_batch.shape)
                # print(mel_out.shape)            # torch.Size([24, 603, 80])   # B = 24
                # print(attn_dot_list[0].shape)   # torch.Size([96, 603, 603])  # 96 = B * 4 (num att. heads)
                # print(attn_dec_list[0].shape)   # torch.Size([96, 603, 603])
                # https://tutorials.pytorch.kr/intermediate/tensorboard_tutorial.html
                # https://www.tensorflow.org/tensorboard/image_summaries


                util.save_model(model, optimizer, args.save, step)
                loss_list = list()
            # print(nn.L1Loss()(mel_out.cuda(), mel_batch.cuda()).item())
            optimizer.zero_grad()
            loss.backward()

            # YUNA! Do not miss the gradient update!
            # https://tutorials.pytorch.kr/beginner/pytorch_with_examples.html
            optimizer.step()

            # break

        loss_list_test = list()
        for i, data in tqdm(enumerate(valid_data_loader), total=int(len(valid_data_loader.dataset) / valid_data_loader.batch_size)):
            path_list, mel_batch, encoded_batch, text_list, mel_length_list, encoded_length_list = data
            mel_out, stop_tokens = model(encoded_batch, mel_batch)
            loss = nn.L1Loss()(mel_out.cuda(), mel_batch.cuda())
            loss_list_test.append(loss.item())

        writer.add_scalar('loss_valid', np.mean(loss_list), step)
        writer.add_text('script_valid', text_list[0], step)

        image = matrix_to_plt_image(mel_batch[0].cpu().detach().numpy().T, text_list[0])
        writer.add_image('mel_in_valid', image, step, dataformats="HWC")  # (1, 80, T)

        image = matrix_to_plt_image(mel_out[0].cpu().detach().numpy().T, text_list[0])
        writer.add_image('mel_out_valid', image, step, dataformats="HWC")  # (1, 80, T)

        loss_list_test = list()
        for i, data in tqdm(enumerate(valid_data_loader), total=int(len(valid_data_loader.dataset) / valid_data_loader.batch_size)):
            path_list, mel_batch, encoded_batch, text_list, mel_length_list, encoded_length_list = data
            zero_batch = torch.zeros_like(mel_batch)
            mel_out, stop_tokens = model(encoded_batch, zero_batch)
            loss = nn.L1Loss()(mel_out.cuda(), mel_batch.cuda())
            loss_list_test.append(loss.item())

        writer.add_scalar('loss_infer', np.mean(loss_list), step)
        writer.add_text('script_infer', text_list[0], step)

        image = matrix_to_plt_image(mel_batch[0].cpu().detach().numpy().T, text_list[0])
        writer.add_image('mel_in_infer', image, step, dataformats="HWC")  # (1, 80, T)

        image = matrix_to_plt_image(mel_out[0].cpu().detach().numpy().T, text_list[0])
        writer.add_image('mel_out_infer', image, step, dataformats="HWC")  # (1, 80, T)

        # break

    # torch.save(model, PATH)

    return


def inference(args):
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str,
                        help='Training or Inference Mode')
    parser.add_argument('--save', default='test1', type=str,
                        help='Save model')
    parser.add_argument('--load', default='', type=str,
                        help='load model')

    args = parser.parse_args()
    # I should overwrite configs with parsed arguments
    if args.mode == 'train':
        train(args)
    else:
        inference(args)