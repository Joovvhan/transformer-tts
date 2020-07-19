from torch.utils.data import Dataset, DataLoader
import torch
from collections import namedtuple
import os
import scipy.io.wavfile as wavfile
import numpy as np
from tqdm import tqdm
import librosa
import torch

import shutil

import threading

from time import sleep

from scipy import signal

import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from settings import configs

FTPair = namedtuple('FileTextPair', ['file_path', 'text'])

class AudioTextDataset(Dataset):

    def __init__(self, meta_file_path, configs):# , transform=None):
        self.file_text_pair_list = load_file_text_pair_list(meta_file_path)
        self.configs = configs
        # self.transform = transform

    def __len__(self):
        return len(self.file_text_pair_list)

    def __getitem__(self, idx):

        file_path, text = self.file_text_pair_list[idx]

        fs, audio = wavfile.read(file_path)

        if configs['audio_dtype'] == np.int16:
            audio = audio / 2 ** 15
        elif configs['audio_dtype'] == np.int32:
            audio = audio / 2 ** 31
        else:
            assert False, f"Unknown audio_dtype {configs['audio_dtype']}"

        assert fs == configs['fs'], f"{file_path} sampling rate does not match {fs} != {configs['fs']}"
        
        mel = audio2mel(audio, self.configs)

        return (file_path, mel, text)

def load_file_text_pair_list(meta_file_path, meta_type='kss'):

    dataset_dir = os.path.dirname(meta_file_path)

    if meta_type == 'kss':

        dataset_dir = os.path.join(dataset_dir, 'kss')

        file_text_pair_list = list()
        
        with open(meta_file_path, 'r', encoding='UTF-8') as file:

            for line in file:
                line_segments = line.split('|')
                file_text_pair_list.append(FTPair(os.path.join(dataset_dir, line_segments[0]), 
                line_segments[3]))
        
        return file_text_pair_list

    else:
        print(f"Unknown data type: {meta_type}")
        return None

def check_file_existence(file_text_pair_list):

    print('* Wav file existence check started')

    invalid_path_count = 0
    
    for file_text_pair in file_text_pair_list:
        file_path = getattr(file_text_pair, 'file_path')
        if not os.path.isfile(file_path):
            invalid_path_count += 1
            print(f'{file_path} does not exist')

            if invalid_path_count >= 10:
                print(f'* Over {invalid_path_count} was missing. Pause check_file_existence')
                break

    if invalid_path_count == 0:
        print(f'* Wav file existence check was successful Failure/Success = {invalid_path_count}/{len(file_text_pair_list)}', end='\n\n')

    return

def check_configuration_mismatch(fs, dtype, audio_shape, configs):

    fs_mismatch = (fs != configs['fs'])
    dtype_mismatch = (dtype != configs['audio_dtype'])
    if configs['audio_format'] == 'mono':
        shape_mismatch = (len(audio_shape) >= 2 and audio_shape[1] != 1)
    else:
        assert False, f"Unknown audio format {configs['audio_format']}"

    return (fs_mismatch, dtype_mismatch, shape_mismatch)

def check_wavfile_configuration(file_text_pair_list):
    
    '''
    Check followings formats
    1. Sampling Rate
    2. dtype
    3. mono
    '''

    print('* Wav file format check started')

    invalid_format_count = 0
    
    for file_text_pair in file_text_pair_list:
        file_path = getattr(file_text_pair, 'file_path')

        fs, audio = wavfile.read(file_path) # (T, channel)

        mismatches = check_configuration_mismatch(fs, audio.dtype, audio.shape, configs)
        
        if any(mismatches):
            invalid_format_count += 1
            print(f'* {file_path}:', end=' ')
            if mismatches[0]:
                print(f"fs {fs} != {configs['fs']}", end=' ')
            if mismatches[1]:
                print(f"dtype {audio.dtype} != {configs['audio_dtype']}", end=' ')
            if mismatches[2]:
                print(f"audio_format {audio.shape} != {configs['audio_format']}", end=' ')
            print()

        if invalid_format_count >= 10:
            print(f'Over {invalid_format_count} files mismatched. Check the file format and the configuration', end='\n\n')
            break

    if invalid_format_count == 0:
        print(f'* Wav file format check was successful Failure/Success = {invalid_format_count}/{len(file_text_pair_list)}')
        return True

    return False

def wav_converting_thread(file_path, new_file_path, multiplier, configs):
    y, sr = librosa.core.load(file_path, sr=configs['fs']) # dtype=configs['audio_dtype'])
    y *= multiplier
    y = y.astype(configs['audio_dtype'])
    try:
        wavfile.write(new_file_path, sr, y)
    except FileNotFoundError:
        dir_name = os.path.dirname(new_file_path)
        os.makedirs(dir_name, exist_ok=True)
        wavfile.write(new_file_path, sr, y)
    
    return

def convert_dataset_as_configuration(file_text_pair_list, configs):

    print('* Dataset conversion started')

    conversion_count = 0
    exist_count = 0
    threads = list()

    if os.path.isfile(configs['converted_meta_path']):
        print(f"* Converted meta file {configs['converted_meta_path']} exists")
        # print('* Skipping conversion', end='\n\n')
        # return True

    if configs['audio_dtype'] == np.int16:
        multiplier = 2 ** 15
    elif configs['audio_dtype'] == np.int32:
        multiplier = 2 ** 31
    else:
        assert False, f"Invalid audio data type {configs['audio_dtype']}"

    for file_path, text in tqdm(file_text_pair_list):
        new_file_path = file_path.replace(configs['dataset_path'], 
                                        configs['converted_dataset_path'])
        if os.path.isfile(new_file_path):
            exist_count += 1
            pass
        else:

            while len(threads) >= configs['num_converting_thread']:
                threads = [t for t in threads if t.is_alive()]
                sleep(0.1)

            t = threading.Thread(target=wav_converting_thread, args=(file_path, new_file_path, multiplier, configs))
            t.start()
            threads.append(t)
            conversion_count += 1

    for t in threads:
        t.join()

    shutil.copyfile(configs['dataset_meta_path'], configs['converted_meta_path'])
    
    print('* Conversion finished [(Converted + Existing)/Total = ' + \
         f'({conversion_count} + {exist_count} = {conversion_count + exist_count})/{len(file_text_pair_list)}]')

    return True

def audio2mel(audio, configs):

    print()

    f, t, Zxx = signal.stft(audio, 
                      fs=configs['sampling_rate'], 
                      nperseg=configs['win_length'], 
                      noverlap=configs['nov_length'])
    Sxx = np.abs(Zxx)

    mel = configs['mel_basis'] @ Sxx
    log_mel = np.log(np.maximum(mel, configs['mel_min_val']))

    print(audio.shape, Zxx.shape, configs['mel_basis'].shape, Sxx.shape, log_mel.shape)

    return log_mel

if __name__ == '__main__':
    file_text_pair_list = load_file_text_pair_list(configs['dataset_meta_path'])
    check_file_existence(file_text_pair_list)
    proper_configuration = check_wavfile_configuration(file_text_pair_list)

    if not proper_configuration:
        convert_dataset_as_configuration(file_text_pair_list, configs)
        configs['dataset_meta_path'] = configs['converted_meta_path']
        configs['dataset_path'] = configs['converted_dataset_path']
    
    dataset = AudioTextDataset(configs['dataset_meta_path'], configs)
    
    print(file_text_pair_list[0], len(file_text_pair_list))

    for i in range(10):

        path, mel, text = dataset[i]

        print(path, text, configs['dataset_meta_path'])

        mel_spectrogram_path = path.replace(configs['dataset_path'], './mel_spectrograms') + '.pt'

        mel_tensor = torch.tensor(mel + 6)
        
        try:
            torch.save(mel_tensor, mel_spectrogram_path)
        except FileNotFoundError:
            dir_name = os.path.dirname(mel_spectrogram_path)
            os.makedirs(dir_name, exist_ok=True)
            torch.save(mel_tensor, mel_spectrogram_path)


