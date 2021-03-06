import numpy as np
import os
import librosa

config_dict = dict()

config_dict.update({
    'dataset_path': './korean-single-speaker-speech-dataset', 
    'fs': 22050,
    'audio_dtype': np.int16,
    'audio_format': 'mono',
    'num_converting_thread': 16,
})

config_dict.update({
    'dataset_meta_path': os.path.join(config_dict['dataset_path'], 'transcript.v.1.4.txt')
})

config_dict.update({
    'dataset_meta_path_train': config_dict['dataset_meta_path'].replace('.txt', '_train.txt'),
    'dataset_meta_path_valid': config_dict['dataset_meta_path'].replace('.txt', '_valid.txt'),
    'valid_ratio': 0.002,
})

config_dict.update({
    'converted_dataset_path': f"{config_dict['dataset_path']}-{config_dict['fs']}", 
})

config_dict.update({
    'converted_meta_path': config_dict['dataset_meta_path'].replace(config_dict['dataset_path'], config_dict['converted_dataset_path']),
})

# WaveGlow Parameters
config_dict.update({
    # "segment_length": 16000,
    "sampling_rate": config_dict['fs'],
    # "filter_length": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "mel_fmin": 0.0,
    "mel_fmax": 8000.0,
    "n_mel_channels": 80,
})

config_dict.update({
    "nov_length": config_dict["win_length"] - config_dict['hop_length']
})

config_dict.update({
    "mel_basis": librosa.filters.mel(sr=config_dict['fs'], 
                                     n_fft=config_dict['win_length'], 
                                     n_mels=config_dict['n_mel_channels'], 
                                     fmin=config_dict['mel_fmin'],
                                     fmax=config_dict['mel_fmax']),
})

config_dict.update({
    "mel_min_val": np.exp(-12),
})

config_dict.update({
    "batch_size": 16,
})

config_dict.update({
    # "text_encoding_mode": 'jamo',
    "text_encoding_mode": 'phoneme',
})

config_dict.update({
    "mel_loading": False
})

config_dict.update({
    "embedding_dim": 512,
    "encoder_prenet_dim": 512,
})

config_dict.update({
    "num_hidden": 256,
    "multihead_num": 4,
    "mode": 'train',
})

print('Configuration Dictionary List')
for key in config_dict:
    if key == "mel_basis":
        print(f'{key:^30s} | {config_dict[key].shape}')
    else:
        print(f'{key:^30s} | {config_dict[key]}')
print()