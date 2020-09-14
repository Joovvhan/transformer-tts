import os
import random
import shutil
from glob import glob
from tqdm import tqdm

import numpy as np

import scipy.io.wavfile as wavfile
import librosa

import itertools

from concurrent.futures import ProcessPoolExecutor

KSS_PATH = 'korean-single-speaker-speech-dataset'  # wav
SR = 22050
NEW_DATASET_PATH = f'{KSS_PATH}-{SR}'

NUM_WORKERS = 8

TEST_RATIO = 0.02

def resample_wav(file, new_file, sr=22050, scale=2 ** 15):
    if not os.path.isfile(new_file):
        data, fs = librosa.core.load(file, sr=sr, mono=True)
        int_16_data = (data * scale).astype(np.int16)
        os.makedirs(os.path.dirname(new_file), exist_ok=True)
        wavfile.write(new_file, sr, int_16_data)

        return True

    return False

def resample_kss(file, meta, sr=22050, scale=2 ** 15):
    new_file = file.replace(KSS_PATH, NEW_DATASET_PATH)

    resample_wav(file, new_file, sr, scale)

    # ['1/1_0000.wav', '그는 괜찮은 척하려고 애쓰는 것 같았다.', '그는 괜찮은 척하려고 애쓰는 것 같았다.', '그는 괜찮은 척하려고 애쓰는 것 같았다.', '3.5', 'He seemed to be pretending to be okay.']
    file_id = meta[0]
    script = meta[2]
    speaker_id = 'kss'

    if file_id in new_file:
        new_meta = (new_file, script, speaker_id)
    else:
        print(f"Metadata mismatch! {new_file} / {meta}")

    return new_meta

def preprocess_kss(dataset_path, sr):
    print("Searching KSS wav files")

    with open(os.path.join(dataset_path, 'transcript.v.1.4.txt'), 'r') as file:
        raw_metadata = [line.strip().split('|') for line in file]

    wav_files = sorted(glob(os.path.join(dataset_path, 'kss/*/*.wav')))
    print(f'The number of KSS wav files: {len(wav_files)}')
    print(wav_files[0])
    print(f'The number of KSS meta files: {len(raw_metadata)}')
    print(raw_metadata[0])

    scale = 2 ** 15

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        metadata_list = list(tqdm(pool.map(resample_kss,
                                           wav_files,
                                           raw_metadata,
                                           itertools.cycle([sr]),
                                           itertools.cycle([scale])),
                                  total=len(wav_files)))

    num_test = int(TEST_RATIO * len(metadata_list))
    random.shuffle(metadata_list)
    metadata_list_test = sorted(metadata_list[:num_test], key=lambda meta: meta[0])
    metadata_list_train = sorted(metadata_list[num_test:], key=lambda meta: meta[0])

    print()

    return metadata_list_train, metadata_list_test


if __name__ == "__main__":
    os.makedirs(NEW_DATASET_PATH, exist_ok=True)
    shutil.copy(os.path.join(KSS_PATH, 'transcript.v.1.4.txt'), os.path.join(NEW_DATASET_PATH, 'transcript.v.1.4.txt'))
    meta_kss_train, meta_kss_test = preprocess_kss(KSS_PATH, SR)
