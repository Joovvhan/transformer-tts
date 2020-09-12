import jamotools
import os

import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from settings import configs

'''
Hangul Jamo: Consist of Choseong, Jungseong, Jongseong. It is divided mordern Hangul and old Hangul that does not use in nowadays. Jamotools supports modern Hangul Jamo area.
1100 ~ 1112 (Choseong)
1161 ~ 1175 (Jungseong)
11A8 ~ 11C2 (Jongseong)
'''

PAD = '_'

encodable_chars = list()
encodable_chars.extend([PAD, '.', ',', '?', '!', ' '])

if configs['text_encoding_mode'] == 'jamo':
    encodable_chars.extend([chr(char) for char in range(0x1100, 0x1112 + 1)])
    encodable_chars.extend([chr(char) for char in range(0x1161, 0x1175 + 1)])
    encodable_chars.extend([chr(char) for char in range(0x11A8, 0x11C2 + 1)])
elif configs['text_encoding_mode'] == 'phoneme':
    encodable_chars.extend([chr(char) for char in range(0x1100, 0x1112 + 1)])
    encodable_chars.extend([chr(char) for char in range(0x1161, 0x1175 + 1)])
    encodable_chars.extend([chr(char) for char in [0x11a8, 0x11ab, 0x11ae, 0x11af, 0x11b7, 0x11b8, 0x11bc]]) # ㄱ, ㄴ, ㄷ, ㄹ, ㅁ, ㅂ, ㅇ

encoding_dict = dict()
for i, char in enumerate(encodable_chars):
    encoding_dict[char] = i

decoding_dict = dict()
for key in encoding_dict:
    decoding_dict[encoding_dict[key]] = key

ENCODING_SIZE = len(encodable_chars)

def text2encoding(text):
    text = jamotools.split_syllables(text, jamo_type="JAMO")
    return [encoding_dict[char] for char in text]

def encoding2text(encoding):
    return ''.join([decoding_dict[num] for num in encoding])

def jamo2text(jamo):
    return jamotools.join_jamos(jamo)

print(f'Encodable Character List (#{len(encoding_dict)})')
for key in encoding_dict:
    print((key, hex(ord(key)), encoding_dict[key]), end=' ')
print('\n')


