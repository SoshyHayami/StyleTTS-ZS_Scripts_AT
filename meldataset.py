#coding: utf-8
import os
import os.path as osp
import time
import random
import numpy as np
import random
import soundfile as sf
import librosa

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import os
import os.path as osp
import pandas as pd

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
        print(len(dicts))
    def __call__(self, text):
        indexes = []
        # for char in text:
        #     try:
        #         indexes.append(self.word_index_dictionary[char])
        #     except KeyError:
        #         # print(text)
        return indexes

np.random.seed(1)
random.seed(1)
SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
}

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 sr=24000,
                 data_augmentation=False,
                 validation=False,
                 ):

        spect_params = SPECT_PARAMS
        mel_params = MEL_PARAMS

        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [data if len(data) == 4 else (*data, 0) for data in _data_list]
        self.text_cleaner = TextCleaner()
        self.sr = sr

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        
#         self.global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):        
        data = self.data_list[idx]
        path = data[0]
        try:
            wave, text_tensor, speaker_id = self._load_tensor(data)
        except:
            print(path)
            return self.__getitem__(idx - 1 if idx > 1 else len(self.data_list) - 1)
        if 'LibriTTS_R' in path:
            wave = wave[..., :-50]
        
        mel_tensor = preprocess(wave).squeeze()
        
        try:
            if bool(random.getrandbits(1)):
                adj_path = data[1]
                adj_wave, sr = sf.read(adj_path)
                if adj_wave.shape[-1] == 2:
                    adj_wave = wave[:, 0].squeeze()
                if sr != 24000:
                    adj_wave = librosa.resample(adj_wave, sr, 24000)
                    print(adj_path, sr)

                adj_wave = np.concatenate([np.zeros([5000]), adj_wave, np.zeros([5000])], axis=0)

                mel_adj = preprocess(adj_wave).squeeze()
            else:
                mel_adj = mel_tensor
        except:
            mel_adj = mel_tensor
#         if (text_tensor.size(0)+1) >= (mel_tensor.size(1) // 3):
#             mel_tensor = F.interpolate(
#                 mel_tensor.unsqueeze(0), size=(text_tensor.size(0)+1)*3, align_corners=False,
#                 mode='linear').squeeze(0)
        
        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]
        
        acoustic_adj = mel_adj.squeeze()
        length_adj = acoustic_adj.size(1)
        acoustic_adj = acoustic_adj[:, :(length_adj - length_adj % 2)]
        
        try:
            ref_data = random.choice(self.data_list)
            ref_mel_tensor, ref_label = self._load_data(ref_data)
        except:
            # print(ref_data[0])
            return self.__getitem__(idx - 1 if idx > 1 else len(self.data_list) - 1)
        
        return speaker_id, acoustic_feature, text_tensor, ref_mel_tensor, ref_label, path, wave, acoustic_adj

    def _load_tensor(self, data):
        wave_path, _, text, speaker_id = data
        speaker_id = int(speaker_id)
        wave, sr = sf.read(wave_path)
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 24000:
            wave = librosa.resample(wave, sr, 24000)
            print(wave_path, sr)
            
        wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
        
        text = self.text_cleaner(text)
        
        text.insert(0, 0)
        text.append(0)
        
        text = torch.LongTensor(text)

        return wave, text, speaker_id

    def _load_data(self, data):
        wave, text_tensor, speaker_id = self._load_tensor(data)
        mel_tensor = preprocess(wave).squeeze()

        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, speaker_id


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave
        

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])
        
        max_ref_length = max([b[-1].shape[1] for b in batch])

        labels = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        input_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        paths = ['' for _ in range(batch_size)]
        waves = [None for _ in range(batch_size)]
        
        adj_mels = torch.zeros((batch_size, nmels, max_ref_length)).float()
        adj_mels_lengths = torch.zeros(batch_size).long()
        
        for bid, (label, mel, text, ref_mel, ref_label, path, wave, adj_mel) in enumerate(batch):
            
            mel_size = mel.size(1)
            text_size = text.size(0)
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            input_lengths[bid] = text_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            
            adj_mels_size = adj_mel.size(1)
            adj_mels[bid, :, :adj_mels_size] = adj_mel
            
            ref_labels[bid] = ref_label
            waves[bid] = wave
            adj_mels_lengths[bid] = adj_mels_size
            
#             if(text_size < (mel_size//2)):
#                 print(text_size, mel_size//2)

        return waves, texts, input_lengths, mels, output_lengths, labels, ref_mels, ref_labels, adj_mels, adj_mels_lengths




def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):
    
    dataset = FilePathDataset(path_list, validation=validation, **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader
