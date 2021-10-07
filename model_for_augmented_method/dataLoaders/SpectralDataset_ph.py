import os
import dill
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
#import pdb

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg


class AudioToSpectralRep:
    def __init__(self, rep_params):
        self.params = rep_params

    def extract_features(self, y, sr):
        try:
            if self.params['normalize']:
                rms = np.sqrt(np.mean(y * y))
                if rms > 1e-4:
                    y = y / rms
        except KeyError:
            pass

        try:
            if self.params['remove_silence']:
                y = self.remove_silence(y,
                                        window=32,
                                        hop=32,
                                        threshold=self.params['sil_threshold'])
        except KeyError:
            pass

        if self.params['method'] == 'FFT':
            x = np.abs(
                librosa.stft(y,
                             n_fft=self.params['n_fft'],
                             hop_length=self.params['hop_length']))
            x = librosa.logamplitude(x**2)

        elif self.params['method'] == 'Mel Spectrogram':
            x = librosa.feature.melspectrogram(
                y,
                sr,
                n_fft=self.params['n_fft'],
                hop_length=self.params['hop_length'],
                n_mels=self.params['n_mels'])
            x = librosa.amplitude_to_db(x**2)

        elif self.params['method'] == 'CQT':
            x = librosa.hybrid_cqt(
                y,
                sr,
                hop_length=self.params['hop_length'],
                n_bins=self.params['n_bins'],
                bins_per_octave=self.params['bins_per_octave'])
            x = librosa.amplitude_to_db(x**2)

        elif self.params['method'] == 'Chromagram':
            x = librosa.feature.chroma_cqt(
                y,
                sr,
                hop_length=self.params['hop_length'],
                n_chroma=self.params['n_chroma'],
                cqt_mode=self.params['cqt_mode'])

        elif self.params['method'] == 'MFCC':
            x = librosa.feature.mfcc(y,
                                     sr,
                                     n_fft=self.params['n_fft'],
                                     n_mels=self.params['n_mels'],
                                     n_mfcc=self.params['n_mfcc'],
                                     hop_length=self.params['hop_length'])
            delta = librosa.feature.delta(x)
            d_delta = librosa.feature.delta(x, order=2)
            x = np.concatenate([x, delta, d_delta], axis=0)

        return torch.FloatTensor(x)

    def remove_silence(self, y, window, hop, threshold):
        frames = librosa.util.frame(y, window, hop)
        sil_frames = []
        for i, f in enumerate(frames.transpose()):
            if np.sqrt(np.mean(f * f)) < threshold:
                sil_frames.append(i)
        y = np.delete(frames, sil_frames, 1)
        y = y.reshape(-1)
        return y


class SpectralDataset(Dataset, AudioToSpectralRep):
    """Dataset class for spectral feature based music performance assessment data"""
    def __init__(self, data_path, label_id, rep_params):
        """
        Initializes the class, defines the number of datapoints
        Args:
            data_path:  full path to the file which contains the pitch contour data
            label_id:   the label to use for training
            rep_params: parameters for spectral representation
        """
        super(SpectralDataset, self).__init__(rep_params)
        self.perf_data = dill.load(open(data_path, 'rb'))
        self.label_id = label_id
        self.length = len(self.perf_data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        y, sr = self.perf_data[idx]['audio']
        X = self.extract_features(y, sr)
        label = self.perf_data[idx]['ratings'][self.label_id]
        ph_notes = self.perf_data[idx]['pitch_histogram'] # 不同
        pitchscore = self.perf_data[idx]['pitchscore']
        y = label
        return torch.unsqueeze(X, 0), y, ph_notes,pitchscore # 不同


def _collate_fn(batch):
    def func(p):
        return p[0].size(2)

    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(1)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(2)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    targets = []
    ph_notes = []
    pitchscores = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        ph_note = sample[2]
        pitchscore = sample[3]

        ########## normalize pitch histogram ##########
        # way 1: normalize to 0 to 1
        ph_note_np = np.array(ph_note)  # 不同
        ph_min = np.min(ph_note_np)
        ph_max = np.max(ph_note_np)
        ph_norm = (ph_note_np - ph_min) / (ph_max - ph_min)

        # way 2: normalize to -1 to 1
        # ph_note_np = np.array(ph_note)
        # ph_min = np.min(ph_note_np)
        # ph_max = np.max(ph_note_np)
        # ph_norm = -1 + 2 * (ph_note_np - ph_min) / (ph_max - ph_min)

        # way 3: normalize to 0 to 1 (possibility distribution)
        # ph_note_np = np.array(ph_note)
        # ph_norm = ph_note_np / np.sum(ph_note_np)
        ##############################################

        seq_length = tensor.size(2)
        inputs[x].narrow(2, 0, seq_length).copy_(tensor)
        targets.append(target)
        ph_notes.append(ph_norm)
        pitchscores.append(pitchscore)
    targets = torch.Tensor(targets)
    ph_notes = torch.Tensor(ph_notes)
    pitchscores = torch.Tensor(pitchscores)
    #pdb.set_trace()
    return inputs, targets, ph_notes,pitchscores
    # return inputs, targets, ph_notes,pitchscores


class SpectralDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(SpectralDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
