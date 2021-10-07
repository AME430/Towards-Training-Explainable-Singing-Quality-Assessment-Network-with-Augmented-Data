import os
import glob
import numpy as np
import scipy.io.wavfile as wav
import librosa
from pydub import AudioSegment
import random
import numpy
import numba

def merge_files(path,song_name):
    path_read_folder = path + os.sep + song_name  + os.sep+ 'notescut'
    path_write_wav_file = path + os.sep + song_name + os.sep + 'small.wav'

    files= []
    for f in os.listdir(path_read_folder):
        files.append(f)

    files.sort(key = lambda x: int(x[:-8]))

    merged_signal = []
    for i in range(len(files)):
        if os.path.exists(path_read_folder + os.sep + str(i+1) + "_" + song_name + ".wav"):
            signal1,sr = librosa.load(path_read_folder + os.sep + str(i+1) + "_" + song_name + ".wav",sr=None)
            # sr,signal1 = wav.read(path_read_folder + os.sep + str(i+1) + "_" + song_name + ".wav")
            # signal1,sr = librosa.load(path_read_folder + os.sep + str(i+1) + "_" + song_name + ".wav")
            if np.size(signal1) >=50:
                list_i = [-1,1]
                temp = random.sample(list_i,1)
                # shift_list = [0]
                y = librosa.effects.pitch_shift(signal1, sr, n_steps=temp[0]*150, bins_per_octave=1200)
                merged_signal.append(y)

    merged_signal = np.hstack(merged_signal)
    merged_signal = np.asarray(merged_signal, dtype=np.float32)
    # merged_signal = np.asarray(merged_signal)
    librosa.output.write_wav(path_write_wav_file, merged_signal,sr)
    # print(librosa.__version__)
    # print(numba.__version__)


    # louder
    song = AudioSegment.from_wav(path_write_wav_file)

    song = song + 10

    song.export(path + os.sep + song_name + os.sep + 'result.wav', "wav")




