import os
import glob
import numpy as np
import scipy.io.wavfile as wav
import librosa

# merge speech 
def merge_speech(path_read_folder, path_write_wav_file,song_name):
    files= []
    for f in os.listdir(path_read_folder):
        if f.endswith('.wav') and f[0:3] == song_name:
            files.append(f)

    files.sort(key = lambda x: int(x[:-4]))

    merged_signal = []
    for f in files:
        signal1,sr = librosa.load(path_read_folder + os.sep + f,sr=None)
        merged_signal.append(signal1)

    merged_signal = np.hstack(merged_signal)
    merged_signal = np.asarray(merged_signal, dtype=np.float32)
    writen_path = path_write_wav_file + os.sep + 'speech_' + song_name + '.wav'
    
    librosa.output.write_wav(writen_path, merged_signal,sr)

