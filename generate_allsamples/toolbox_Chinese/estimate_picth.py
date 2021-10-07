######## estimate pitch ########

import scipy.io.wavfile as wav
import numpy as np
# from fastdtw import fastdtw
from matplotlib import pylab as plt
# import pdb
# import pickle
import os
import csv
# from scipy.spatial.distance import euclidean
import seaborn as sns
# import scipy.stats
import librosa

sns.set()
plot = 0


def PitchMedianSubtraction(time_pitch):
    time = time_pitch[:, 0]
    pitch = time_pitch[:, 1]
    median = np.median(pitch)
    pitch_new = pitch - median
    time_pitch_mediansubtracted = np.hstack(
        [time[np.newaxis].T, pitch_new[np.newaxis].T])
    return time_pitch_mediansubtracted


def InitialFinalSilenceRemoved(sig):
    energy_thresh = 0.01
    window = 512
    hop = window / 2
    energy = []
    i = 0
    energy_index = []
    while i < (len(sig) - window):
        chunk = sig[int(i):int(i + window)][np.newaxis]  # 扩展维度
        energy.append(chunk.dot(chunk.T)[0][0])
        energy_index.append(i)
        i = i + hop

    energy = np.array(energy)
    significant_indices = np.where(energy > energy_thresh)[0]
    if significant_indices[0] == 0:
        start_point_sample = 0
    else:
        start_point_sample = (significant_indices[0] - 1) * hop
    if significant_indices[-1] == len(energy) - 1:
        end_point_sample = len(energy) * hop
    else:
        end_point_sample = (significant_indices[-1] + 1) * hop
    new_sig = sig[int(start_point_sample):int(end_point_sample + 1)]
    return new_sig


def extract_time_pitch(file):
    f = open(file, 'r')
    obj = csv.reader(f, delimiter=' ')
    cols = []
    for row in obj:
        if len(row) < 2:
            break
        if np.size(cols) == 0:
            cols = [
                -100.0 if 'undefined' in elem else float(elem) for elem in row
            ]
        else:
            cols = np.vstack((cols, [
                -100.0 if 'undefined' in elem else float(elem) for elem in row
            ]))
    cols_modified = []
    for row in cols:
        if row[1] == -100.0:
            continue
        if np.size(cols_modified) == 0:
            cols_modified = row[:]
        else:
            cols_modified = np.vstack((cols_modified, row))
    return cols_modified


def extract_pitch(wavfile, pitchfile, hop):
    pitch_ceiling = 650.0
    # Extracting pitch
    # pitch_extract_cmd = r'C:\Users\hl\Downloads\praat6103_win64\Praat.exe --run ExtractPitch.praat ' + wavfile + ' ' + pitchfile + ' ' + str(
    pitch_extract_cmd = r'C:\Praat.exe --run C:\Users\77490\Desktop\ExtractPitch.praat ' + wavfile + ' ' + pitchfile + ' ' + str(pitch_ceiling) + ' ' + str(hop)
    os.system(pitch_extract_cmd)


def GridMap(time_pitch):
    pitch = time_pitch[:, 1]
    pitch_new = []
    for elem in pitch:
        if elem > 6:
            elem = np.mod(elem, 6) - 6
        elif elem < -6:
            elem = np.mod(elem, 6) + 6
        pitch_new.append(elem)
    return pitch_new


def plotpitch(pitch):
    plt.figure()
    plt.plot(pitch)
    # plt.show()


def GetFinerNoteHistogram(griddedpitch):
    notes = [0] * 120
    for elem in griddedpitch:
        count = 0
        for ind in np.arange(-6, 5.5, 0.1):
            left = ind
            right = ind + 0.1

            if elem >= left and elem < right:
                notes[count] = notes[count] + 1
            count = count + 1
        count = 0
        for ind in np.arange(6, 5.5, -0.1):
            right = ind
            left = ind - 0.1

            if elem >= left and elem < right:
                notes[count] = notes[count] + 1
            count = count + 1
    return notes


def plotHistogram(notes):
    plt.figure()
    plt.bar(range(len(notes)), notes / np.trapz(notes))
    plt.xlabel('Pitch x 10 cents', fontsize=15)
    plt.ylabel('Normalized number of frames', fontsize=15)
    plt.show()


def CreateBWSdict(bwsfile):
    # This function takes in bws file and outputs a dictionary with the singername and the corresponding score
    flines = open(bwsfile, 'r').readlines()
    names_scores = []
    for line in flines:
        name, score = line.replace('\n', '').split(',')
        names_scores.append((name, float(score)))
    names_scores.sort(key=lambda x: (-x[1], x[0]))  # tup: tup[1]
    SingerScore = {}
    for name, score in names_scores:
        SingerScore[name] = score
    return names_scores, SingerScore


def EstimatePitch(path,label):  # 检测pitch

    original1 = path + label + '.wav'
    ori_sig, rate = librosa.load(original1, sr=None)

    # ori_sig = ori_sig / 32768.0
    ori_sig = ori_sig - np.mean(ori_sig)  # remove DC offset
    window = NFFT / (rate * 1.0)
    hop = window / 2.0
    ori_sig = InitialFinalSilenceRemoved(ori_sig)

    wav.write(
        path+ 'nosilence'+ '.wav',
        rate, np.int16(ori_sig * 32767))  # 写入新的去掉silence的wav

    original_pitch_file = path+ 'pitch' + '.pitch'
    extract_pitch(
        path+ 'nosilence'+ '.wav',
        original_pitch_file, 0.01)


def runph(path,song_name):
    global NFFT
    NFFT = 512
    mother_dir = path + os.sep + 'badsample_changepitch' + os.sep + song_name + os.sep
    label = 'result' # choose sample
    # global wav_folder
    # wav_folder = mother_dir + os.sep
    # global pitch_folder
    # pitch_folder = mother_dir + os.sep

    EstimatePitch(mother_dir,label)
