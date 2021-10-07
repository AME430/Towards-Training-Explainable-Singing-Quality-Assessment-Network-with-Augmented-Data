######## estimate pitch ########

import scipy.io.wavfile as wav
import numpy as np
from fastdtw import fastdtw
from matplotlib import pylab as plt
# import pdb
import pickle
import os
import csv
from scipy.spatial.distance import euclidean
import seaborn as sns
import scipy.stats

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
        chunk = sig[int(i):int(i + window)][np.newaxis]
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
    pitch_extract_cmd = r'C:\Praat.exe --run ExtractPitch.praat ' + wavfile + ' ' + pitchfile + ' ' + str(
        pitch_ceiling) + ' ' + str(hop)
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
    ## This function takes in bws file and outputs a dictionary with the singername and the corresponding score
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


def EstimatePitch():  # 检测pitch
    for singer, bwsscore in singer_score_tuple_sorted:
        filename = singer

        for snippet_num in snippet_nums:
            original = wavfolder + os.sep + snippet_num + os.sep + songname + os.sep + filename + '.wav'

            (rate, ori_sig) = wav.read(original) 
            ori_sig = ori_sig / 32768.0
            ori_sig = ori_sig - np.mean(ori_sig)  # remove DC offset
            window = NFFT / (rate * 1.0)
            hop = window / 2.0
            ori_sig = InitialFinalSilenceRemoved(ori_sig)
            wav.write(
                wav_folder + os.sep + snippet_num + '_' + filename + '.wav',
                rate, np.int16(ori_sig * 32767)) # 写入新的去掉silence的wav

            # print( "### Extract Pitch Using Praat tool (autocorrelation-based pitch extraction)")
            original_pitch_file = pitch_folder + os.sep + snippet_num + '_' + filename + '.pitch'
            extract_pitch(
                wav_folder + os.sep + snippet_num + '_' + filename + '.wav',
                original_pitch_file, 0.01)


NFFT = 512
# mother_dir = r"C:\Users\hl\Desktop\NUS\EE5003 project\Data_HuangLin"
mother_dir = r"C:\Dataset\PESnQ+Discover_DATA"
bwsfolder = mother_dir + os.sep + 'bws_files'
wavfolder = mother_dir
songnames = [
    '_cups_pitch_perfect', '_let_it_go', '_stay_rihanna',
    '_when_i_was_your_man'
]

# snippet_nums = [
#     'audio_snippets_1', 'audio_snippets_2', 'audio_snippets_3',
#     'audio_snippets_4', 'audio_snippets_5', 'wavfiles'
# ]

snippet_nums = [
    'audio_snippets', 'audio_snippets_2', 'audio_snippets_3',
    'audio_snippets_4', 'audio_snippets_5', 'wavfiles'
]


for songname in songnames:
    print('name of the song:', songname)

    bwsfile = bwsfolder + os.sep + 'bws' + songname + '.txt'
    ### Create dictionary of Human BWS Scores
    singer_score_tuple_sorted, singer_score_dict = CreateBWSdict(bwsfile) 

    # wav_folder = 'wav_folder'
    wav_folder = 'C:/Dataset/PESnQ+Discover_DATA/writewav'
    pitch_folder = 'C:/Dataset/PESnQ+Discover_DATA/pitch_folder'

    EstimatePitch()
