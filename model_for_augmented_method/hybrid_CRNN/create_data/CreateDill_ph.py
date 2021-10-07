import os
import pdb
import scipy.io.wavfile
import dill
import sys
sys.path.append('E:\\Codes_For_Python\\.vscode\\Huang')
from hybrid_CRNN.create_data.pitch_histogram import CreateNoteHistogram
from matplotlib import pylab as plt
import numpy as np

# mother_dir = "/data07/chitra/workspace/data/PESnQ+Discover_DATA"
mother_dir = "C:/Dataset/PESnQ+Discover_DATA"
songs = ['_cups_pitch_perfect','_let_it_go','_stay_rihanna','_when_i_was_your_man']
audio_snippets = ['audio_snippets','audio_snippets_2','audio_snippets_3','audio_snippets_4','audio_snippets_5','wavfiles']

# test_singers = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
# val_singers = [2, 12, 22, 32, 42, 52, 62, 72, 82, 92]
# train_singers = list(set(range(100)) - set(test_singers) - set(val_singers))

test_singers = [11]
val_singers = [12]
# train_singers = list(set(range(100)) - set(test_singers) - set(val_singers))
train_singers = [13]

def CreateBWSdict(bwsfile):
    ## This function takes in bws file and outputs a dictionary with the singername and the corresponding score
    flines = open(bwsfile, 'r').readlines()
    names_scores = []
    for line in flines:
        name, score = line.replace('\n', '').split(',')
        names_scores.append((name, float(score)))
    names_scores.sort(key=lambda x: (-x[1], x[0]))
    SingerScore = {}
    for name, score in names_scores:
        SingerScore[name] = score
    return names_scores, SingerScore


def plotpitch(pitch):
    plt.figure()
    plt.plot(pitch)
    plt.show()


if __name__ == '__main__':
    # cv1_folder = '/data07/huanglin/SingEval/LeaderboardData/data_ph'
    cv1_folder = 'D:/nuspro/dilldata_ph/test'
    train_dill = open(cv1_folder + os.sep + 'train_1.dill', 'wb')
    test_dill = open(cv1_folder + os.sep + 'test_1.dill', 'wb')
    val_dill = open(cv1_folder + os.sep + 'val_1.dill', 'wb')

    train_tobedumped = []
    test_tobedumped = []
    val_tobedumped = []

    for songname in songs:
        print(songname)
        bwsfile = mother_dir+os.sep+'MTurkProcessing/bws' + songname + '.txt'
        ### Create dictionary of Human BWS Scores
        singer_score_tuple_sorted, singer_score_dict = CreateBWSdict(bwsfile)
        # pitch_folder = '/data07/huanglin/SingEval/LeaderboardData/pitch_folder'
        pitch_folder = 'C:/pitch_folder'

        for idx in train_singers:
            singername = singer_score_tuple_sorted[idx][0]
            rating = singer_score_tuple_sorted[idx][1]
            for snippet_num in audio_snippets:
                wavfile = mother_dir + os.sep + snippet_num + os.sep + songname + os.sep + singername + '.wav'
                fs, audio = scipy.io.wavfile.read(wavfile)
                original_pitch_file = pitch_folder + os.sep + snippet_num + '_' + singername + '.pitch'
                ph_notes = CreateNoteHistogram(original_pitch_file)# 已经生成了histogram然后放入dill中
                train_tobedumped.append({'audio': [audio / 32768.0, fs], 'pitch_histogram': ph_notes, 'ratings': [rating]})

        for idx in test_singers:
            singername = singer_score_tuple_sorted[idx][0]
            rating = singer_score_tuple_sorted[idx][1]
            for snippet_num in audio_snippets:
                wavfile = mother_dir + os.sep + snippet_num + os.sep + songname + os.sep + singername + '.wav'
                fs, audio = scipy.io.wavfile.read(wavfile)
                original_pitch_file = pitch_folder + os.sep + snippet_num + '_' + singername + '.pitch'
                ph_notes = CreateNoteHistogram(original_pitch_file)
                test_tobedumped.append({'audio': [audio / 32768.0, fs], 'pitch_histogram': ph_notes, 'ratings': [rating]})

        for idx in val_singers:
            singername = singer_score_tuple_sorted[idx][0]
            rating = singer_score_tuple_sorted[idx][1]
            for snippet_num in audio_snippets:
                wavfile = mother_dir + os.sep + snippet_num + os.sep + songname + os.sep + singername + '.wav'
                fs, audio = scipy.io.wavfile.read(wavfile)
                original_pitch_file = pitch_folder + os.sep + snippet_num + '_' + singername + '.pitch'
                ph_notes = CreateNoteHistogram(original_pitch_file)
                val_tobedumped.append({'audio': [audio / 32768.0, fs], 'pitch_histogram': ph_notes, 'ratings': [rating]})

    dill.dump(train_tobedumped, train_dill)
    dill.dump(test_tobedumped, test_dill)
    dill.dump(val_tobedumped, val_dill)
