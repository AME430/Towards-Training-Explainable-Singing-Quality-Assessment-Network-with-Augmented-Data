import os
import pdb
import scipy.io.wavfile
import dill
from pitch_histogram import CreateNoteHistogram
from matplotlib import pylab as plt
import numpy as np

mother_dir = "/data07/chitra/workspace/data/PESnQ+Discover_DATA"
songs = ['_cups_pitch_perfect','_let_it_go','_stay_rihanna','_when_i_was_your_man']
audio_snippets = ['audio_snippets','audio_snippets_2','audio_snippets_3','audio_snippets_4','audio_snippets_5','wavfiles']

test_singers = [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]  # modify
val_singers = [4, 14, 24, 34, 44, 54, 64, 74, 84, 94]
train_singers = list(set(range(100)) - set(test_singers) - set(val_singers))


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
    cv1_folder = '/data07/huanglin/SingEval/LeaderboardData/data_ph'
    train_dill = open(cv1_folder + os.sep + 'train_3.dill', 'wb')  # modify
    test_dill = open(cv1_folder + os.sep + 'test_3.dill', 'wb')
    val_dill = open(cv1_folder + os.sep + 'val_3.dill', 'wb')

    train_tobedumped = []
    test_tobedumped = []
    val_tobedumped = []

    # to normalize the bws score between 0 and 1
    min_bws = -1.0  # modify
    max_bws = 1.0

    for songname in songs:
        print(songname)
        bwsfile = mother_dir+os.sep+'MTurkProcessing/bws' + songname + '.txt'
        ### Create dictionary of Human BWS Scores
        singer_score_tuple_sorted, singer_score_dict = CreateBWSdict(bwsfile)
        pitch_folder = '/data07/huanglin/SingEval/LeaderboardData/pitch_folder'

        for idx in train_singers:
            singername = singer_score_tuple_sorted[idx][0]
            rating = singer_score_tuple_sorted[idx][1]
            # rating = (rating-min_bws)/(max_bws-min_bws) #normalize the bws rating between 0 and 1
            for snippet_num in audio_snippets:
                wavfile = mother_dir + os.sep + snippet_num + os.sep + songname + os.sep + singername + '.wav'
                fs, audio = scipy.io.wavfile.read(wavfile)
                original_pitch_file = pitch_folder + os.sep + snippet_num + '_' + singername + '.pitch'
                ph_notes = CreateNoteHistogram(original_pitch_file) # class: list
                train_tobedumped.append({'audio': [audio / 32768.0, fs], 'pitch_histogram': ph_notes, 'ratings': [rating]})

        for idx in test_singers:
            singername = singer_score_tuple_sorted[idx][0]
            rating = singer_score_tuple_sorted[idx][1]
            # rating = (rating-min_bws)/(max_bws-min_bws) #normalize the bws rating between 0 and 1
            for snippet_num in audio_snippets:
                wavfile = mother_dir + os.sep + snippet_num + os.sep + songname + os.sep + singername + '.wav'
                fs, audio = scipy.io.wavfile.read(wavfile)
                original_pitch_file = pitch_folder + os.sep + snippet_num + '_' + singername + '.pitch'
                ph_notes = CreateNoteHistogram(original_pitch_file)  # class: list
                test_tobedumped.append({'audio': [audio / 32768.0, fs], 'pitch_histogram': ph_notes, 'ratings': [rating]})

        for idx in val_singers:
            singername = singer_score_tuple_sorted[idx][0]
            rating = singer_score_tuple_sorted[idx][1]
            # rating = (rating-min_bws)/(max_bws-min_bws) #normalize the bws rating between 0 and 1
            for snippet_num in audio_snippets:
                wavfile = mother_dir + os.sep + snippet_num + os.sep + songname + os.sep + singername + '.wav'
                fs, audio = scipy.io.wavfile.read(wavfile)
                original_pitch_file = pitch_folder + os.sep + snippet_num + '_' + singername + '.pitch'
                ph_notes = CreateNoteHistogram(original_pitch_file)  # class: list
                val_tobedumped.append({'audio': [audio / 32768.0, fs], 'pitch_histogram': ph_notes, 'ratings': [rating]})

    dill.dump(train_tobedumped, train_dill)
    dill.dump(test_tobedumped, test_dill)
    dill.dump(val_tobedumped, val_dill)
