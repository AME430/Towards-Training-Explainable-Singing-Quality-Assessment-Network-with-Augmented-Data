import os
import pdb
import scipy.io.wavfile
import dill
from pitch_histogram import CreateNoteHistogram

mother_dir = "/data07/chitra/workspace/data/PESnQ+Discover_DATA"
songs = ['_cups_pitch_perfect','_let_it_go','_stay_rihanna','_when_i_was_your_man']
songs_train = ['_cups_pitch_perfect','_let_it_go','_when_i_was_your_man']
songs_leave = ['_stay_rihanna']
audio_snippets = ['audio_snippets','audio_snippets_2','audio_snippets_3','audio_snippets_4','audio_snippets_5','wavfiles']

# test_singers = [1,11,21,31,41,51,61,71,81,91]
# val_singers = [2,12,22,32,42,52,62,72,82,92]

test_singers = list(set(range(100)))
val_singers = list(set(range(1,100,3)))
train_singers = list(set(range(100)) - set(val_singers))

def CreateBWSdict(bwsfile):
    ## This function takes in bws file and outputs a dictionary with the singername and the corresponding score
    flines = open(bwsfile,'r').readlines()
    names_scores = []
    for line in flines:
        name,score = line.replace('\n', '').split(',')
        names_scores.append((name,float(score)))
    names_scores.sort(key=lambda x:(-x[1],x[0]))
    SingerScore = {}
    for name, score in names_scores:
        SingerScore[name] = score
    return names_scores,SingerScore

if __name__=='__main__':
    cv1_folder = '/data07/huanglin/SingEval/LeaderboardData/data_ph'
    train_dill = open(cv1_folder+os.sep+'train_leave_3.dill','wb')
    test_dill = open(cv1_folder+os.sep+'test_leave_3.dill','wb')
    val_dill = open(cv1_folder+os.sep+'val_leave_3.dill','wb')

    train_tobedumped = []
    test_tobedumped = []
    val_tobedumped = []

    # get training set
    print('training set:')
    for songname_train in songs_train:
        print(songname_train)
        bwsfile_train = mother_dir+os.sep+'MTurkProcessing/bws' + songname_train + '.txt'
        singer_score_tuple_sorted_train, singer_score_dict_train = CreateBWSdict(bwsfile_train)
        pitch_folder = '/data07/huanglin/SingEval/LeaderboardData/pitch_folder'

        for idx in train_singers:
            singername = singer_score_tuple_sorted_train[idx][0]
            rating = singer_score_tuple_sorted_train[idx][1]
            for snippet_num in audio_snippets:
                wavfile = mother_dir + os.sep + snippet_num + os.sep + songname_train + os.sep + singername + '.wav'
                fs, audio = scipy.io.wavfile.read(wavfile)
                original_pitch_file = pitch_folder + os.sep + snippet_num + '_' + singername + '.pitch'
                ph_notes = CreateNoteHistogram(original_pitch_file)
                train_tobedumped.append(
                    {'audio': [audio / 32768.0, fs], 'pitch_histogram': ph_notes, 'ratings': [rating]})

        for idx in val_singers:
            singername = singer_score_tuple_sorted_train[idx][0]
            rating = singer_score_tuple_sorted_train[idx][1]
            for snippet_num in audio_snippets:
                wavfile = mother_dir + os.sep + snippet_num + os.sep + songname_train + os.sep + singername + '.wav'
                fs, audio = scipy.io.wavfile.read(wavfile)
                original_pitch_file = pitch_folder + os.sep + snippet_num + '_' + singername + '.pitch'
                ph_notes = CreateNoteHistogram(original_pitch_file)
                val_tobedumped.append(
                    {'audio': [audio / 32768.0, fs], 'pitch_histogram': ph_notes, 'ratings': [rating]})

    dill.dump(train_tobedumped, train_dill)
    dill.dump(val_tobedumped, val_dill)

    print('leave one song: ')
    for songname in songs_leave:
        print(songname)
        bwsfile = mother_dir + os.sep + 'MTurkProcessing/bws' + songname + '.txt'
        ### Create dictionary of Human BWS Scores
        singer_score_tuple_sorted, singer_score_dict = CreateBWSdict(bwsfile)
        pitch_folder = '/data07/huanglin/SingEval/LeaderboardData/pitch_folder'

        for idx in test_singers:
            singername = singer_score_tuple_sorted[idx][0]
            rating = singer_score_tuple_sorted[idx][1]
            for snippet_num in audio_snippets:
                wavfile = mother_dir + os.sep + snippet_num + os.sep + songname + os.sep + singername + '.wav'
                fs, audio = scipy.io.wavfile.read(wavfile)
                original_pitch_file = pitch_folder + os.sep + snippet_num + '_' + singername + '.pitch'
                ph_notes = CreateNoteHistogram(original_pitch_file)
                test_tobedumped.append(
                    {'audio': [audio / 32768.0, fs], 'pitch_histogram': ph_notes, 'ratings': [rating]})
        # pdb.set_trace()

    dill.dump(test_tobedumped, test_dill)







