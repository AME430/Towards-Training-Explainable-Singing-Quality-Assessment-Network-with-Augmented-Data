import os
import pdb
import scipy.io.wavfile
import dill

# mother_dir = "/data07/chitra/workspace/data/PESnQ+Discover_DATA"
mother_dir = "C:/Dataset/PESnQ+Discover_DATA"
songs = [
    '_cups_pitch_perfect', '_let_it_go', '_stay_rihanna',
    '_when_i_was_your_man'
]
audio_snippets = [
    'audio_snippets', 'audio_snippets_2', 'audio_snippets_3',
    'audio_snippets_4', 'audio_snippets_5', 'wavfiles'
]

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
    names_scores.sort(key=lambda x: (-x[1], x[0])) #生成列表排序
    SingerScore = {}
    for name, score in names_scores:
        SingerScore[name] = score   # 生成字典 用排序好的列表来产生字典，也是从高到低的顺序
    return names_scores, SingerScore


if __name__ == '__main__':
    # cv1_folder = '/data07/huanglin/SingEval/LeaderboardData/data'
    cv1_folder = 'D:/dill_data/test'
    train_dill = open(cv1_folder + os.sep + 'train_1.dill', 'wb')  # modify
    test_dill = open(cv1_folder + os.sep + 'test_1.dill', 'wb')
    val_dill = open(cv1_folder + os.sep + 'val_1.dill', 'wb')

    train_tobedumped = []
    test_tobedumped = []
    val_tobedumped = []

    for songname in songs:
        print(songname)

        bwsfile = mother_dir + os.sep + 'MTurkProcessing/bws' + songname + '.txt'
        ### Create dictionary of Human BWS Scores
        singer_score_tuple_sorted, singer_score_dict = CreateBWSdict(bwsfile)

        for idx in test_singers:
            singername = singer_score_tuple_sorted[idx][0]
            rating = singer_score_tuple_sorted[idx][1] # 排序好的也有测试集和验证集 后面的分数
            for snippet_num in audio_snippets: # 不同段的音频和总的音频
                wavfile = mother_dir + os.sep + snippet_num + os.sep + songname + os.sep + singername + '.wav'
                fs, audio = scipy.io.wavfile.read(wavfile)  # Return the sample rate (in samples/sec) and data from a WAV file.
                test_tobedumped.append({
                    'audio': [audio / 32768.0, fs],  # 这个操作的作用？？
                    'ratings': [rating]
                })
        #pdb.set_trace()  

        for idx in val_singers:
            singername = singer_score_tuple_sorted[idx][0]
            rating = singer_score_tuple_sorted[idx][1]
            for snippet_num in audio_snippets:
                wavfile = mother_dir + os.sep + snippet_num + os.sep + songname + os.sep + singername + '.wav'
                fs, audio = scipy.io.wavfile.read(wavfile)
                val_tobedumped.append({
                    'audio': [audio / 32768.0, fs],
                    'ratings': [rating]
                })

        for idx in train_singers:
            singername = singer_score_tuple_sorted[idx][0]
            rating = singer_score_tuple_sorted[idx][1]
            for snippet_num in audio_snippets:
                wavfile = mother_dir + os.sep + snippet_num + os.sep + songname + os.sep + singername + '.wav'
                fs, audio = scipy.io.wavfile.read(wavfile)
                train_tobedumped.append({
                    'audio': [audio / 32768.0, fs],
                    'ratings': [rating]
                })

    dill.dump(train_tobedumped, train_dill)
    dill.dump(test_tobedumped, test_dill)
    dill.dump(val_tobedumped, val_dill)
