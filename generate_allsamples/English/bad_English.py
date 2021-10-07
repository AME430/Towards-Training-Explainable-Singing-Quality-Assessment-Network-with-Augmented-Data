import os
from toolbox_English.mapping import mapping
from toolbox_English.cutandchange import cutandchange
from toolbox_English.merge import merge_files
from toolbox_English.estimate_picth import *
from toolbox_English.plot import plotph
from toolbox_English.delete import delete_files

def Generate_pitch_original(path,label):  # 检测pitch
    NFFT = 512
    original1 = path + label + '.wav'
    ori_sig, rate = librosa.load(original1, sr=None)

    ori_sig = ori_sig - np.mean(ori_sig)  # remove DC offset
    window = NFFT / (rate * 1.0)
    hop = window / 2.0
    ori_sig = InitialFinalSilenceRemoved(ori_sig)

    wav.write(
        path+ label + '_' + 'nosilence'+ '.wav',
        rate, np.int16(ori_sig * 32767))  # 写入新的去掉silence的wav

    original_pitch_file = path+ label + '_' +'pitch' + '.pitch'
    extract_pitch(
        path+ label + '_' + 'nosilence'+ '.wav',
        original_pitch_file, 0.01)


# The most primitive path
ori_path = 'C:/Dataset/NHSS_Database/Data'
singer_names = ['F01','F02','F03','F04','F05','M01','M02','M03','M04','M05']
# singer_names = ['F02','F03','F04','F05','M01','M02','M03','M04','M05']
# singer_names = ['F01']

for singer_name in singer_names:
    second_path = ori_path + os.sep + singer_name
    f_list = os.listdir(second_path)
    song_names = []

    for f in f_list:
        if f.startswith('S'):
            song_names.append(f)

# read song names
#     song_names = ['S09','S11','S12','S15']

    for song_name in song_names:
        print((singer_name,song_name))
        delete_files(ori_path,singer_name,song_name)
    
    # mapping
        mapping(ori_path + os.sep + singer_name,song_name)

    # cut and change
        cutandchange(ori_path + os.sep + singer_name,song_name)

    # merge
        merge_files(ori_path + os.sep + singer_name, song_name)



    # pitch Histigram
        runph(ori_path + os.sep + singer_name, song_name)
        delete_files(ori_path,singer_name,song_name)
    # plot
    #     plotph(ori_path + os.sep + singer_name + os.sep + song_name + os.sep + 'pitch.pitch')

    # generate ori pitch
    # song_names = ['S05']
    # for song_name in song_names:
    #     Generate_pitch_original(ori_path + os.sep + singer_name + os.sep + song_name + os.sep, 'song')

        # plotph(ori_path + os.sep + singer_name + os.sep + song_name + os.sep + 'song_pitch.pitch')


