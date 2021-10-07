import os
from toolbox_Chinese.mapping import mapping
# from toolbox_Chinese.cutandchange import cutandchange
from toolbox_Chinese.new_cut import cutandchange
from toolbox_Chinese.merge import merge_files
from toolbox_Chinese.estimate_picth import *
from toolbox_Chinese.plot import plotph
from toolbox_Chinese.delete import delete_files

def Generate_pitch_original(path,label):  # 检测pitch
    NFFT = 512
    original1 = path + os.sep + 'Vox' + os.sep + label + '.wav'
    ori_sig, rate = librosa.load(original1, sr=None)

    ori_sig = ori_sig - np.mean(ori_sig)  # remove DC offset
    window = NFFT / (rate * 1.0)
    hop = window / 2.0
    ori_sig = InitialFinalSilenceRemoved(ori_sig)

    wav.write(
        ori_path + os.sep + 'pitch_folder' + os.sep + 'nosilence' + os.sep + singer_name +song_name + '.wav',
        rate, np.int16(ori_sig * 32767))  # 写入新的去掉silence的wav

    original_pitch_file = ori_path + os.sep + 'pitch_folder' + os.sep + singer_name + '_' + song_name + '.pitch'
    extract_pitch(
        ori_path + os.sep + 'pitch_folder' + os.sep + 'nosilence' + os.sep + singer_name +song_name + '.wav',
        original_pitch_file, 0.01)

# The most primitive path
ori_path = 'C:/DataBaker_songs'
singer_names = ['DB-DM-001-F-001','DB-DM-002-F-002','DB-DM-003-F-003','DB-DM-004-M-001',
'DB-DM-005-M-002','DB-DM-006-F-004','DB-DM-007-M-003','DB-DM-008-M-004']
# singer_names = [ 'DB-DM-005-M-002','DB-DM-007-M-003','DB-DM-008-M-004']

# creat file for bad samples
for singer_name in singer_names:
    if not os.path.exists(ori_path + os.sep + singer_name + '/badsample_changepitch'):
        os.makedirs(ori_path + os.sep + singer_name + '/badsample_changepitch')

# read song names
for singer_name in singer_names:
    second_path = ori_path + os.sep + singer_name + os.sep + 'Vox'
    f_list = os.listdir(second_path)
    Vox_list = []
    for f in f_list:
        if os.path.splitext(f)[1] == '.wav':  # seperate with extension name
            Vox_list.append(f.split('.')[0])
    # generate mapping list
    # Vox_list = ['007','008','009','010','011','012']
    for song_name in Vox_list:
        print((singer_name,song_name))
        # delete_files(ori_path,singer_name,song_name)
        mapping(ori_path + os.sep + singer_name, song_name)

        # cut and change
        cutandchange(ori_path + os.sep + singer_name, song_name)

        # merge
        merge_files(ori_path + os.sep + singer_name, song_name)

        # ph
        runph(ori_path + os.sep + singer_name, song_name)
        # runph(ori_path + os.sep + singer_name, song_name)

        # delete
        delete_files(ori_path,singer_name,song_name)


        # plot
        # plotph(ori_path + os.sep + singer_name + os.sep + 'badsample_changepitch' + os.sep + song_name + os.sep + 'pitch.pitch' )



