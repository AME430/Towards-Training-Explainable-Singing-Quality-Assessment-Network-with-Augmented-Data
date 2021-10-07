import os
from toolbox_Chinese.merge import merge_files
from toolbox_Chinese.estimate_picth import *
import shutil

def delete_files(ori_path,singer_name,song_name):
    # ori_path = 'C:/DataBaker_songs'
    # singer_names = ['DB-DM-001-F-001','DB-DM-002-F-002','DB-DM-003-F-003','DB-DM-004-M-001',
    #                 'DB-DM-005-M-002','DB-DM-006-F-004','DB-DM-007-M-003','DB-DM-008-M-004']
    #
    # todelete_files = ['nosilence.wav','small.wav','speech_']

    # for singer_name in singer_names:
    # second_path = ori_path + os.sep + singer_name + os.sep + 'Vox'
    # f_list = os.listdir(second_path)
    # Vox_list = []
    # for f in f_list:
    #     if os.path.splitext(f)[1] == '.wav':  # seperate with extension name
    #         Vox_list.append(f.split('.')[0])
    # for song_name in Vox_list:
    delete_path = ori_path+ os.sep + singer_name + os.sep + song_name + os.sep
    delete_file1 = 'nosilence.wav'
    delete_file2 = 'small.wav'
    delete_file3 ='speech_' + song_name +'.wav'
    delete_dir1 = 'notescut'

    if os.path.exists(delete_path + delete_file1):
        os.remove(delete_path + delete_file1)
    if os.path.exists(delete_path + delete_file2):
        os.remove(delete_path +delete_file2)
    if os.path.exists(delete_path + delete_file3):
        os.remove(delete_path +delete_file3)
    if os.path.exists(delete_path + delete_dir1):
        shutil.rmtree(delete_path + delete_dir1)

    ###################################################
    # ori_path = 'C:/Dataset/NHSS_Database/Data'
    # singer_names = ['F01','F02','F03','F04','F05','M01','M02','M03','M04','M05']
    #
    # for singer_name in singer_names:
    #     second_path = ori_path + os.sep + singer_name
    #     f_list = os.listdir(second_path)
    #     song_names = []
    #
    #     for f in f_list:
    #         if f.startswith('S'):
    #             song_names.append(f)
    #
    #     for song_name in song_names:
    #         delete_path = ori_path + os.sep + singer_name + os.sep + song_name +os.sep
    #         delete_file1 = 'song_nosilence.wav'
    #         delete_file2 = 'small.wav'
    #         delete_file3 ='speech.wav'
    #
    #         if os.path.exists(delete_path + delete_file1):
    #             os.remove(delete_path + delete_file1)
    #         if os.path.exists(delete_path + delete_file2):
    #             os.remove(delete_path +delete_file2)
    #         if os.path.exists(delete_path + delete_file3):
    #             os.remove(delete_path +delete_file3)




