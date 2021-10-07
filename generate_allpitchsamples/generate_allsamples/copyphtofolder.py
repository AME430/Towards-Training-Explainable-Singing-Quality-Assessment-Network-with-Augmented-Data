import  os
import shutil

ori_path1 = 'C:/DataBaker_songs'
singer_names1 = ['DB-DM-001-F-001','DB-DM-002-F-002','DB-DM-003-F-003','DB-DM-004-M-001',
                                'DB-DM-005-M-002','DB-DM-006-F-004','DB-DM-007-M-003','DB-DM-008-M-004']

for singer_name in singer_names1:
    print(singer_name)
    second_path = ori_path1 + os.sep + singer_name + os.sep + 'Vox'
    f_list = os.listdir(second_path)
    Vox_list = []
    for f in f_list:
        if os.path.splitext(f)[1] == '.wav':  # seperate with extension name
            Vox_list.append(f.split('.')[0])

    for song_name in Vox_list:
        old_file = ori_path1 + os.sep + singer_name + os.sep + 'badsample_changepitch' + os.sep + song_name + os.sep + 'pitch.pitch'
        new_file = 'C:/DataBaker_songs/pitch_folder_exp3/' + singer_name + '_' + song_name + '_pshift150' +'.pitch'
        shutil.copyfile(old_file,new_file)

#////////////////////////////

ori_path = 'C:/Dataset/NHSS_Database/Data'
singer_names = ['F01','F02','F03','F04','F05','M01','M02','M03','M04','M05']

for singer_name in singer_names:
    print(singer_name)
    second_path = ori_path + os.sep + singer_name
    f_list = os.listdir(second_path)
    song_names = []

    for f in f_list:
        if f.startswith('S'):
            song_names.append(f)
            
    for song_name in song_names:
        old_file = second_path + os.sep + song_name + os.sep + 'pitch.pitch'
        new_file = 'C:/DataBaker_songs/pitch_folder_exp3/' + singer_name + '_' + song_name + '_pshift150' +'.pitch'
        shutil.copyfile(old_file,new_file)
