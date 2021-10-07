import  os

mother_dir1 = 'C:/DataBaker_songs'
singer_names1 = ['DB-DM-001-F-001','DB-DM-002-F-002','DB-DM-003-F-003','DB-DM-004-M-001',
                'DB-DM-005-M-002','DB-DM-006-F-004','DB-DM-007-M-003','DB-DM-008-M-004']

mother_dir2 = 'C:/Dataset/NHSS_Database/Data'
singer_names2 = ['F01','F02','F03','F04','F05','M01','M02','M03','M04','M05']

bws_path1 = mother_dir1
bws_file1 = open(bws_path1 + os.sep + 'bws_file3.txt','w')

for singer_name in singer_names1:
    second_path = mother_dir1 + os.sep + singer_name + os.sep + 'Vox'
    f_list1 = os.listdir(second_path)
    Vox_list = []
    for f in f_list1:
        if os.path.splitext(f)[1] == '.wav':  # seperate with extension name
            Vox_list.append(f.split('.')[0])

    for song_name in Vox_list:
        thisline = singer_name + '_' + song_name + ',' + '1'
        bws_file1.write(thisline)
        bws_file1.write('\n')

        second_line = singer_name + '_' + song_name + '_pshift150' + ',' + '0.4'
        bws_file1.write(second_line)
        bws_file1.write('\n')

        third_line = singer_name + '_' + song_name + '_speech' + ',' + '-1'
        bws_file1.write(third_line)
        bws_file1.write('\n')

bws_file1.close()
########
# bws_path2 = mother_dir2
bws_path2 = mother_dir2
bws_file2 = open(bws_path2 + os.sep + 'bws_file3.txt','w')

for singer_name in singer_names2:
    second_path = mother_dir2+ os.sep + singer_name
    f_list2 = os.listdir(second_path)
    song_names = []

    for f in f_list2:
        if f.startswith('S'):
            song_names.append(f)

    for song_name in song_names:
        # thisline = singer_name + '_' + song_name + ',' + '1'
        thisline = singer_name + '_' + song_name + ',' + '1'
        bws_file2.write(thisline)
        bws_file2.write('\n')

        # second_line = singer_name + '_' + song_name + '_speech' + ',' + '-1'
        second_line = singer_name + '_' + song_name + '_pshift150' + ',' + '0.4'
        bws_file2.write(second_line)
        bws_file2.write('\n')

        third_line = singer_name + '_' + song_name + '_speech' + ',' + '-1'
        bws_file2.write(third_line)
        bws_file2.write('\n')

bws_file2.close()
