
# read song names
#     song_names = ['S09']

for song_name in song_names:
    print((singer_name,song_name))
    if not os.path.exists(ori_path + os.sep + singer_name + os.sep + song_name + '/badsample'):
        os.makedirs(ori_path + os.sep + singer_name + os.sep + song_name + '/badsample')
