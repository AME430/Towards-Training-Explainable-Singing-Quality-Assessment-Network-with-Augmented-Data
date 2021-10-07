import os
import re


def mapping(path,song_name):
    # PATH = 'C:/DataBaker_songs/DB-DM-001-F-001'
    song_interval_path = path + os.sep + 'interval'

    with open(song_interval_path + os.sep + song_name + '.interval') as file:
        lines = file.readlines()
        lines = lines[12:-1] # delete some lines at start
        label = 0
        mid = []
        phoneme_sing = []
        for line in lines:
            if label == 2:
                mid.append(line.split('"')[1])
            else:
                mid.append(eval(line))
            label = label + 1
            if label == 3:
                label = 0
                mid[1] = mid[1] - mid[0]
                phoneme_sing.append(mid)
                mid = []
    if not os.path.exists(path + '/badsample_changepitch/' + song_name):
        os.mkdir(path + '/badsample_changepitch/' + song_name)

    mapping = open(path + '/badsample_changepitch/' + song_name + '/mapping.txt', 'w')
    for i in phoneme_sing:
        for j in i:
            mapping.write(str(j) + ' ')
        mapping.write('\n')
    mapping.close()
