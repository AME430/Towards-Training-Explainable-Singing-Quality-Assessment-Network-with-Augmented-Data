import os
import re


def mapping(path,song_name):
    song_interval_path = path + os.sep + song_name

    with open(song_interval_path + os.sep + 'song' + '.TEXTGRID') as file:
        lines = file.readlines()
        str1 = 'item'
        str2 = '[2]'
        for i in range(len(lines)):
            if str1 in lines[i] and str2 in lines[i]:
                break
        lines = lines[i+6:-1] # delete some lines at start
        label = 0
        s = []
        phoneme_sing = []
        mid = []
        for i in range(len(lines)):
            if i % 4 == 0 and i != 0:
                phoneme_sing.append(mid)
                mid = []
            if i % 4 == 1 or i % 4 == 2:
                for j in lines[i]:
                    if str.isdigit(j) or j == ".":
                        s.append(j)
                s2 = ''.join(s)
                s = []
                mid.append(eval(s2))
            if i % 4 == 3:
                for j in lines[i]:
                    if label == 1 and j != '"':
                        s.append(j)
                    if j == '"':
                        label = label + 1
                s2 = ''.join(s)
                s = []
                mid.append(s2)
                label = 0
    if phoneme_sing[0][0] != 0:
        phoneme_sing[0][0] = 0

    mapping = open(path + os.sep + song_name + '/mapping.txt', 'w')
    for i in phoneme_sing:
        for j in i:
            mapping.write(str(j) + ' ')
        mapping.write('\n')
    mapping.close()
