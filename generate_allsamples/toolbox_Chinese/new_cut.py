import librosa
import wave
import numpy as np
from toolbox_Chinese.speech_cat import merge_speech
import os

def cutandchange(path,song_name):
    file = open(path + '/badsample_changepitch/' + song_name + '/mapping.txt') # load mapping list
    lines = file.readlines()
    starttime = []
    duration = []
    for line in lines:
        phone = line.split(" ")[0:3]
        starttime.append(phone[0])
        duration.append(phone[1])

    if not os.path.exists(path + '/badsample_changepitch/' + song_name + os.sep + 'notescut'):
        os.mkdir(path + '/badsample_changepitch/' + song_name + os.sep + 'notescut')
    write_folder = path + '/badsample_changepitch/' + song_name
    y, sr = librosa.load(path + os.sep + 'Vox' + os.sep + song_name + '.wav' ,sr=None)
    f = song_name + '.wav'
    for i in range(len(starttime)):
        point1 = int(eval(starttime[i]) * sr)
        point2 = int((eval(starttime[i])+eval(duration[i]))*sr)
        b = y[point1:point2]
        librosa.output.write_wav(path + '/badsample_changepitch/' + song_name + os.sep + 'notescut' + os.sep + str(i + 1) + "_" + f, b,sr)