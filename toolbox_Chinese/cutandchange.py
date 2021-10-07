import librosa
import wave
import numpy as np
from toolbox_Chinese.speech_cat import merge_speech
import os
import random
import math

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
    w = wave.open(path + os.sep + 'Vox' + os.sep + song_name + '.wav', "rb")
    params = w.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    f = song_name + '.wav'

    idx = random.sample(range(0,len(starttime)),math.floor(len(starttime)*1))
    idx.sort()
    str_data = w.readframes(nframes)
    wave_data = np.frombuffer(str_data,dtype=np.short)
    # head = wave_data[0:int(eval(starttime[idx[0]])*framerate)]
    # f0 = wave.open(
    #     path + '/badsample_changepitch/' + song_name + os.sep + 'notescut' + os.sep + "note_0" + "-" + f, "wb")
    # # 配置声道数、量化位数和取样频率
    # f0.setnchannels(nchannels)
    # f0.setsampwidth(sampwidth)
    # f0.setframerate(framerate)
    # f0.writeframes(head.tostring())

    for i in range(len(idx)):
        str_data1 = wave_data[int(eval(starttime[idx[i]])*framerate):int(eval(starttime[idx[i]])*framerate) + int(eval(duration[idx[i]])*framerate)]
        # if i != len(idx)-1:
        #     str_data2 = wave_data[int(eval(starttime[idx[i]])*framerate) + int(eval(duration[idx[i]])*framerate) : int(eval(starttime[idx[i+1]])*framerate)]
        # else:
        #     str_data2 = wave_data[int(eval(starttime[idx[i]])*framerate) + int(eval(duration[idx[i]])*framerate) : nframes]
        w.close()

        f1 = wave.open(
            path + '/badsample_changepitch/' + song_name + os.sep + 'notescut' + os.sep + str(i + 1) + "_" + f, "wb")
        f1.setnchannels(nchannels)
        f1.setsampwidth(sampwidth)
        f1.setframerate(framerate)
        # 将wav_data转换为二进制数据写入文件
        # if duration[i] >
        f1.writeframes(str_data1.tostring())
        # f1.tobytes(str_data1.tostring())

        # f2 = wave.open(
        #     path + '/badsample_changepitch/' + song_name + os.sep + 'notescut' + os.sep + "note_" + str(i + 1) + "-" + f, "wb")
        # # 配置声道数、量化位数和取样频率
        # f2.setnchannels(nchannels)
        # f2.setsampwidth(sampwidth)
        # f2.setframerate(framerate)
        # f2.writeframes(str_data2.tostring())

