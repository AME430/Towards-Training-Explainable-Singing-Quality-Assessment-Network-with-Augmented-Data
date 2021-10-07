import os
import scipy.io.wavfile
import  librosa
import  wave

mother_dir1 = 'C:/DataBaker_songs'
singer_names1 = ['DB-DM-002-F-002']

pitch_folder =  mother_dir1 + os.sep +  'pitch_folder_exp1'

for singer_name in singer_names1:
    second_path = mother_dir1 + os.sep + singer_name + os.sep + 'Vox'
    f_list = os.listdir(second_path)
    Vox_list = []
    for f in f_list:
        if os.path.splitext(f)[1] == '.wav': # seperate with extension name
            Vox_list.append(f.split('.')[0])

    train_tobedumped = []
    for song in Vox_list:
        wavfile1 = second_path + os.sep + song + '.wav'
        # result = wave.open(wavfile1)
        y,sr = librosa.load(wavfile1,sr = None)
        y = (y * 32768).astype(int)
        # new = result.getsampwidth()
        # fs, audio = scipy.io.wavfile.read(wavfile1)
        # ph_notes = CreateNoteHistogram(original_pitch_file)
        open(mother_dir1 + os.sep + 'test.wav','w')
        scipy.io.wavfile.write(mother_dir1 + os.sep + 'test.wav',sr,y)
        fs, audio = scipy.io.wavfile.read(mother_dir1 + os.sep + 'test.wav')
        train_tobedumped.append({'audio': [audio / 32768.0, fs], 'pitch_histogram': [11], 'ratings': [1]})