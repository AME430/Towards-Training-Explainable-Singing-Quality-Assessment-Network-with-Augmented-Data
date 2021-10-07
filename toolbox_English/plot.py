
from toolbox_English.pitch_histogram import CreateNoteHistogram,extract_time_pitch,PitchMedianSubtraction,GridMap,GetFinerNoteHistogram,plotHistogram
#先要有pitch

def plotph(pitchpath):
    notes1 = CreateNoteHistogram(pitchpath)
    plotHistogram(notes1)
