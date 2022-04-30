#!/usr/bin/env python
# coding: utf-8

# In[43]:


import parselmouth

import pandas as pd
import glob
import os.path

# list of wav files
actual_filenames = [
    'happy.wav',
    'angry.wav',
    'sad.wav',
    'fear.wav',
    'surprise.wav',
    'disgust.wav',
    'neutral.wav'
]


## Extract pitch features ##
# pitch floor = 75.0 Hz, pitch ceiling = 600.0 Hz

# Extract minimum pitch
def min_pitch(filepath):
    sound = parselmouth.Sound(filepath)
    pitch_file = parselmouth.praat.call(sound, "To Pitch", 0, 75, 600)
    min_pitch_value = parselmouth.praat.call(pitch_file, "Get minimum", 0, 0, "Hertz", "parabolic")
    return min_pitch_value

# Extract maximum pitch
def max_pitch(filepath):
    sound = parselmouth.Sound(filepath)
    pitch_file = parselmouth.praat.call(sound, "To Pitch", 0, 75, 600)
    max_pitch_value = parselmouth.praat.call(pitch_file, "Get maximum", 0, 0, "Hertz", "parabolic")
    return max_pitch_value

# Extract Mean pitch
def mean_pitch(filepath):
    sound = parselmouth.Sound(filepath)
    pitch_file = parselmouth.praat.call(sound, "To Pitch", 0, 75, 600)
    mean_pitch_value = parselmouth.praat.call(pitch_file, "Get mean", 0, 0, "Hertz")
    return mean_pitch_value

# Extract Sd pitch
def sd_pitch(filepath):
    sound = parselmouth.Sound(filepath)
    pitch_file = parselmouth.praat.call(sound, "To Pitch", 0, 75, 600)
    sd_pitch_value = parselmouth.praat.call(pitch_file, "Get standard deviation", 0, 0, "Hertz")
    return sd_pitch_value

## Extract intensity features ##

# Extract minimum intensity
def min_intensity(filepath):
    sound = parselmouth.Sound(filepath)
    intensity_file = parselmouth.praat.call(sound, "To Intensity", 100, 0)
    min_intensity_value = parselmouth.praat.call(intensity_file, "Get minimum", 0, 0, "parabolic")
    return min_intensity_value

# Extract maximum intensity
def max_intensity(filepath):
    sound = parselmouth.Sound(filepath)
    intensity_file = parselmouth.praat.call(sound, "To Intensity", 100, 0)
    max_intensity_value = parselmouth.praat.call(intensity_file, "Get maximum", 0, 0, "parabolic")
    return max_intensity_value

# Extract mean intensity
def mean_intensity(filepath):
    sound = parselmouth.Sound(filepath)
    intensity_file = parselmouth.praat.call(sound, "To Intensity", 100, 0)
    mean_intensity_value = parselmouth.praat.call(intensity_file, "Get mean", 0, 0, "energy")
    return mean_intensity_value

# Extract sd intensity
def sd_intensity(filepath):
    sound = parselmouth.Sound(filepath)
    intensity_file = parselmouth.praat.call(sound, "To Intensity", 100, 0)
    sd_intensity_value = parselmouth.praat.call(intensity_file, "Get standard deviation", 0, 0)
    return sd_intensity_value

## Extract jitter ##
# extract local jitter only
# set period floor = 0.0001s, period ceiling = 0.02s, maximum period factor = 1.3
def jitter(filepath):
    sound = parselmouth.Sound(filepath)
    jitter_file = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)
    jitter_value = parselmouth.praat.call(jitter_file, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    return jitter_value

## Extract shimmer ##
# extract local shimmer only
# and set period floor = 0.0001s, period ceiling to 0.02s, 
# maximum period factor = 1.3, maximum amplitude factor = 1.6

def shimmer(filepath):
    sound = parselmouth.Sound(filepath)
    shimmer_file = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)
    shimmer_value = parselmouth.praat.call([sound, shimmer_file], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return shimmer_value

# Extract harmonicity ##
# HNR (harmonics-to-noise ratio)
# extract harmonicity (cc)
# time step = 0.01, minimum pitch = 75Hz, silence threshold = 0.1, number of periods per window = 1.0
def harmonicity(filepath):
    sound = parselmouth.Sound(filepath)
    harmonicity_file = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
    mean_harmonicity_value = parselmouth.praat.call(harmonicity_file, "Get mean", 0, 0)
    return mean_harmonicity_value

# Extract speaking rate: number of syllables divided by utterance duration
import math
from parselmouth.praat import call

def speech_rate(filepath):
    silencedb = -25
    mindip = 2
    minpause = 0.3

    # print a single header line with column names and units
    # cols = ['soundname', 'nsyll', 'npause', 'dur(s)', 'phonationtime(s)', 'speechrate(nsyll / dur)', 'articulation '
    #        'rate(nsyll / phonationtime)', 'ASD(speakingtime / nsyll)']
    # df = pd.DataFrame(columns = cols)
    
    sound = parselmouth.Sound(filepath)
    originaldur = sound.get_total_duration()
    intensity = sound.to_intensity(50)
    start = call(intensity, "Get time from frame number", 1)
    nframes = call(intensity, "Get number of frames")
    end = call(intensity, "Get time from frame number", nframes)
    min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
    max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")

    # get .99 quantile to get maximum (without influence of non-speech sound bursts)
    max_99_intensity = call(intensity, "Get quantile", 0, 0, 0.99)

    # estimate Intensity threshold
    threshold = max_99_intensity + silencedb
    threshold2 = max_intensity - max_99_intensity
    threshold3 = silencedb - threshold2
    if threshold < min_intensity:
        threshold = min_intensity

    # get pauses (silences) and speakingtime
    textgrid = call(intensity, "To TextGrid (silences)", threshold3, minpause, 0.1, "silent", "sounding")
    silencetier = call(textgrid, "Extract tier", 1)
    silencetable = call(silencetier, "Down to TableOfReal", "sounding")
    npauses = call(silencetable, "Get number of rows")
    speakingtot = 0
    for ipause in range(npauses):
        pause = ipause + 1
        beginsound = call(silencetable, "Get value", pause, 1)
        endsound = call(silencetable, "Get value", pause, 2)
        speakingdur = endsound - beginsound
        speakingtot += speakingdur

    intensity_matrix = call(intensity, "Down to Matrix")
    # sndintid = sound_from_intensity_matrix
    sound_from_intensity_matrix = call(intensity_matrix, "To Sound (slice)", 1)
    # use total duration, not end time, to find out duration of intdur (intensity_duration)
    # in order to allow nonzero starting times.
    intensity_duration = call(sound_from_intensity_matrix, "Get total duration")
    intensity_max = call(sound_from_intensity_matrix, "Get maximum", 0, 0, "Parabolic")
    point_process = call(sound_from_intensity_matrix, "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")
    # estimate peak positions (all peaks)
    numpeaks = call(point_process, "Get number of points")
    t = [call(point_process, "Get time from index", i + 1) for i in range(numpeaks)]

    # fill array with intensity values
    timepeaks = []
    peakcount = 0
    intensities = []
    for i in range(numpeaks):
        value = call(sound_from_intensity_matrix, "Get value at time", t[i], "Cubic")
        if value > threshold:
            peakcount += 1
            intensities.append(value)
            timepeaks.append(t[i])

    # fill array with valid peaks: only intensity values if preceding
    # dip in intensity is greater than mindip
    validpeakcount = 0
    currenttime = timepeaks[0]
    currentint = intensities[0]
    validtime = []

    for p in range(peakcount - 1):
        following = p + 1
        followingtime = timepeaks[p + 1]
        dip = call(intensity, "Get minimum", currenttime, timepeaks[p + 1], "None")
        diffint = abs(currentint - dip)
        if diffint > mindip:
            validpeakcount += 1
            validtime.append(timepeaks[p])
        currenttime = timepeaks[following]
        currentint = call(intensity, "Get value at time", timepeaks[following], "Cubic")

    # Look for only voiced parts
    pitch = sound.to_pitch_ac(0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)
    voicedcount = 0
    voicedpeak = []

    for time in range(validpeakcount):
        querytime = validtime[time]
        whichinterval = call(textgrid, "Get interval at time", 1, querytime)
        whichlabel = call(textgrid, "Get label of interval", 1, whichinterval)
        value = pitch.get_value_at_time(querytime) 
        if not math.isnan(value):
            if whichlabel == "sounding":
                voicedcount += 1
                voicedpeak.append(validtime[time])

    # calculate time correction due to shift in time for Sound object versus
    # intensity object
    timecorrection = originaldur / intensity_duration

    # Insert voiced peaks in TextGrid
    call(textgrid, "Insert point tier", 1, "syllables")
    for i in range(len(voicedpeak)):
        position = (voicedpeak[i] * timecorrection)
        call(textgrid, "Insert point", 1, position, "")

    # return speaking rate
    speakingrate = voicedcount / originaldur
    
    return speakingrate    

if __name__ == "__main__":
    ###### MY FEATURES #####
    df = pd.read_csv("/Users/jiyoungchoi/Dropbox/TCColumbiaUniversity/1st_yr_Spring2022/CS6998 Spoken Language Processing/myfeatures.csv")
    df['filename'] = "/Users/jiyoungchoi/Dropbox/TCColumbiaUniversity/1st_yr_Spring2022/CS6998 Spoken Language Processing/wavfiles/" + pd.Series(actual_filenames)
    df["Jitter"] = df.filename.map(jitter)
    df["Shimmer"] = df.filename.map(shimmer)
    df["HNR"] = df.filename.map(harmonicity)
    df["Speaking Rate"] = df.filename.map(speech_rate)
    print(df)
    df.to_csv("/Users/jiyoungchoi/Dropbox/TCColumbiaUniversity/1st_yr_Spring2022/CS6998 Spoken Language Processing/myfeatures_final.csv", index=False)
    
    ###### MSP samples ######
    df2 = pd.read_csv("/Users/jiyoungchoi/Dropbox/TCColumbiaUniversity/1st_yr_Spring2022/CS6998 Spoken Language Processing/msp-features.csv")
    df2['filename'] = "/Users/jiyoungchoi/Dropbox/TCColumbiaUniversity/1st_yr_Spring2022/CS6998 Spoken Language Processing/MSP_samples/" + pd.Series(actual_filenames)
    df2["Min Pitch"] = df2.filename.map(min_pitch)
    df2["Max Pitch"] = df2.filename.map(max_pitch)
    df2["Mean Pitch"] = df2.filename.map(mean_pitch)
    df2["Sd Pitch"] = df2.filename.map(sd_pitch)
    df2["Min Intensity"] = df2.filename.map(min_intensity)
    df2["Max Intensity"] = df2.filename.map(max_intensity)
    df2["Mean Intensity"] = df2.filename.map(mean_intensity)
    df2["Sd Intensity"] = df2.filename.map(sd_intensity)
    df2["Jitter"] = df2.filename.map(jitter)
    df2["Shimmer"] = df2.filename.map(shimmer)
    df2["HNR"] = df2.filename.map(harmonicity)
    df2["Speaking Rate"] = df2.filename.map(speech_rate)
    print(df2)
    df2.to_csv("/Users/jiyoungchoi/Dropbox/TCColumbiaUniversity/1st_yr_Spring2022/CS6998 Spoken Language Processing/msp-features_final.csv", index=False)
    

