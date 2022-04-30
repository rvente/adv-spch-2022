Submission

## Feature Analysis


## Classification Experiments


## Error Analysis


```
sound_angry = parselmouth.Sound("angry.wav")

pitch_file = parselmouth.praat.call(sound_angry, "To Pitch", 0, 75, 600)

min_pitch = parselmouth.praat.call(pitch_file, "Get minimum", 0, 0, "Hertz", "parabolic")
