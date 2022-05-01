Submission

## Feature Analysis

1. n the nominalization plots, elation and hot-anger showed the highest values across features except for the min intensity. 

2. In the min intensity plots in both non- and normalization plots, hot-anger is one of the two lowest values.

3. In the mean intensity graph in the normalization plot, only anxiety showed a negative value. In the mean intensity graph in the non-normalization plot, anxiety showed the lowest values, which is comparable with neutral.

4. In the min pitch graph in the normalization plot, the two highest value is hot-anger. However, in the corresponding graph in the non-normalization plot, the highest value is panic.

5. Boredom had a high value in the min intensity in both non- and normalization plots, whereas values of boredom in other features were relatively low.
6. The values of disgust were high in the max intensity and max pitch features in the non-normalization plots, but not in the normalization plots.
7. Hot anger showed higher values than cold anger across features in both non- and normalization plots, except for the min intensity feature


## Classification Experiments


## Error Analysis


```
sound_angry = parselmouth.Sound("angry.wav")

pitch_file = parselmouth.praat.call(sound_angry, "To Pitch", 0, 75, 600)

min_pitch = parselmouth.praat.call(pitch_file, "Get minimum", 0, 0, "Hertz", "parabolic")
