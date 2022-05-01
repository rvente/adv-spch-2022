# %%
import numpy as np
import seaborn as sns
from IPython.display import display
import pandas as pd
from pathlib import Path
import parselmouth
import matplotlib.pyplot as plt

FEATURE_FILES = list(Path("./opensmile_features").iterdir())

# %%
# TODO: set pitch mode to autocorrelation


def min_pitch(sound):
    pitch_file = parselmouth.praat.call(sound, "To Pitch", 0, 75, 600)
    return parselmouth.praat.call(pitch_file, "Get minimum", 0, 0, "Hertz", "parabolic")


def max_pitch(sound):
    pitch_file = parselmouth.praat.call(sound, "To Pitch", 0, 75, 600)
    return parselmouth.praat.call(pitch_file, "Get maximum", 0, 0, "Hertz", "parabolic")


def mean_pitch(sound):
    pitch_file = parselmouth.praat.call(sound, "To Pitch", 0, 75, 600)
    return parselmouth.praat.call(pitch_file, "Get mean", 0, 0, "Hertz")


def sd_pitch(sound):
    pitch_file = parselmouth.praat.call(sound, "To Pitch", 0, 75, 600)
    return parselmouth.praat.call(pitch_file, "Get standard deviation", 0, 0, "Hertz")


def min_intensity(sound):
    intensity_file = parselmouth.praat.call(sound, "To Intensity", 100, 0)
    return parselmouth.praat.call(intensity_file, "Get minimum", 0, 0, "parabolic")


def max_intensity(sound):
    intensity_file = parselmouth.praat.call(sound, "To Intensity", 100, 0)
    return parselmouth.praat.call(intensity_file, "Get maximum", 0, 0, "parabolic")


def mean_intensity(sound):
    intensity_file = parselmouth.praat.call(sound, "To Intensity", 100, 0)
    return parselmouth.praat.call(intensity_file, "Get mean", 0, 0, "energy")


def sd_intensity(sound):
    intensity_file = parselmouth.praat.call(sound, "To Intensity", 100, 0)
    return parselmouth.praat.call(intensity_file, "Get standard deviation", 0, 0)
# %%


FEATURE_FILES = list(Path("./hw3_speech_files").iterdir())
df = pd.DataFrame(columns=["filepath"], data=FEATURE_FILES)

features_to_extract = [
    "Min Pitch",
    "Max Pitch",
    "Mean Pitch",
    "Min Intensity",
    "Max Intensity",
    "Mean Intensity",
]

extraction_functions = [
    min_pitch,
    max_pitch,
    mean_pitch,
    min_intensity,
    max_intensity,
    mean_intensity,

]
df["filename"] = df.filepath.map(lambda x: str(x.absolute()))
df["sound"] = df.filename.map(parselmouth.Sound).map(lambda sound: sound.extract_left_channel())

for feature_name, extractor in zip(features_to_extract, extraction_functions):
    df[feature_name] = df.sound.map(extractor)

name_to_attribute = [
    "speaker", "session", "emotion", "start_time", "content"
]
# %%

for idx, name in enumerate(name_to_attribute):
    df[name] = df.filepath.map(lambda x: x.name.split("_")[idx])
# %%
df
# %%
# %%
df_std = df.copy()
speakers = pd.unique(df["speaker"])
for speaker in speakers:
    df_usr = df[df.speaker == speaker]
    df_neutral_stats = df_usr[df_usr.emotion == "neutral"].describe()
    # df_neutral_stats = df_usr.describe()
    df_subtracted_mean = df_usr - df_neutral_stats.loc["mean"]
    df_normalized = (df_subtracted_mean /
                     df_neutral_stats.loc["std"]).dropna(axis="columns")
    # display(df_normalized)
    df_std.loc[df.speaker == speaker, df_normalized.columns] = df_normalized

# %%
df_std.describe().round(5)

# %%
# %%
for df_ in [df, df_std]:
    for feature in features_to_extract:
        sns.barplot(
            y=feature,
            x="emotion",
            data=df_,
            capsize=.2,
            ci="sd",
            estimator=np.mean
        )
        plt.xticks(rotation=45)
        plt.show()
# %%
