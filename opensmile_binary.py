import subprocess
from pathlib import Path

SMILE_BIN="opensmile/build/progsrc/smilextract/SMILExtract"
CONFIG="opensmile/config/is09-13/IS09_emotion.conf"
FILE_LIST=Path("hw3_speech_files/").glob("*.wav")
OUTPUT_DIR=Path("opensmile_features/")
'-instname "{filename}"'

for wav_file in FILE_LIST:
    subprocess.call([SMILE_BIN, "-C", CONFIG, "-I", wav_file, "-csvoutput", OUTPUT_DIR/wav_file.name.replace(".wav", ".csv")])