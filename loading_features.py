# %%
import pandas as pd
from pathlib import Path

FEATURE_FILES = list(Path("./opensmile_features").iterdir())
CACHED_FILENAME = "data_df.pkl"
LOAD_FEATURES = not Path(CACHED_FILENAME).is_file()

# dfs = []
# for feature_file in FEATURE_FILES:
#     dfs.append(pd.read_csv(feature_file, sep=";"))  

if LOAD_FEATURES:
    files = pd.DataFrame(columns=["_filename"], data=FEATURE_FILES)
    features = files._filename.map(lambda f: pd.read_csv(f, sep=";", header=None).iloc[1])
    header = list(pd.read_csv(FEATURE_FILES[0], sep=";").columns)
    features[0]
    df = pd.DataFrame(data=features.values.tolist())
    df.columns = header
    df.name = FEATURE_FILES
    df.to_pickle(CACHED_FILENAME)
    df.to_csv(CACHED_FILENAME+".csv")
else:
    df = pd.read_pickle(CACHED_FILENAME)

df
# %%
name_to_attribute = [
    "speaker", "session", "emotion", "start_time", "content"
]

for idx, name in enumerate(name_to_attribute):
    df[name] = df.name.map(lambda x: x.name.split("_")[idx])
# %%
from sklearn.ensemble import RandomForestClassifier

trainable_features = set(df.columns) - {"name","frameTime"} - set(name_to_attribute)
label = "emotion"
df = df.astype(float, errors="ignore")
df["speaker_cat"] = df["speaker"].astype('category').cat.codes
X = df[trainable_features]
X
# %%
y = df[label]
y
# %%
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = RandomForestClassifier(min_samples_leaf=3)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
clf_rprt = classification_report(y_test, y_pred)
print(clf_rprt)


# %%
from sklearn.metrics import confusion_matrix

# confusion_matrix(y_test, y_pred)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from IPython.display import set_matplotlib_formats, display
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
set_matplotlib_formats('svg')
# sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sns.set_context('notebook')
sns.set_style("ticks")


# %%

predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
# %%

sns.set(rc = {'figure.figsize':(15,15)})

df_cm = pd.DataFrame(data=cm, index=clf.classes_, columns=clf.classes_)
cm_plot = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, ) # font size
cm_plot.set(xlabel="Predicted Label", ylabel='True Label')



# %%

# %%

# %%

speakers = pd.unique(df["speaker"])


df_std = df.copy()
speakers = pd.unique(df["speaker"])
for speaker in speakers:
    df_usr = df[df.speaker == speaker]
    # df_neutral_stats = df_usr[df_usr.emotion == "neutral"][trainable_features].astype(float).describe()
    df_neutral_stats = df_usr[trainable_features].astype(float).describe()
    df_subtracted_mean = df_usr[trainable_features].astype(float) - df_neutral_stats.loc["mean"]
    df_normalized = (df_subtracted_mean /
                     df_neutral_stats.loc["std"]).dropna(axis="columns")
    df_std.loc[df.speaker == speaker, df_normalized.columns] = df_normalized

# %%
for df_ in [df_std]:
    classification_reports = []
    confusion_matrices = []
    for speaker in speakers:
        test_mask = df.speaker == speaker
        df_test = df_[test_mask]
        df_train = df_[~test_mask]

        clf = RandomForestClassifier(min_samples_leaf=3, random_state=42)
        # clf = LinearDiscriminantAnalysis()
        # clf = DecisionTreeClassifier(min_samples_leaf=30, random_state=42)
        selected_features = trainable_features | {"speaker_cat"}
        clf.fit(df_train[selected_features], df_train[label])
        # assert that absolutely no training instances are in the test set
        df_test_speakers = df_test.speaker_cat.describe().T
        assert(
            df_test_speakers.loc["mean"]
            == df_test_speakers.loc["max"]
            == df_test_speakers.loc["min"])
        pred = clf.predict(df_test[selected_features])
        clf_rprt = classification_report(df_test[label], pred, output_dict=True)
        clf_df = pd.DataFrame(clf_rprt)
        classification_reports.append(clf_df)
        display(clf_df)

        cm = confusion_matrix(df_test[label], pred, labels=clf.classes_)
        df_cm = pd.DataFrame(data=cm, index=clf.classes_, columns=clf.classes_)
        confusion_matrices.append(df_cm)


    average_clf_report = sum(classification_reports) / len(classification_reports)
    print(average_clf_report.T.round(4).to_markdown())


    average_confusion_matrix = sum(confusion_matrices) / len(confusion_matrices)
    cm_plot = sns.heatmap(average_confusion_matrix, annot=True, annot_kws={"size": 16}, ) # font size
    cm_plot.set(xlabel="Predicted Label", ylabel='True Label')
    plt.show()
    # %%

    # %%

# %%

# %%
