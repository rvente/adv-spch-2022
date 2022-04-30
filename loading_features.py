# %%
import pandas as pd
from pathlib import Path

FEATURE_FILES = list(Path("./opensmile_features").iterdir())


# dfs = []
# for feature_file in FEATURE_FILES:
#     dfs.append(pd.read_csv(feature_file, sep=";"))  

files = pd.DataFrame(columns=["_filename"], data=FEATURE_FILES)
features = files._filename.map(lambda f: pd.read_csv(f, sep=";", header=None).iloc[1])
header = list(pd.read_csv(FEATURE_FILES[0], sep=";").columns)
features[0]

# %%
data_df = pd.DataFrame(data=features.values.tolist())
data_df.columns = header
data_df.name = FEATURE_FILES

# %%
data_df.to_csv("data_df.csv")
data_df

# %%
name_to_attribute = [
    "speaker", "session", "emotion", "start_time", "content"
]

for idx, name in enumerate(name_to_attribute):
    data_df[name] = data_df.name.map(lambda x: x.name.split("_")[idx])
# %%
from sklearn.ensemble import RandomForestClassifier

X = data_df[set(header) - {"name","frameTime"}]
X
# %%
y = data_df["emotion"]
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

confusion_matrix(y_test, y_pred)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from IPython.display import set_matplotlib_formats
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
