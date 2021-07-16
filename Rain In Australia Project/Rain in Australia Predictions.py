from sklearn.metrics._plot.roc_curve import plot_roc_curve
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import resample
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from scipy import stats

# COLOR PALLETE:
# sns.color_palette("rocket")

# Read in data (version of data read  for VS Code)
df = pd.read_csv("./weatherAUS.csv")

print("Data: (rows, columns)")
print(df.shape)

# Exploratory Data Analysis (EDA) - Data cleaning:
# Missing labels
df.dropna(subset=["RainTomorrow"], inplace=True)

# Duplicate data
df.drop_duplicates(inplace=True)

# Features missing a large proportion of data
msno.matrix(df)
plt.title("Missing Values in Data")
# plt.show()

# After looking at the missing values chart, see which features should be dropped
df.drop(["Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"], axis=1, inplace=True)

# Highly correlated features (& reduce dimensionality)
plt.subplots(figsize=(18, 18))
corrMatrix = sns.heatmap(
    df.corr(),
    cmap=sns.color_palette("rocket").reverse(),
    annot=True,
    square=True,
    mask=np.tril(df.corr()),
)
plt.title("Correlation Matrix")
# plt.show()

# After looking at the correlation matrix, see which features should be dropped
df.drop(["Temp9am", "Pressure9am"], axis=1, inplace=True)


# Outliers (outside of 3 standard deviations)
# Z-score
plt.figure(figsize=(20, 10))
sns.boxenplot(data=df, palette=sns.color_palette("rocket"))
# plt.show()

zScore = np.abs(stats.zscore(df._get_numeric_data(), nan_policy="omit"))
df = df[(zScore < 4).all(axis=1)]

# Same plot after dropping outliers
plt.figure(figsize=(20, 10))
sns.boxenplot(data=df, palette=sns.color_palette("rocket"))
plt.title("Data without Outliers (Removed anything 3 SDs away)")
# plt.show()

# Categorical features (change to numeric)
print(df.info())
df["Date"] = pd.to_datetime(df["Date"])
df["Date"] = df["Date"].apply(lambda d: d.dayofyear)


encoder = LabelEncoder()
for cat in df.columns[df.dtypes == "object"]:
    df[cat].fillna(df[cat].mode()[0], inplace=True)
    df[cat] = encoder.fit_transform(df[cat])

# Features missing some data
for val in df.columns[df.dtypes == "float64"]:
    df[val].fillna(df[val].median(), inplace=True)

plt.figure(figsize=(16, 6))

# Imbalanced data set
# more days without rain than with rain
plt.subplot(1, 2, 1)
sns.countplot(x=df["RainTomorrow"], palette=sns.color_palette("rocket"))
plt.title("Unbalanced Data Set")
# plt.show()

# Oversample days with rain
no = df[df["RainTomorrow"] == 0]
yes = df[df["RainTomorrow"] == 1]
yes_oversampled = resample(yes, replace=True, n_samples=len(no), random_state=2021)
df = pd.concat([no, yes_oversampled])

plt.subplot(1, 2, 2)
sns.countplot(x=df["RainTomorrow"], palette=sns.color_palette("rocket"))
plt.title("Balanced Data Set")
# plt.show()

# Feature values need to be scaled
scaled = MinMaxScaler().fit_transform(df)  # Change to range 0 - 1
df = pd.DataFrame(
    scaled, columns=df.columns
)  # Convert the scaled numpy array back into a dataframe

plt.figure(figsize=(16, 6))
sns.boxenplot(data=df, palette=sns.color_palette("rocket"))
plt.xticks(rotation=60)
# plt.show()

# Classifiers used for voting
vClassifiers = [
    ("dt", DecisionTreeClassifier(max_depth=64)),
    ("nbc", MultinomialNB()),
    ("rfc", RandomForestClassifier()),
]

# NOTE: Imported HW2 - following general setup for the project
classifiers = {
    # "Decision Tree": DecisionTreeClassifier(max_depth=64),  # fast, decent results
    # "AdaBoost Classifier": AdaBoostClassifier(n_estimators=100, learning_rate=1),
    # "Logistic Regression": LogisticRegression(),  # fast, poor results
    # "Multilayer Perceptron": MLPClassifier(
    #     hidden_layer_sizes=(32, 32), activation="relu", solver="adam", max_iter=128
    # ),  # slow, poor results
    # "Naive Bayes Classifier": MultinomialNB(),  # fast, poor results
    # "Random Forest Classifier": RandomForestClassifier(),  # fast, best results
    # "Support Vector Machine": SVC(kernel="linear"),  # slow, poor results
    "Voting Classifier": VotingClassifier(
        estimators=vClassifiers, voting="soft"
    ),  # fast, good results
}

features = df.drop(["RainTomorrow"], axis=1)
labels = df["RainTomorrow"]
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, shuffle=True
)
for name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    # Analyze Performance of models:
    # Classification Report (Precision, Recall, F1 scores), ROC curve, Confusion Matrix
    print(name)
    print(classification_report(y_test, predicted, digits=2))
    plot_roc_curve(classifier, X_test, y_test)
    plot_confusion_matrix(classifier, X_test, y_test, cmap="RdPu", normalize="all")
    plt.show()
