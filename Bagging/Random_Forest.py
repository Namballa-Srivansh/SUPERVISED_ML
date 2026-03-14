import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

titanic = sns.load_dataset("titanic")

features = ["pclass", "sex", "fare", "embarked", "age"]
target = ["survived"]

imp_median = SimpleImputer(strategy="median")
titanic[["age"]] = imp_median.fit_transform(titanic[["age"]])

imp_freq = SimpleImputer(strategy="most_frequent")
titanic[["embarked"]] = imp_freq.fit_transform(titanic[["embarked"]])

le = LabelEncoder()

titanic["sex"] = le.fit_transform(titanic["sex"])
titanic["embarked"] = le.fit_transform(titanic["embarked"])

X = titanic[features]
y = titanic[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------------------Decision Tree------------------------------------------------
# Classic case of overfitting
# model = DecisionTreeClassifier()

# model.fit(X_train, y_train)

# y_pred_test = model.predict(X_test)
# y_pred_train = model.predict(X_train)

# print("Training Accuracy: ", accuracy_score(y_train, y_pred_train)*100, "%")
# print("Testing Accuracy: ", accuracy_score(y_test, y_pred_test)*100, "%")


# -----------------------------------------Random Forest-------------------------------------------------

rf = RandomForestClassifier(
    n_estimators=501,
    oob_score=True
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("OOB Score: ", rf.oob_score_*100, "%")
print("Testing Accuracy: ", accuracy_score(y_test, y_pred)*100, "%")