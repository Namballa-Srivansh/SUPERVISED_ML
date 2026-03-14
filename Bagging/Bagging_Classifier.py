from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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

# ----------------------------------------------------Decision Tree-------------------------------------------------------------------

# base_model = DecisionTreeClassifier()

# bagging = BaggingClassifier(
#     base_model,
#     n_estimators=201
# )

# bagging.fit(X_train, y_train)

# y_pred = bagging.predict(X_test)

# print("accuracy: ", accuracy_score(y_test, y_pred))

# ------------------------------------------------------Logical Regression----------------------------------------------------------------

# base_model = LogisticRegression(max_iter=1000)

# bagging = BaggingClassifier(
#     base_model,
#     n_estimators=201
# )

# bagging.fit(X_train, y_train)

# y_pred = bagging.predict(X_test)

# print("accuracy: ", accuracy_score(y_test, y_pred))

# ------------------------------------------------------SVC-------------------------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

base_model = SVC()

bagging = BaggingClassifier(
    base_model,
    n_estimators=201
)

bagging.fit(X_train, y_train)

y_pred = bagging.predict(X_test)

print("accuracy: ", accuracy_score(y_test, y_pred))