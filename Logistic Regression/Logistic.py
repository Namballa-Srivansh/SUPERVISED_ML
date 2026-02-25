import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import r2_score

heart_df = pd.read_csv("C:\\Users\\sriva\\Desktop\\ML\\Logistic Regression\\heart.csv")
# print(heart_df.head())
# # print(heart_df.columns)
# # print(heart_df.info())
# print(heart_df["target"].nunique())

X = heart_df.drop(columns=["target"])
y = heart_df["target"]

# ---------------------------------------------Train Test split-------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("accuracy", accuracy_score(y_test, y_pred) * 100, "%")
print("precision", precision_score(y_test, y_pred) * 100, "%")