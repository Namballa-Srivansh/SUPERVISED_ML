import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

heart_df = pd.read_csv("C:\\Users\\sriva\\Desktop\\ML\\Logistic Regression\\heart.csv")

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

cm = confusion_matrix(y_test, y_pred)
print(cm)
# print(classification_report(y_test, y_pred))
print("recall score", recall_score(y_test, y_pred))
print("F1 score", f1_score(y_test, y_pred))