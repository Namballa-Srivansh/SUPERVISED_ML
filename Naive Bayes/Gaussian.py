import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

heart_df = pd.read_csv("CSV Files\heart.csv")
# print(heart_df.head())

X = heart_df.drop(columns = ["target"])
y = heart_df["target"]

# print(X.head(), y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)

y_pred = gnb_model.predict(X_test)

# Evaluation

print("recall score:", recall_score(y_test, y_pred))
print("accuracy score:", accuracy_score(y_test, y_pred))
print("precision score:", precision_score(y_test, y_pred))