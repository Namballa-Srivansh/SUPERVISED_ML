import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor,plot_tree 
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.datasets import load_diabetes

df = load_diabetes(as_frame=True).frame

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = DecisionTreeRegressor(max_depth=7, min_samples_leaf=20)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("MSE train: ", mean_squared_error(y_train, y_pred_train))
print("MSE tests: ", mean_squared_error(y_test, y_pred_test))

print("r2 train: ", r2_score(y_train, y_pred_train))
print("r2 tests: ", r2_score(y_test, y_pred_test))

plt.figure(figsize=(18, 10))

plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    max_depth=1
)

plt.tight_layout()