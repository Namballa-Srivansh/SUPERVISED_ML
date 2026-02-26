import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

insuarance_data = pd.read_csv("CSV Files\heart.csv")


# ----------------------------------------------One Hot Encoding----------------------------------------

# X = insuarance_data.drop(columns = ["charges"])
# y = insuarance_data["charges"]

# X = pd.get_dummies(X, columns=["region"], drop_first=False, dtype=int) # Encoding

# X["sex"] = X["sex"].map({"female": 1, "male": 0})
# X["smoker"] = X["smoker"].map({"yes": 1, "no": 0})

# # print(X.head())

# X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.2, random_state=42)

# model = LinearRegression()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# # print(y_pred)

# r2 = r2_score(y_test, y_pred)
# print(f"r-squared: {r2}")

# -----------------------------------------------Interaction Features------------------------------------------

X = insuarance_data.drop(columns = ["charges"])
y = insuarance_data["charges"]

X = pd.get_dummies(X, columns=["region"], drop_first=False, dtype=int) # Encoding

X["sex"] = X["sex"].map({"female": 1, "male": 0})
X["smoker"] = X["smoker"].map({"yes": 1, "no": 0})

X["age_smoker"] = X["age"] * X["smoker"]
X["bmi_smoker"] = X["bmi"] * X["smoker"]

X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# print(y_pred)

r2 = r2_score(y_test, y_pred)
# print(f"r-squared: {r2}")

# -----------------------------------------------Underfit Vs Overfit---------------------------------------------
# r2 training is low & r2 testing is also low -> underfit
# r2 training >> r2 testing is also low -> overfit

y_train_pred = model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)

print(f"training data r2: {r2_train}")
print(f"test data r2: {r2}")