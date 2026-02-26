import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score

insuarance_data = pd.read_csv("CSV Files\insurance.csv")

X = insuarance_data.drop(columns = ["charges"])
y = insuarance_data["charges"]

X = pd.get_dummies(X, columns=["region"], drop_first=False, dtype=int) # Encoding

X["sex"] = X["sex"].map({"female": 1, "male": 0})
X["smoker"] = X["smoker"].map({"yes": 1, "no": 0})

X["age_smoker"] = X["age"] * X["smoker"]
X["bmi_smoker"] = X["bmi"] * X["smoker"]

X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42)

# lasso_model = Lasso(alpha = 0.5)
# lasso_model.fit(X_train, y_train)

# y_pred = lasso_model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)

# print(f"MSE: {mse}")

a = [0.001, 0.1, 1, 2, 5, 10, 20, 30, 40, 50, 100]

lasso_cv_model = LassoCV(
    alphas=a,
    cv=5,
    max_iter=1000,
    random_state=42
)

lasso_cv_model.fit(X_train, y_train)
# print(f"Best apha: {lasso_cv_model.alpha_}")

y_pred = lasso_cv_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse}")
print(f"r2: {r2}")