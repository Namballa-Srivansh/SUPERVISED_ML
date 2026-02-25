import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

insuarance_data = pd.read_csv("insurance.csv")
# print(insuarance_data)

# Visualize

# sns.scatterplot(x=insuarance_data["bmi"], y=insuarance_data["charges"], hue=insuarance_data["smoker"])
# plt.show()

# X = insuarance_data.drop(columns = ["charges", "region"])
# y = insuarance_data["charges"]

# X["sex"] = X["sex"].map({"female": 1, "male": 0})
# X["smoker"] = X["smoker"].map({"yes": 1, "no": 0})

# print(X.head())

# ----------------------------------------Train Test Split-----------------------------------------

# X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.2, random_state=42)

# print(X_train.head())

# -----------------------------------------TRAIN MODEL-------------------------------------------

# model = LinearRegression()
# model.fit(X_train, y_train)

# print("Training Complete")

# ------------------------------------------PREDICT VALUES----------------------------------------

# y_pred = model.predict(X_test)
# print(y_pred)
# print(y_test)

# -------------------------------------------EVALUATE-------------------------------------------------

# r2 = r2_score(y_test, y_pred)
# print(f"r-squared: {r2}")

# n = X_test.shape[0]
# p = X_test.shape[1]

# adjusted_r2 = 1 - ((1-r2) * (n-1) / (n-p-1))
# print(f"adjusted_r-squared: {adjusted_r2}")
