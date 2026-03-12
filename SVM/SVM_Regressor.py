from sklearn.svm import SVR, LinearSVR
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn import datasets

df = datasets.load_diabetes(as_frame=True).frame

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

# model = SVR()

# model.fit(X_train, y_train_scaled)

# y_test_pred_scaled = model.predict(X_test)
# y_train_pred_scaled = model.predict(X_train)

# print("train r2: ", r2_score(y_train_scaled, y_train_pred_scaled))
# print("test r2: ", r2_score(y_test_scaled, y_test_pred_scaled))

# -----------------------------------------Hyperparameter tuning using GridSearchCV---------------------------------------------

# param_grid = {
#     "C": [1, 2, 5, 10, 50, 100],
#     "kernel": ["rbf", "linear"],
#     "epsilon": [0.001, 0.1, 0.2, 0.3, 0.5],
# }

# svr = SVR()

# grid_search = GridSearchCV(svr, param_grid, scoring="r2", cv=5)

# grid_search.fit(X_train, y_train_scaled)
# print("best params - ", grid_search.best_params_)

# best_model = grid_search.best_estimator_


# y_test_pred_scaled = best_model.predict(X_test)
# y_train_pred_scaled = best_model.predict(X_train)

# print("train r2: ", r2_score(y_train_scaled, y_train_pred_scaled))
# print("test r2: ", r2_score(y_test_scaled, y_test_pred_scaled))

# ---------------------------------------------Linear SVR--------------------------------------------------------------------

model = LinearSVR(C=10, epsilon=0.1, max_iter=5000)
model.fit(X_train, y_train_scaled)

y_test_pred_scaled = model.predict(X_test)
y_train_pred_scaled = model.predict(X_train)

print("train r2: ", r2_score(y_train_scaled, y_train_pred_scaled))
print("test r2: ", r2_score(y_test_scaled, y_test_pred_scaled))