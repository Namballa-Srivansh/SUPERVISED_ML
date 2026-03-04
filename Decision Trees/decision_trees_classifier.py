import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier,plot_tree 
from sklearn.metrics import accuracy_score

titanic = sns.load_dataset("titanic")

titanic.head()
titanic.isnull().sum()

features = ["pclass", "sex", "fare", "embarked", "age"]
target = ["survived"]

# missing data

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
    X, y, test_size=0.2, random_state=42
)

# Decision Tree Model - No pruning

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("accuracy: ", accuracy_score(y_test, y_pred))

plt.figure(figsize=(18, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Died", "Survived"],
    filled=True,
    max_depth=2
)

plt.tight_layout()
plt.show()


# # Decision Tree with pre-pruning

# max_depths = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# for depth in max_depths:
#     model = DecisionTreeClassifier(max_depth=depth)
#     model.fit(X_train, y_train)

#     acc = model.score(X_test, y_test)
#     print(f"for depth = {depth}, accuracy = {acc}")

#     if depth == 7:
#         plt.figure(figsize=(18, 10))
#         plot_tree(
#             model,
#             feature_names=X.columns,
#             class_names=["Died", "Survived"],
#             filled=True,
#         )

#         plt.tight_layout()
#         plt.show()


# min_sample_splits = [10, 15, 20, 25, 30]

# for split in min_sample_splits:
#     model = DecisionTreeClassifier(max_depth=7, min_samples_split=split)
#     model.fit(X_train, y_train)

#     acc = model.score(X_test, y_test)
#     print(f"for sample split = {split}, accuracy = {acc}")

#     if split == 10:
#         plt.figure(figsize=(18, 10))
#         plot_tree(
#             model,
#             feature_names=X.columns,
#             class_names=["Died", "Survived"],
#             filled=True,
#         )

#         plt.tight_layout()
#         plt.show()


# # Decision Tree with post-pruning

full_tree = DecisionTreeClassifier(random_state=42)
full_tree.fit(X_train, y_train)

path = full_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

print(ccp_alphas)

# train our model for all alphas

tree = []

for alpha in ccp_alphas:
    model = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    model.fit(X_train, y_train)

    tree.append((model, alpha))

best_acc = 0
best_alpha = 0

for model, alpha in tree:
    curr_acc = model.score(X_test, y_test)
    if curr_acc > best_acc:
        best_acc = curr_acc
        best_alpha = alpha

print(best_acc)

best_model = DecisionTreeClassifier(ccp_alpha=best_alpha)
best_model.fit(X_train, y_train)

plt.figure(figsize=(18, 10))
plot_tree(
    best_model,
    feature_names=X.columns,
    class_names=["Died", "Survived"],
    filled=True,
)

plt.tight_layout()
plt.show()


print(best_model.score(X_test, y_test))