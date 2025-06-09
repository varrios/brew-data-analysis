import random

from sklearn.tree import export_text
from constants import CLEAN_DATAFILE_PATH
from utility.raport_helper_functions import load_recipe_data
from model.prepare_data import *
from sklearn.model_selection import GridSearchCV


def split_data(df, features):
    X = df[features]
    y = df['StyleGroup']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def predict(model, features, sample):
    X_sample = pd.DataFrame([sample], columns=features)
    return model.predict(X_sample)[0]

# def optimize_model(X_train, y_train):
#     param_grid = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [None, 10, 20, 30],
#         # 'min_samples_split': [2, 5, 10],
#         # 'min_samples_leaf': [1, 2, 4],
#         # 'bootstrap': [True, False]
#     }
#
#     rf = RandomForestClassifier(random_state=42)
#     grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
#
#     grid_search.fit(X_train, y_train)
#
#     print("Najlepsze parametry:", grid_search.best_params_)
#     return grid_search.best_estimator_
