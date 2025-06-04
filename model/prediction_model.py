import random

from constants import CLEAN_DATAFILE_PATH
from utility.raport_helper_functions import load_recipe_data
from model.prepare_data import *


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




dataset = load_recipe_data()

df = initial_dataset_preparation(dataset)
features = [
    'OG', 'FG', 'ABV', 'IBU', 'Color', 'BoilSize', 'BoilTime',
    'Efficiency', 'MashThickness', 'SugarScale', 'BrewMethod'
]

X_train, X_test, y_train, y_test = split_data(df, features)

model = train_model(X_train, y_train)

evaluate_model(model, X_test, y_test)

TEST_RUNS = 20

for _ in range(TEST_RUNS):
    random_sample = X_test.sample(1)
    example_features = random_sample.iloc[0].to_dict()
    example_true_group = y_test.loc[random_sample.index[0]]
    predicted_group = predict(model, features, example_features)

    print(f"============ PRZYKŁAD #{_} ===========")
    print("Przykład cech:", example_features)
    print("Prawdziwa grupa:", example_true_group)
    print("Przewidziana grupa:", predicted_group)
    print("=======================================")