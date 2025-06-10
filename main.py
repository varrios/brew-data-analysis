import pandas as pd
from utility.raport_helper_functions import load_recipe_data
from model.prediction_model import split_data, train_model, evaluate_model, predict
from model.prepare_data import initial_dataset_preparation, drop_missing_values, fill_missing_values_regression

def main():
    dataset = load_recipe_data()

    features = [
        'OG',
        'FG',
        'ABV',
        'IBU',
        'Color',
        'BoilSize',
        'BoilTime',
        'PrimaryTemp',
        'Size(L)',
        'Efficiency',
        'MashThickness',
        'SugarScale',
        'BrewMethod',
        'BoilGravity',
    ]

    df = initial_dataset_preparation(dataset)

    X_train, X_test, y_train, y_test = split_data(dataset, features)

    X_train = fill_missing_values_regression(X_train)

    test_data = pd.concat([X_test, y_test], axis=1)
    test_data = drop_missing_values(test_data)
    X_test = test_data[features]
    y_test = test_data["StyleGroup"]

    model = train_model(X_train, y_train)
    # model = optimize_model(X_train, y_train)

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

if __name__ == "__main__":
    main()