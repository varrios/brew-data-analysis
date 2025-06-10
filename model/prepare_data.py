from utility.raport_helper_functions import *
from constants import BEER_STYLE_MAP


def group_by_style(dataset: pd.DataFrame) -> pd.DataFrame:
    # Drop recipes with unknown styles
    dataset.dropna(subset=['Style'], inplace=True)

    # Group styles of beers according to map
    dataset['StyleGroup'] = dataset['Style'].map(BEER_STYLE_MAP)

    return dataset


def initial_dataset_preparation(dataset: pd.DataFrame) -> pd.DataFrame:

    #dataset = group_by_style(dataset)

    # Remove outliers with IRQ
    #dataset = _remove_outliers_iqr(dataset)

    # Change nominal values to numerical
    sugar_scale_map = {'Specific Gravity': 0, 'Plato': 1}
    brew_method_map = {'All Grain': 0, 'extract': 1, 'Partial Mash': 2, 'BIAB': 3}

    dataset['SugarScale'] = dataset['SugarScale'].map(sugar_scale_map)
    dataset['BrewMethod'] = dataset['BrewMethod'].map(brew_method_map)

    return dataset

def _remove_outliers_iqr(dataset: pd.DataFrame) -> pd.DataFrame:
    numerical_cols = dataset.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        dataset = dataset[(dataset[col] >= Q1 - 1.5 * IQR) & (dataset[col] <= Q3 + 1.5 * IQR)]
    return dataset


def drop_missing_values(dataset: pd.DataFrame) -> pd.DataFrame:
    missing_rows = dataset[['BoilGravity', 'PrimaryTemp', 'MashThickness']].isna().any(axis=1).sum()
    print(f"Liczba wierszy z brakującymi wartościami: {missing_rows} / {len(dataset)}")

    dataset.dropna(subset=['BoilGravity', 'PrimaryTemp', 'MashThickness'], inplace=True)
    return dataset

