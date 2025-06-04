import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from constants import DATAFILE_PATH, CLEAN_DATAFILE_PATH
from utility.raport_helper_functions import *
from constants import BEER_STYLE_MAP


def initial_dataset_preparation(dataset: pd.DataFrame) -> pd.DataFrame:
    # Drop recipes with unknown styles

    dataset.dropna(subset=['Style'], inplace=True)
    # Drop columns with the majority of N/A values (>50%)
    dataset = dataset.drop(
        columns=[
            'PrimingMethod',
            'PrimingAmount',
            'PitchRate'
        ]
    )

    # Remove outliers with IRQ
    #dataset = _remove_outliers_iqr(dataset)
    dataset = _fill_missing_values(dataset)

    # Group styles of beers according to map
    dataset['StyleGroup'] = dataset['Style'].map(BEER_STYLE_MAP)

    # Change nominal values to numerical
    sugar_scale_map = {'Specific Gravity': 0, 'Plato': 1}
    brew_method_map = {'All Grain': 0, 'extract': 1, 'Partial Mash': 2, 'BIAB': 3}

    dataset['SugarScale'] = dataset['SugarScale'].map(sugar_scale_map)
    dataset['BrewMethod'] = dataset['BrewMethod'].map(brew_method_map)

    # Save to seperate .csv file
    dataset.to_csv(CLEAN_DATAFILE_PATH, index=False)
    return dataset

def _remove_outliers_iqr(dataset: pd.DataFrame) -> pd.DataFrame:
    numerical_cols = dataset.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        dataset = dataset[(dataset[col] >= Q1 - 1.5 * IQR) & (dataset[col] <= Q3 + 1.5 * IQR)]
    return dataset

def _fill_missing_values(dataset: pd.DataFrame) -> pd.DataFrame:
    # Fill columns with <50% values missing with median values
    dataset.fillna({
        'BoilGravity': dataset['BoilGravity'].median(),
        'PrimaryTemp': dataset['PrimaryTemp'].median(),
        'MashThickness': dataset['MashThickness'].median()
    }, inplace=True)
    return dataset




