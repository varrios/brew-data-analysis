from constants import DATAFILE_PATH, EXCLUDED_COLUMNS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_recipe_data(filepath=DATAFILE_PATH, excluded_cols=EXCLUDED_COLUMNS) -> pd.DataFrame | None :
    try:
        df = pd.read_csv(filepath, encoding="ISO-8859-1")
        if excluded_cols:
            df = df.drop(columns=excluded_cols, errors='ignore')
        return df
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def generate_histogram(df, column_name, bins=30) -> None:
    try:
        if column_name not in df.columns:
            print(f"Column '{column_name}' does not exist in the DataFrame.")
            return

        plt.figure(figsize=(10, 6))
        df[column_name].plot.hist(bins=bins)
        plt.title(f"Histogram atrybutu {column_name}")
        plt.xlabel(column_name)
        plt.ylabel("Liczba próbek")
        # plt.yscale("log")
        plt.show()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def generate_histograms_for_dataset(df, bins=30) -> None:
    try:
        for column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                continue
            generate_histogram(df, column, bins)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def generate_spearman_matrix(df) -> None:
    try:
        numerical_df = df.select_dtypes(include=['number'])
        corr = numerical_df.corr(method="spearman")
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Macierz korelacji Spearmana")
        plt.show()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
