import numpy as np
from IPython.core.pylabtools import figsize
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator

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



def _generate_histogram(dataset: pd.DataFrame, column_name: str, bins=30) -> None:
    if column_name not in dataset.columns:
        print(f"Column '{column_name}' does not exist in DataFrame.")
        return

    attribute_series = dataset[column_name].dropna()

    std_dev = attribute_series.std()
    mean_value = attribute_series.mean()
    q1 = attribute_series.quantile(0.25)
    q2 = attribute_series.quantile(0.50)
    q3 = attribute_series.quantile(0.75)
    min_val = attribute_series.min()
    max_val = attribute_series.max()

    count_min_values = attribute_series[attribute_series == min_val].count()
    count_max_values = attribute_series[attribute_series == max_val].count()
    count_to_q1 = attribute_series[attribute_series <= q1].count()
    count_to_q2 = attribute_series[attribute_series <= q2].count()
    count_to_q3 = attribute_series[attribute_series <= q3].count()
    count_from_q3_to_max = attribute_series[attribute_series > q3].count()

    # print(f"\nAnaliza atrybutu: {column_name}")
    # print(f"Średnia: {mean_value:.2f}")
    # print(f"Odchylenie standardowe: {std_dev:.2f}")
    # print(f"Min: {min_val} (Liczba próbek: {count_min_values})")
    # print(f"Max: {max_val} (Liczba próbek: {count_max_values})")
    # print(f"Q1 (25%): {q1:.2f} (Liczba próbek: {count_to_q1})")
    # print(f"Q2 (50%): {q2:.2f} (Liczba próbek: {count_to_q2})")
    # print(f"Q3 (75%): {q3:.2f} (Liczba próbek: {count_to_q3})")
    # print(f"Liczba próbek powyżej Q3 (75%): {count_from_q3_to_max}")
    # print(f"Rozstęp międzykwartylowy (IQR): {q3 - q1:.2f}")
    # print(f"Zakres wartości: {min_val} - {max_val}")
    # print(f"Zakres wartości bez outlierów: {q1 - 1.5 * (q3 - q1):.2f} - {q3 + 1.5 * (q3 - q1):.2f}")
    # print(f"Liczba próbek: {len(attribute_series)}")
    # print(f"Liczba próbek bez outlierów: {len(attribute_series[(attribute_series >= q1 - 1.5 * (q3 - q1)) & (attribute_series <= q3 + 1.5 * (q3 - q1))])}")

    table_text_quantiles = [
        [f"{q1:.2f}", f"{count_to_q1}"],
        [f"{q2:.2f}", f"{count_to_q2}"],
        [f"{q3:.2f}", f"{count_to_q3}"],
        [f"-" ,f"{count_from_q3_to_max}"],
    ]
    row_labels_quantiles = ['Q1 (25%)', 'Q2 (50%)', 'Q3 (75%)', '> Q3 (75%)']
    col_labels_quantiles = ['Wartość kwantylu', 'Liczba próbek']

    table_text_stats = [
        [f"{mean_value:.2f}"],
        [f"{std_dev:.2f}"],
        [f"{count_min_values}"],
        [f"{count_max_values}"],
    ]
    row_labels_stats = ['Średnia', 'Odchylenie std.', f'Liczba próbek min. (={min_val})', f'Liczba próbek max. (={max_val})']
    col_labels_stats = ['Wartość']

    # Przygotowanie wykresu
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1])

    # Histogram 1 – z outlierami
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_hist.hist(attribute_series, bins=bins, color='skyblue', edgecolor='black')
    ax_hist.set_title(f"Histogram atrybutu: {column_name}")
    ax_hist.set_xlabel(column_name)
    ax_hist.set_ylabel("Liczba próbek")

    # Histogram 2 – bez outlierów
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_series = attribute_series[(attribute_series >= lower_bound) & (attribute_series <= upper_bound)]

    ax_hist_filtered = fig.add_subplot(gs[0, 1])
    ax_hist_filtered.hist(filtered_series, bins=bins, color='lightgreen', edgecolor='black')
    ax_hist_filtered.set_title(f"Histogram atrybutu: {column_name} (bez outlierów)")
    ax_hist_filtered.set_xlabel(column_name)
    ax_hist_filtered.set_ylabel("Liczba próbek")

    # Tabela 1 – kwantyle
    ax_table1 = fig.add_subplot(gs[1, 0])
    ax_table1.axis('off')
    ax_table1.set_title("Tabela 1: Kwantyle")
    table1 = ax_table1.table(
        cellText=table_text_quantiles,
        rowLabels=row_labels_quantiles,
        colLabels=col_labels_quantiles,
        cellLoc='center',
        loc='center'
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(9.5)
    table1.scale(1, 1.5)

    # Tabela 2 – statystyki
    ax_table2 = fig.add_subplot(gs[1, 1])
    ax_table2.axis('off')
    ax_table2.set_title("Tabela 2: Statystyki")
    table2 = ax_table2.table(
        cellText=table_text_stats,
        rowLabels=row_labels_stats,
        colLabels=col_labels_stats,
        cellLoc='center',
        loc='center'
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(9.5)
    table2.scale(1, 1.5)

    fig.suptitle(f"Analiza rozkładu dla atrybutu: {column_name}", fontsize=16, y=0.98)
    fig.tight_layout()
    plt.show()
    print("\n")



def generate_histograms_for_dataset(dataset: pd.DataFrame, bins=30) -> None:
    try:
        for column in dataset.columns:
            if not pd.api.types.is_numeric_dtype(dataset[column]):
                continue
            _generate_histogram(dataset, column, bins)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def generate_spearman_matrix(dataset: pd.DataFrame, top_n: int = 10) -> None:
    try:
        numerical_df = dataset.select_dtypes(include=['number'])
        correlations = numerical_df.corr(method="spearman")

        plt.figure(figsize=(12, 10))
        sns.heatmap(correlations, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Macierz korelacji Spearmana")
        plt.show()


        # Wypisanie wyników korelacji
        # corr_pairs = (
        #     correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))
        #     .stack()
        #     .reset_index()
        # )
        # corr_pairs.columns = ["Atrybut 1", "Atrybut 2", "Korelacja"]
        #
        #
        # top_corr = corr_pairs.reindex(corr_pairs['Korelacja'].abs().sort_values(ascending=False).index)
        #
        # print("\nNajbardziej skorelowane pary atrybutów:")
        # for _, row in top_corr.head(top_n).iterrows():
        #     print(f"{row['Atrybut 1']} ↔ {row['Atrybut 2']} : korelacja = {row['Korelacja']:.2f}")
        #
        #
        # mean_abs_corr = correlations.abs().mean().sort_values(ascending=False)
        # top_attr = mean_abs_corr.index[0]
        # print(f"\nNajbardziej globalnie skorelowany atrybut: {top_attr} (średnia |korelacja| = {mean_abs_corr.iloc[0]:.2f})")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")



# recipe_data = load_recipe_data()
# generate_histograms_for_dataset(recipe_data, bins=30)
# #generate_spearman_matrix(recipe_data)