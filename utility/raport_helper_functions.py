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

    table_text_quantiles = [
        [f"{q1:.2f}", f"{count_to_q1}"],
        [f"{q2:.2f}", f"{count_to_q2}"],
        [f"{q3:.2f}", f"{count_to_q3}"],
    ]
    row_labels_quantiles = ['Q1 (25%)', 'Q2 (50%)', 'Q3 (75%)']
    col_labels_quantiles = ['Wartość kwantylu', 'Liczba próbek ≤ tej wartości']

    table_text_stats = [
        [f"{mean_value:.2f}"],
        [f"{std_dev:.2f}"],
        [f"{count_min_values}"],
        [f"{count_max_values}"],
        [f"{count_from_q3_to_max}"],
    ]
    row_labels_stats = ['Średnia', 'Odchylenie std.', f'Liczba próbek min. (={min_val})', f'Liczba próbek max. (={max_val})', 'Liczba próbek > Q3']
    col_labels_stats = ['Wartość']

    # Przygotowanie wykresu
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1])

    # Histogram 1 – z outlierami
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_hist.hist(attribute_series, bins=bins, color='skyblue', edgecolor='black')
    ax_hist.set_title(f"Histogram atrybutu: {column_name} (z outlierami)")
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

    fig.tight_layout()
    plt.show()



def generate_histograms_for_dataset(dataset: pd.DataFrame, bins=30) -> None:
    # try:
        for column in dataset.columns:
            if not pd.api.types.is_numeric_dtype(dataset[column]):
                continue
            _generate_histogram(dataset, column, bins)
            return
    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")


def generate_spearman_matrix(dataset: pd.DataFrame) -> None:
    try:
        numerical_df = dataset.select_dtypes(include=['number'])
        correlations = numerical_df.corr(method="spearman")
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlations, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Macierz korelacji Spearmana")
        plt.show()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# def generate_boxplots_for_dataset(df) -> None:
#     try:
#         numerical_df = df.select_dtypes(include=['number'])
#         for column in numerical_df.columns:
#             plt.figure(figsize=(12, 6))
#             sns.violinplot(data=numerical_df[column])
#             plt.title(f"Boxplot dla {column}")
#             plt.show()
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

recipe_data = load_recipe_data()
generate_histograms_for_dataset(recipe_data, bins=30)