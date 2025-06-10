
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # Redukcja wymiarów do wizualizacji
from sklearn.metrics import silhouette_score
from utility.raport_helper_functions import load_recipe_data
from model.prediction_model import split_data, train_model, evaluate_model, predict
from model.prepare_data import initial_dataset_preparation, drop_missing_values, fill_missing_values_regression


def perform_clustering(df, features, n_clusters=5):
    """Klasteryzuje dane i dodaje kolumnę z przynależnością do klastra."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    return df

def visualize_clusters(df, features):
    """Wizualizuje klastry w przestrzeni 2D PCA."""
    if len(df) > 1000:
        df = df.sample(frac=0.1, random_state=42)

    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(StandardScaler().fit_transform(df[features]))

    X_pca = df[features].to_numpy()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(
        X_pca[:, 0],  # PC1 (x-axis)
        X_pca[:, 1],  # PC2 (y-axis)
        X_pca[:, 2],  # PC3 (z-axis)
        c=df['Cluster'],  # Color by cluster labels
        s=10,  # Marker size
        marker='o',  # Simpler marker
        alpha=0.7,  # Slightly transparent
        edgecolors='none',
        cmap='viridis'
    )
    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label('Cluster Number')

    plt.title('Wizualizacja klastrów (PCA)')
    plt.show()



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
    df = drop_missing_values(df)

    features_2 = ['Color', 'IBU', 'ABV']
    df = perform_clustering(df, features, n_clusters=6)
    visualize_clusters(df[features_2 + ['Cluster']], features_2 + ['Cluster'])

    cluster_stats = df.groupby('Cluster')[features_2].agg(['mean', 'median', 'std', 'min', 'max'])

    # Print the statistics
    print("\nStatystyki klastrów dla cech:", features_2)
    print("=" * 60)
    for cluster in sorted(df['Cluster'].unique()):
        print(f"\nCluster {cluster}:")
        print("-" * 30)
        for feature in features_2:
            stats = cluster_stats.loc[cluster, feature]
            print(f"{feature}:")
            print(f"  Średnia: {stats['mean']:.2f}")
            print(f"  Mediana: {stats['median']:.2f}")
            print(f"  Odchylenie: {stats['std']:.2f}")
            print(f"  Min: {stats['min']:.2f}")
            print(f"  Max: {stats['max']:.2f}")
        print("=" * 60)

    features_with_cluster = features + ['Cluster']

    X_train, X_test, y_train, y_test = split_data(df, features, 'Cluster')

    #X_train = fill_missing_values_regression(X_train)

    test_data = pd.concat([X_test, y_test], axis=1)
    test_data = drop_missing_values(test_data)
    X_test = test_data[features]
    y_test = test_data['Cluster'] # test_data["StyleGroup"]

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