from model.prediction_model import *


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
        'Size(L)',
        'Efficiency',
        'SugarScale',
        'BrewMethod',
        'BoilGravity',
        'MashThickness',
        'PrimaryTemp',
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

    X_train, X_test, y_train, y_test = split_data(df, features, 'Cluster')

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()