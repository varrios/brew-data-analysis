from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score, \
    ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from model.prepare_data import *
from sklearn.model_selection import train_test_split


def split_data(df, features, label_column='StyleGroup'):
    X = df[features]
    y = df[label_column]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, plot_cm=True):
    y_pred = model.predict(X_test)
    print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    if not plot_cm:
        return
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot().figure_.savefig('raports\\plots\\confusion_matrix.png')
    plt.show()

def predict(model, features, sample):
    X_sample = pd.DataFrame([sample], columns=features)
    return model.predict(X_sample)[0]


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


def visualize_clusters_2d(df, features):
    if len(df) > 1000:
        df = df.sample(frac=0.1, random_state=42)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(df[features]))

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=df['Cluster'],
        cmap='viridis',
        s=10,
        alpha=0.7
    )
    plt.colorbar(scatter, label='Cluster Number')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Wizualizacja klastrów (PCA 2D)')
    plt.tight_layout()
    plt.savefig('raports\\plots\\pca_clusters_2d.png')
    plt.show()


def plot_cluster_means_separate(df, features):
    cluster_means = df.groupby('Cluster')[features].mean()

    # Plot for Color
    plt.figure(figsize=(10, 5))
    sns.barplot(x=cluster_means.index, y=cluster_means['Color'], palette='viridis')
    plt.title('Średnia wartość Color w klastrach')
    plt.xlabel('Numer klastra')
    plt.ylabel('Średnia wartość Color')
    plt.savefig('raports\\plots\\klaster_color.png')
    plt.show()

    # Plot for IBU
    plt.figure(figsize=(10, 5))
    sns.barplot(x=cluster_means.index, y=cluster_means['IBU'], palette='viridis')
    plt.title('Średnia wartość IBU w klastrach')
    plt.xlabel('Numer klastra')
    plt.ylabel('Średnia wartość IBU')
    plt.savefig('raports\\plots\\klaster_ibu.png')
    plt.show()

    # Plot for ABV
    plt.figure(figsize=(10, 5))
    sns.barplot(x=cluster_means.index, y=cluster_means['ABV'], palette='viridis')
    plt.title('Średnia wartość ABV w klastrach')
    plt.xlabel('Numer klastra')
    plt.ylabel('Średnia wartość ABV')
    plt.savefig('raports\\plots\\klaster_abv.png')
    plt.show()

def plot_feature_importances(model, features):
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
    plt.title('Ważność cech w modelu Random Forest')
    plt.xlabel('Ważność')
    plt.ylabel('Cechy')
    plt.tight_layout()
    plt.savefig('raports\\plots\\feature_importance.png')
    plt.show()
