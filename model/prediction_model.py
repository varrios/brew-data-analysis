from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
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

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

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
