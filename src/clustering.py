from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def elbow_method(data):
    inertia = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(data)
        inertia.append(km.inertia_)

    plt.plot(range(2, 11), inertia)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.show()


def apply_kmeans(data, k=5):
    kmeans = KMeans(n_clusters=k, random_state=42)
    return kmeans.fit_predict(data)


def pca_visualization(data, labels):
    pca = PCA(n_components=2)
    components = pca.fit_transform(data)

    plt.figure(figsize=(8, 6))
    plt.scatter(components[:, 0], components[:, 1], c=labels, alpha=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Spotify Song Clusters (PCA)")
    plt.show()


def playlist_cluster_analysis(df):
    """
    Analyzes how playlist genres are distributed across clusters
    """
    print(
        df.groupby('cluster')['playlist_genre']
        .value_counts()
        .head()
    )
