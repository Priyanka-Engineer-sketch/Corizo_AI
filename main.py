from src.data_preprocessing import load_and_preprocess
from src.eda_visualization import plot_distributions, correlation_matrix
from src.clustering import (
    elbow_method,
    apply_kmeans,
    pca_visualization,
    playlist_cluster_analysis
)
from src.recommendation import recommend_songs


# Load and preprocess data
df, df_features, scaled = load_and_preprocess("data/spotify_data.csv")


# Exploratory Data Analysis
plot_distributions(df_features)
correlation_matrix(df_features)


# Clustering
elbow_method(scaled)
labels = apply_kmeans(scaled)


# Add cluster labels to dataframe
df['cluster'] = labels


# PCA Visualization
pca_visualization(scaled, labels)


# Playlist genre vs cluster analysis
playlist_cluster_analysis(df)


# Recommendation system demo
print(
    recommend_songs(
        df,
        df['track_name'].iloc[0]
    )
)
