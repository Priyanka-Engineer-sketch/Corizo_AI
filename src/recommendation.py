def recommend_songs(df, song_name, top_n=5):
    cluster = df[df['track_name'] == song_name]['cluster'].values[0]
    recommendations = df[df['cluster'] == cluster]
    return recommendations[['track_name','track_artist']].head(top_n)
