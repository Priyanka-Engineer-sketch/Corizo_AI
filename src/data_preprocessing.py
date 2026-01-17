import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Selecting numerical audio features
    features = [
        'danceability','energy','key','loudness','mode',
        'speechiness','acousticness','instrumentalness',
        'liveness','valence','tempo'
    ]

    df_features = df[features].dropna()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_features)

    return df, df_features, scaled_features
