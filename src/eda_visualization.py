import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions(df):
    df.hist(figsize=(15,10))
    plt.suptitle("Distribution of Audio Features")
    plt.show()

def correlation_matrix(df):
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), cmap='coolwarm')
    plt.title("Correlation Matrix of Spotify Audio Features")
    plt.show()
