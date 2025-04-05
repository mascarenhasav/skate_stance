from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from cycler import cycler

plt.style.use("bmh")
# Set default color for bars globally
plt.rcParams['axes.prop_cycle'] = cycler(color=['tab:brown'])
# Set global defaults for bar borders
plt.rcParams['patch.edgecolor'] = 'black'  # Border color
plt.rcParams['patch.linewidth'] = 1.5     # Border width

path_clustering = "../images/clusters.png"
path_clustering_number = "../images/clusters_number.png"
path_clustering_stance = "../images/clusters_stance.png"

def clustering(df):
    # remove some columns
    df_cleaned = df.drop(columns=["stance_binary", "skate_stance"])

    # standardize the data to clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_cleaned)

    # Elbow Method to determine the ideal clustering number
    print(f"[CLUSTERING] elbow method to determmine the number of clusters")
    inertia = []
    k_range = range(1, 10)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    # ploting the elbow results
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o')
    plt.title('Elbow method to determine the number of clusters')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(k_range)
    plt.savefig(path_clustering_number)
    plt.show()
    
    # applying the clustering -----------------------------------
    print(f"[CLUSTERING] applying KMeans with 3 clusters")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cleaned['Cluster'] = kmeans.fit_predict(scaled_data)

    # visualize the clustering ----------------------------------
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    df_cleaned['PCA1'] = pca_result[:, 0]
    df_cleaned['PCA2'] = pca_result[:, 1]

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x='PCA1',
        y='PCA2',
        hue='Cluster',
        data=df_cleaned,
        palette='viridis',
        s=100,
        alpha=0.8,
        edgecolor='black'  # Contorno preto
    )

    plt.title('Clusters of the lateralizations with PCA')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(title='Cluster')
    plt.savefig(path_clustering)
    plt.show()


    # Describing the clusters
    cluster_summary = df_cleaned.groupby('Cluster').mean()
    print(cluster_summary)
    
    # putting skate stance together
    df_clusters = df[['skate_stance']].copy()
    df_clusters['Cluster'] = df_cleaned['Cluster'].values  # associating with the clusters

    # calculating the distribution in each cluster
    stance_distribution = df_clusters.groupby('Cluster')['skate_stance'].value_counts(normalize=True).unstack().fillna(0)

    # showing as table
    stance_distribution_percentage = stance_distribution * 100
    stance_distribution_percentage.round(1)
    
    # Assuming stance_distribution_percentage is a DataFrame
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Distribution of Skate Stance within Clusters')
    ax.bar(
        stance_distribution_percentage.index,
        stance_distribution_percentage.iloc[:, 0],
        label=stance_distribution_percentage.columns[0],
        color='skyblue',
        edgecolor='black'
    )

    ax.bar(
        stance_distribution_percentage.index,
        stance_distribution_percentage.iloc[:, 1],
        bottom=stance_distribution_percentage.iloc[:, 0],
        label=stance_distribution_percentage.columns[1],
        color='gold',
        edgecolor='black'
    )

    ax.set_ylabel('Percentage')
    ax.set_xlabel('Cluster')
    ax.legend(
        title='Skate Stance', 
        labels=['Goofy', 'Regular'],
        fontsize=14,           # Tamanho da fonte dos itens da legenda
        title_fontsize=16      # Tamanho da fonte do título da legenda
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path_clustering_stance)
    plt.show()

