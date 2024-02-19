import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

def plot_tsne(embeddings: np.ndarray, labels: np.ndarray):
    # Reduce dimensionality using t-SNE
    tsne = TSNE(n_components=2)
    embedding_2d = tsne.fit_transform(embeddings)

    # Visualize the clusters
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='viridis')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title("t-SNE Visualization")
    plt.colorbar()
    plt.show()

def plot_sim_matrix(matrix: np.ndarray):
    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True)
    plt.title('Cosine Similarity Matrix')
    plt.xlabel('Images')
    plt.ylabel('Images')
    plt.show()

def clustering(embeddings: np.ndarray, metric = 'cosine'):
    # Perform DBSCAN clustering
    dbscan = DBSCAN(metric= metric, eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(embeddings)
    return labels





