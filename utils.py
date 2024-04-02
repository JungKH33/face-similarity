import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
import os
from scipy.optimize import fsolve

def plot_matrix(matrix):
    plt.figure(figsize=(16, 12))
    sns.heatmap(matrix, annot=True, cmap= 'coolwarm', fmt='.2f', cbar=True, square=True, vmin=-1, vmax=1)
    plt.title('Matrix')
    plt.xlabel('Index')
    plt.ylabel('Index')

    plt.show()

def save_matrix(matrix, save_path):
    plt.figure(figsize=(16, 12))
    sns.heatmap(matrix, annot=True, cmap= 'coolwarm', fmt='.2f', cbar=True, square=True, vmin=-1, vmax=1)
    plt.title('Matrix')
    plt.xlabel('People')
    plt.ylabel('People')

    plt.savefig(save_path, format='png')
    plt.close()

def calculate_similarity(embedding1, embedding2 = None, metric = 'cosine'):
    metrics = {
        'cosine': cosine_similarity,
        'l2': pairwise_distances
    }

    if embedding2 is None:
        similarity_matrix = metrics[metric](embedding1)

    else:
        similarity_matrix = metrics[metric](embedding1, embedding2)

    return similarity_matrix

def calculate_average(array, mask_diagonal = False):
    if mask_diagonal:
        mask = ~np.eye(array.shape[0], dtype=bool)
        array = array[mask]
    return np.average(array)

def find_intersection(graph1, graph2, min, max, initial_guess):
    x_values = np.linspace(min, max, 1000)
    def intersection(x):
        return graph1.evaluate(x) - graph2.evaluate(x)

    intersection_point = fsolve(intersection, initial_guess)
    return intersection_point

def save_path_gen(save_dir, model, backend, type, ext = '.png'):
    save_path = os.path.join(save_dir, str(model)+'_'+str(backend) + '_' + str(type) + str(ext))
    return save_path