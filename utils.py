import os
from deepface import DeepFace
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
import deepface.modules.modeling as modeling

#input = highest database path
#output = subset data folder path list
#usage: returnpath(full_family_dataset folder)

def find_deepest_subfolder(directory):
    # List all entries in the given directory
    entries = os.listdir(directory)
    
    # Initialize variables to keep track of the deepest subfolders
    deepest_paths = []
    is_deepest = True
    
    # Iterate over each entry in the directory
    for entry in entries:
        # Construct the full path of the entry
        full_path = os.path.join(directory, entry)
        
        # Check if the entry is a directory
        if os.path.isdir(full_path):
            # If it's a directory, recursively find its deepest subfolder
            is_deepest = False
            deepest_paths.extend(find_deepest_subfolder(full_path))
    
    # If the current directory contains no subdirectories, it is the deepest
    if is_deepest:
        return [directory]
    else:
        return deepest_paths

def get_embeddings(dataset_dir, model, backend):
    for root, dirs, files in os.walk(dataset_dir):
        print("# root : " + root)
        embeddings = []
        for file_name in files:
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file_name)

                try:
                    embedding = DeepFace.represent(img_path= file_path, model_name= model, detector_backend= backend, enforce_detection = False)    
                    print(file_path)              
                    max_confidence = max([elem['face_confidence'] for elem in embedding])
                    for elem in embedding:
                        if elem['face_confidence'] == max_confidence:
                            embeddings.append(elem['embedding'])
                            break
                except Exception as e:
                    print("Face detection error")   
                    print(e)
                    import deepface.modules.modeling as modeling
                    model_client = modeling.build_model(model)
                    embedding = np.zeros((model_client.output_shape))
                    embeddings.append(embedding)
                         
    embeddings = np.array(embeddings)
    return embeddings

def get_average_sim(embedding1, embedding2 = None, metric = 'cosine'):
    
    metrics = {
        'cosine': cosine_similarity,
        'l2': pairwise_distances
    }
    
    
    if embedding2 is None:
        print(embedding1.shape)
        # print("embeddings1 = ", np.where(np.isnan(embedding1)))
        embedding1 = np.nan_to_num(embedding1, nan=0.0)
        # print("embeddings1 = ", np.where(np.isnan(embedding1)))
        cosine_similarity_matrix = metrics[metric](embedding1)

        # mask diagonal elements 
        cosine_similarity_matrix = np.ma.masked_equal(cosine_similarity_matrix, 1.0)
        
    else:
        if np.all(embedding1 == 0):
            embedding2 = embedding1
        
        elif np.all(embedding2 == 0):
            embedding1 = embedding2

        embedding1 = np.nan_to_num(embedding1, nan=0.0)
        embedding2 = np.nan_to_num(embedding2, nan=0.0)
        
        cosine_similarity_matrix = metrics[metric](embedding1, embedding2)
    
    print(cosine_similarity_matrix)
    average_cos_sim = np.mean(cosine_similarity_matrix)
    return average_cos_sim

def get_average_matrix(dataset_dir, model, backend):
    subfolders = find_deepest_subfolder(dataset_dir)
    
    embeddings_list = []
    
    for subfolder in subfolders:
        embeddings = get_embeddings(dataset_dir= subfolder, model= model, backend= backend)
        embeddings_list.append(embeddings)
    
    num_embeddings = len(embeddings_list)
    
    average_matrix = np.zeros((num_embeddings, num_embeddings))
    for i in range(num_embeddings):
        for j in range(i+1):
            if i == j:
                average_matrix[i,j] = get_average_sim(embeddings_list[i])
                
            else:
                average = get_average_sim(embeddings_list[i], embeddings_list[j])
                average_matrix[i,j] = average
                average_matrix[j,i] = average
    
    return average_matrix

def get_scores(matrix):
    diagonal_elements = np.diag(matrix)
    non_diagonal_elements = matrix[~np.eye(matrix.shape[0], dtype=bool)].reshape(matrix.shape[0], -1)
    
    average_diagonal = np.mean(diagonal_elements)
    average_non_diagonal = np.mean(non_diagonal_elements)
    
    return ["{:.02f}".format(average_diagonal), "{:.02f}".format(average_non_diagonal)]

def save_matrix(matrix, title, save_path, score):
    # score[mean_sim, mean_diff]
    # Plotting the heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True, vmin=-1, vmax=1)
    plt.title('Similarity Matrix ' + title)
    plt.xlabel('Datasets')
    plt.ylabel('Datasets')
    
    # Text for the mean similarity and mean difference
    sim_score = 'Scores with similar image: ' + str(score[0])  # Ensure mean_sim is converted to string
    diff_score = 'Scores with different image: ' + str(score[1])  # Ensure mean_diff is converted to string
    
    # Place the text below the heatmap with enough space between them
    plt.figtext(0.5, 0.045, sim_score, ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    plt.figtext(0.5, 0.02, diff_score, ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

    # Ensure the filename is included in save_path
    full_save_path = os.path.join(save_path)
    plt.savefig(full_save_path + '.png', format = 'png')
    plt.close()