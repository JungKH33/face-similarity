import numpy as np
import argparse

from deepface import DeepFace

import dataloader
import utils

models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "SFace",
    "AdaFace"
]

# Without dlib
backends = [
    'opencv',
    'ssd',
    'mtcnn',
    'retinaface',
    'mediapipe',
    'yolov8',
    'yunet',
    'fastmtcnn'
]

metrics = ["cosine", "euclidean", "euclidean_l2"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate similarity')
    parser.add_argument('--dataset_path', type=str, help= 'Path to the dataset')
    parser.add_argument('--save_path', type= str, help= 'Path to the save directory')
    parser.add_argument('--model', type=str, choices= models, default='Facenet512',
                        help='Choose a model from available options')
    parser.add_argument('--backend', type=str, choices= backends, default='retinaface',
                        help='Choose a backend from available options')
    args = parser.parse_args()

    dataset_path = args.dataset_path
    save_path = args.save_path
    model = args.model
    backend = args.backend

    data_dict = dataloader.data_dir_loader(data_path= dataset_path)
    embeddings_list = []

    for person, data_paths in data_dict.items():
        print(person)
        results = DeepFace.represent_list(img_path_list = data_paths, model_name= model, detector_backend= backend, enforce_detection = False)
        embeddings = np.array([result['embedding'] for result in results])
        embeddings_list.append(embeddings)


    num_embeddings = len(embeddings_list)
    average_matrix = np.zeros((num_embeddings, num_embeddings))
    for i in range(num_embeddings):
        for j in range(i + 1):
            if i == j:
                sim = utils.calculate_similarity(embeddings_list[i])
                average = utils.calculate_average(sim, mask_diagonal= True)
                average_matrix[i, j] = average

            else:
                sim = utils.calculate_similarity(embeddings_list[i], embeddings_list[j])
                average = utils.calculate_average(sim, mask_diagonal= False)

                average_matrix[i, j] = average
                average_matrix[j, i] = average

    if save_path is None:
        utils.plot_matrix(average_matrix)

    else:
        utils.save_matrix(average_matrix, save_path)




