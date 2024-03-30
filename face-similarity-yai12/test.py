import os
from utils import get_average_matrix
from utils import get_scores
from utils import save_matrix
import tensorflow as tf
import numpy as np
import pandas as pd

def model_combination_test(weights, dataset_dir, output_dir):

    model1 = ['AdaFace', 'opencv']
    model2 = ['AdaFace', 'retinaface']
    model3 = ['Facenet512', 'fastmtcnn']
    model4 = ['Facenet512', 'opencv']
    
    base_average_matrix = get_average_matrix(dataset_dir, model1[0], model1[1])
    tuning_average_matrix1 = get_average_matrix(dataset_dir, model2[0], model2[1])
    tuning_average_matrix2 = get_average_matrix(dataset_dir, model3[0], model3[1])
    tuning_average_matrix3 = get_average_matrix(dataset_dir, model4[0], model4[1])
    
    
    average_matrix = (base_average_matrix * weights[0] + 
                      tuning_average_matrix1 * weights[1] + tuning_average_matrix2 * weights[2] + tuning_average_matrix3 * weights[3])
    
    scores = get_scores(average_matrix)
    scores1 = get_scores(base_average_matrix)
    scores2 = get_scores(tuning_average_matrix1)
    scores3 = get_scores(tuning_average_matrix2)
    scores4 = get_scores(tuning_average_matrix3)
    
    title = model1[0] + model1[1] + str('{:0.3f}'.format(weights[0])) + ' ' + model2[0] + model2[1] + str('{:0.3f}'.format(weights[1])) + ' ' + model3[0] + model3[1] + str('{:0.3f}'.format(weights[2])) + ' ' + model4[0] + model4[1] + str('{:0.3f}'.format(weights[3]))
    
    save_path = os.path.join(output_dir, title)
    save_matrix(average_matrix, title, save_path, scores)
    
    def score_difference(list1, list2):
        # Calculate and format each difference, returning a list of formatted strings
        output = ['{:0.3f}'.format(float(a) - float(b)) for a, b in zip(list1, list2)]
        return output


    save_matrix(matrix = (average_matrix - base_average_matrix), title = title + 'compared_model1', save_path=save_path + 'compared_model1', score = score_difference(scores, scores1))
    save_matrix(matrix = (average_matrix - tuning_average_matrix1), title = title + 'compared_model2', save_path=save_path + 'compared_model2', score = score_difference(scores, scores2))
    save_matrix(matrix = (average_matrix - tuning_average_matrix2), title = title + 'compared_model3', save_path=save_path + 'compared_model3', score = score_difference(scores, scores3))
    save_matrix(matrix = (average_matrix - tuning_average_matrix3), title = title + 'compared_model4', save_path=save_path + 'compared_model4', score = score_difference(scores, scores4))
    
    matrix = pd.DataFrame([[title, scores[0], scores[1]]], columns=['Title', 'sim_score', 'diff_score'])
    return matrix


if __name__ == '__main__':
    # without DLib 
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
    
    dataset_dir = {'different': ['/path/to/your/dataset', './outputs/test/different'], 'rotation': ['/path/to/your/dataset', './outputs/test/rotation'], 'makeup': ['/path/to/your/dataset', './outputs/test/makeup'], 'hairstyle': ['/home/aipark/dataset/hairstyle', './outputs/test/hairstyle'], 'celeb': ['/path/to/your/dataset', './outputs/test/celeb']}
    
    result = pd.DataFrame()
    
    weights = [0.25, 0.15, 0.4, 0.2]
    
    for keys in dataset_dir:
        new_result = model_combination_test(weights, dataset_dir = dataset_dir[keys][0], output_dir = dataset_dir[keys][1])
        
    