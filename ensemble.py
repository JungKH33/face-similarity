import os
from utils import get_average_matrix
from utils import get_scores
from utils import save_matrix
import tensorflow as tf
import numpy as np
import pandas as pd

#input = highest database path
#output = subset data folder path list
#usage: returnpath(full_family_dataset folder)


def model_combination_test(weights, output_dir, base_average_matrix, tuning_average_matrix1, tuning_average_matrix2, tuning_average_matrix3):
    
    average_matrix = (base_average_matrix * weights[0] + 
                      tuning_average_matrix1 * weights[1] + tuning_average_matrix2 * weights[2] + tuning_average_matrix3 * weights[3])
    
    scores = get_scores(average_matrix)
    
    title = model1[0] + model1[1] + str('{:0.3f}'.format(weights[0])) + ' ' + model2[0] + model2[1] + str('{:0.3f}'.format(weights[1])) + ' ' + model3[0] + model3[1] + str('{:0.3f}'.format(weights[2])) + ' ' + model4[0] + model4[1] + str('{:0.3f}'.format(weights[3]))
    
    save_path = os.path.join(output_dir, title)
    save_matrix(average_matrix, title, save_path, scores)
    
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
    
    dataset_dir = r'/your/path/to/dataset'

    output_dir = './outputs/ensemble_rotation'
    
    
    weights = []
    min_value = 0.1
    gap = 0.05
    
    result = pd.DataFrame()
    model1 = ['AdaFace', 'opencv']
    model2 = ['AdaFace', 'retinaface']
    model3 = ['Facenet512', 'fastmtcnn']
    model4 = ['Facenet512', 'opencv']
    
    base_average_matrix = get_average_matrix(dataset_dir, model1[0], model1[1])
    tuning_average_matrix1 = get_average_matrix(dataset_dir, model2[0], model2[1])
    tuning_average_matrix2 = get_average_matrix(dataset_dir, model3[0], model3[1])
    tuning_average_matrix3 = get_average_matrix(dataset_dir, model4[0], model4[1])
    
    # Iterate through i, j, k with the specified gap and constraints
    for i in np.arange(min_value, 1, gap):
        for j in np.arange(min_value, 1 - i, gap):
            for k in np.arange(min_value, 1 - i - j, gap):
                # Calculate the fourth component
                l = 1 - (i + j + k)
                # Check if the fourth component is at least 0.1
                if l >= min_value:
                    weights = [i, j, k, l]
                    new_result = model_combination_test(weights, output_dir, base_average_matrix, tuning_average_matrix1, tuning_average_matrix2, tuning_average_matrix3)
                    result = pd.concat([result, new_result], ignore_index=True)
                    # print(result)
                    print(weights)
    
    result.to_csv('./outputs/result_rotation.csv', index=False)            
    # result.to_csv('./outputs/result_makeup.csv', index=False)            
    # result.to_csv('./outputs/result_hairstyle.csv', index=False)            
    # result.to_csv('./outputs/result_digiface.csv', index=False)            




