import utils
import os
models = [
        # "VGG-Face",
        # "Facenet",
        # "Facenet512",
        # "OpenFace",
        # "DeepFace",
        # "DeepID",
        # "ArcFace",
        # "SFace",
        "AdaFace"
        ]

backends = [
        'yolov8',
        'fastmtcnn',
        'opencv',
        'ssd',
        'mtcnn',
        'retinaface',
        'mediapipe',
        'yunet',
        ]

if __name__ == '__main__':
    
    
    dataset_dir = r'your/path/to/dataset'
    output_dir = r'./outputs/rotation'

    for model in models:
        for backend in backends:
            print("Using model: ", model, " Using backend: ", backend)
            average_matrix = utils.get_average_matrix(dataset_dir, model,backend)
            scores = utils.get_scores(average_matrix)
            title = str(dataset_dir) + ' ' + str(model) + ' ' + str(backend)
            save_path = os.path.join(output_dir,str(model) + '_' + str(backend) + '.png')
            print(save_path)
            utils.save_matrix(average_matrix, title, save_path, scores)
