import argparse

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
    parser.add_argument('--i', type=str, help= 'Path to the dataset')
    parser.add_argument('--o', type= str, help= 'Path to the save directory')
    parser.add_argument('--model', type=str, choices= models, default='Facenet512',
                        help='Choose a model from available options')
    parser.add_argument('--backend', type=str, choices= backends, default='retinaface',
                        help='Choose a backend from available options')
    args = parser.parse_args()