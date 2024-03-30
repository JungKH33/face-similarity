import random
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from deepface import DeepFace

models = [
  "VGG-Face",
  "Facenet",
  "Facenet512",
  "OpenFace",
  "DeepFace",
  "DeepID",
  "ArcFace",
  "SFace",
]

backends = [
  'opencv',
  'ssd',
  'mtcnn',
  'retinaface',
  'mediapipe',
  'yolov8',
  'yunet'
]

# Define the path to your testing dataset
test_data_path = '/path/to/your/dataset'

# Load the test dataset
test_data = {}
for root, dirs, files in os.walk(test_data_path):
    for file in files:
        person_name = root.split('/')[-1]  # Assuming folder names are the person's names
        if person_name not in test_data:
            test_data[person_name] = []
        test_data[person_name].append(os.path.join(root, file))

def generate_pairs(data):
    genuine_pairs = []
    impostor_pairs = []
    for person, images in data.items():
        for i in range(len(images)):
            for j in range(i+1, len(images)):
                genuine_pairs.append((images[i], images[j], 1))  # genuine pair, label 1
                different_person = random.choice(list(data.keys()))
                while different_person == person:
                    different_person = random.choice(list(data.keys()))
                different_image = random.choice(data[different_person])
                impostor_pairs.append((images[i], different_image, 0))  # impostor pair, label 0
    return genuine_pairs, impostor_pairs

def calculate_roc(model, backend):
    genuine_pairs, impostor_pairs = generate_pairs(test_data)
    all_scores = []
    all_labels = []

    for pair in genuine_pairs + impostor_pairs:
        img1, img2, label = pair
        try:
            result = DeepFace.verify(img1, img2, model_name=model, detector_backend=backend, enforce_detection=False)
            score = result['distance']
            all_scores.append(score)
            all_labels.append(label)
        except:
            pass

    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

# Plot ROC curves for each model and backend
plt.figure(figsize=(10, 6))
for model_name in models:
    for backend_name in backends:
        fpr, tpr, roc_auc = calculate_roc(model_name, backend_name)
        plt.plot(fpr, tpr, label=f'{model_name} - {backend_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
