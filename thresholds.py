import os 

test_data_path = '/path/to/your/dataset'
save_path = '/path/to/your/save_path'

# Load the test dataset
test_data = {}
for root, dirs, files in os.walk(test_data_path):
    for file in files:
        person_name = root.split('/')[-1]  # Assuming folder names are the person's names
        if person_name not in test_data:
            test_data[person_name] = []
        test_data[person_name].append(os.path.join(root, file))

print(test_data)

base_threshold = {"cosine": 0.40, "euclidean": 0.55, "euclidean_l2": 0.75}

thresholds = {
    # "VGG-Face": {"cosine": 0.40, "euclidean": 0.60, "euclidean_l2": 0.86}, # 2622d
    "VGG-Face": {
        "cosine": 0.68,
        "euclidean": 1.17,
        "euclidean_l2": 1.17,
    },  # 4096d - tuned with LFW
    "Facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
    "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
    "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
    "Dlib": {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.4},
    "SFace": {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055},
    "OpenFace": {"cosine": 0.10, "euclidean": 0.55, "euclidean_l2": 0.55},
    "DeepFace": {"cosine": 0.23, "euclidean": 64, "euclidean_l2": 0.64},
    "DeepID": {"cosine": 0.015, "euclidean": 45, "euclidean_l2": 0.17},
}