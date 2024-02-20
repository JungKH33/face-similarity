from retinaface import RetinaFace
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

#test_img_path = r"C:\projects\utils\temp\examples\KakaoTalk_20240216_193742984_11.jpg"
#test_img2_path = r"C:\projects\utils\temp\examples\KakaoTalk_20240217_151849589_01.jpg"
# test_img3_path = r'C:\projects\utils\temp\examples\54efca99-0f55-486c-ae87-666268230f21.jpg'
#resp = RetinaFace.detect_faces(test_img_path)
#(resp)

#faces = RetinaFace.extract_faces(img_path = test_img2_path, align = True)
#for face in faces:
#  plt.imshow(face)
#  plt.show()

#obj = DeepFace.verify(test_img_path, test_img2_path
#                    , model_name='ArcFace', detector_backend='retinaface')
#print(obj)

#embeddings = DeepFace.represent(img_path= test_img3_path, model_name= 'ArcFace', detector_backend= 'retinaface')
#print(len(embeddings))

def get_embeddings(dataset_path):
    pass

import os
dataset_dir = r"C:\projects\utils\temp\hairstyle"
#dataset_dir = r"C:\projects\utils\temp\rotation"
output_dir = r"C:\projects\utils\temp\output"
#dataset_dir = r"C:\projects\utils\temp\makeup_dataset"

models = [
  "VGG-Face",
  "Facenet",
  "Facenet512",
  "OpenFace",
  "DeepFace",
  "DeepID",
  "ArcFace",
  "Dlib",
  "SFace",
]

backends = [
  'opencv',
  'ssd',
  'dlib',
  'mtcnn',
  'retinaface',
  'mediapipe',
  'yolov8',
  'yunet',
  'fastmtcnn',
]

metrics = ["cosine", "euclidean", "euclidean_l2"]

model = models[4]
backend = backends[5]
metric = metrics[0]
print("Using model: ", model)
print("Using backend: ", backend)
print("Using metric: ", metric)

for root, dirs, files in os.walk(dataset_dir):
    print("# root : " + root)
    if len(files) > 1:
        embeddings = []
        for file_name in files:
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                print("file: " + file_name)
                file_path = os.path.join(root, file_name)

                # faces = RetinaFace.extract_faces(img_path= file_path, align=True)
                embedding = DeepFace.represent(img_path= file_path, model_name= model, detector_backend= backend, enforce_detection= False)
                for i in range (len(embedding)):
                    embeddings.append(embedding[i]['embedding'])

        if len(embeddings) > 1:
            embeddings = np.array(embeddings)
            cosine_similarity_matrix = cosine_similarity(embeddings)

            print(cosine_similarity_matrix)
            plt.figure(figsize=(8, 6))
            sns.set(font_scale=1.2)
            sns.heatmap(cosine_similarity_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True)
            plt.title('Cosine Similarity Matrix')
            plt.xlabel('Images')
            plt.ylabel('Images')
            plt.savefig(os.path.join(output_dir, file_name))