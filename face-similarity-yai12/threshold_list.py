import itertools
from deepface import DeepFace
import os
import pandas as pd 
import matplotlib.pyplot as plt
# Load the DeepFace models
models = [
  "AdaFace",
  "VGG-Face",
  "Facenet512",
  "ArcFace",
]

backends = [
  'opencv',
  'mtcnn',
  'retinaface',
  'yunet',
]

metrics = ["cosine", "euclidean", "euclidean_l2"]


# Define the path to your testing dataset
test_data_path = '../../dataset/subjects_4000-5999_72_imgs'
test_data_path = '../../dataset/hairstyle'

# Load the test dataset
test_data = {}
for root, dirs, files in os.walk(test_data_path):
    for file in files:
        person_name = root.split('/')[-1]  # Assuming folder names are the person's names
        if person_name not in test_data:
            test_data[person_name] = []
        test_data[person_name].append(os.path.join(root, file))
        
positives = []
for key, values in test_data.items():
    for i in range(0, len(values)-1):
        for j in range(i+1, len(values)):
            positive = []
            positive.append(values[i])
            positive.append(values[j])
            positives.append(positive)
 
positives = pd.DataFrame(positives, columns = ["file_x", "file_y"])
positives["decision"] = "Yes"

samples_list = list(test_data.values())
 
negatives = []
for i in range(0, len(test_data) - 1):
    for j in range(i+1, len(test_data)):
        cross_product = itertools.product(samples_list[i], samples_list[j])
        cross_product = list(cross_product)
    
        for cross_sample in cross_product:
            negative = []
            negative.append(cross_sample[0])
            negative.append(cross_sample[1])
            negatives.append(negative)
 
negatives = pd.DataFrame(negatives, columns = ["file_x", "file_y"])
negatives["decision"] = "No"

df = pd.concat([positives, negatives]).reset_index(drop = True)

#df.file_x = "/home/aipark/projects/kh/outputs/df/"+df.file_x
#df.file_y = "/home/aipark/projects/kh/outputs/df/"+df.file_y
instances = df[["file_x", "file_y"]].values.tolist()

print(df['file_x'])
print(df['file_y'])

for model in models:
    
    for backend in backends:
        print(model)
        print(backend)
    
        resp_obj = DeepFace.verify_list(df['file_x'], df['file_y'], model_name = model, detector_backend = backend, distance_metric = "cosine", enforce_detection= False)        
        
        print("end of verification")
        distances = []
        for i in range(0, len(instances)):
            distance = round(resp_obj[i]["distance"], 4)
            distances.append(distance)
        print("distances: ", len(distances))
        df["distance"] = distances

        tp_mean = round(df.loc[df.decision == "Yes", ['distance']].mean().values[0], 4)
        tp_std = round(df.loc[df.decision == "Yes", ['distance']].std().values[0], 4)
        fp_mean = round(df.loc[df.decision == "No", ['distance']].mean().values[0], 4)
        fp_std = round(df.loc[df.decision == "No", ['distance']].std().values[0], 4)

        print("tp_mean, tp_std, fpmean, fp_std =",tp_mean, tp_std, fp_mean, fp_std)
        save_path = "./outputs/hairstyle2/df/"+model +"_"+ backend
        plt.figure()
        df[df.decision == "Yes"].distance.plot.kde(label="Yes")
        df[df.decision == "No"].distance.plot.kde(label="No")
        plt.legend()
        plt.savefig(save_path)
        plt.close()
        print("end of savefig")
        sigma = 2
        threshold = round(tp_mean + sigma * tp_std, 4)
        print(threshold)
        txt_path = "./outputs/hairstyle2/text/"+ model + "_" + backend + ".txt"
        f = open(txt_path, "w")
        f.write(model+"_"+ backend + " ")
        f.write(str(threshold))
        f.write("\n")
        f.close()
        print(threshold)


f.close()
      
