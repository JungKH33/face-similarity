import random
import itertools
from deepface import DeepFace
import os
import numpy as np
import pandas as pd 

# Load the DeepFace models
models = [
  "VGG-Face",
  "Facenet512",
  "ArcFace",
  "SFace",
]

backends = [
  'opencv',
  'mtcnn',
  'retinaface',
  'yunet',
]

metrics = ["cosine", "euclidean", "euclidean_l2"]


# Define the path to your testing dataset
test_data_path = '/home/aipark/dataset/different'


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


# sample df 
df_yes = df[df['decision'] == 'Yes']
df_no = df[df['decision'] == 'No']

# Sample an equal number of rows from each subset
sample_size = min(len(df_yes), len(df_no))
sampled_df = pd.concat([df_yes.sample(n=sample_size, random_state=42), df_no.sample(n=sample_size, random_state=42)])

# Shuffle the sampled DataFrame to randomize the order
sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(sampled_df)

model = 'AdaFace'
model = "Facenet512"

backend = 'retinaface'

resp_obj = DeepFace.verify_list(sampled_df['file_x'], sampled_df['file_y'], model_name = model, detector_backend = backend, distance_metric = "cosine", enforce_detection= False)


distances = []
for i in range(0, len(instances)):
 distance = round(resp_obj[i]["distance"], 4)
 distances.append(distance)
 
sampled_df["distance"] = distances

tp_mean = round(sampled_df.loc[sampled_df.decision == "Yes", ['distance']].mean().values[0], 4)
tp_std = round(sampled_df.loc[sampled_df.decision == "Yes", ['distance']].std().values[0], 4)
fp_mean = round(sampled_df.loc[sampled_df.decision == "No", ['distance']].mean().values[0], 4)
fp_std = round(sampled_df.loc[sampled_df.decision == "No", ['distance']].std().values[0], 4)

sampled_df[sampled_df.decision == "Yes"].distance.plot.kde()
sampled_df[sampled_df.decision == "No"].distance.plot.kde()

sigma = 2
threshold = round(tp_mean + sigma * tp_std, 4)

print("True positive Mean and Standard deviation:",tp_mean, tp_std)
print("False positive Mean and Standard deviation:",fp_mean, fp_std)
print(threshold)

filtered_df = sampled_df[sampled_df["distance"] > 1]
print(filtered_df)


df1 = pd.read_csv('/home/aipark/projects/kh/outputs/digiface/df/AdaFace_opencv.csv')
df2 = pd.read_csv('/home/aipark/projects/kh/outputs/digiface/df/AdaFace_retinaface.csv')
df3 = pd.read_csv('/home/aipark/projects/kh/outputs/digiface/df/Facenet512_fastmtcnn.csv')
df4 = pd.read_csv('/home/aipark/projects/kh/outputs/digiface/df/VGG-Face_mtcnn.csv')


'''weighted_distance = (df1['distance']) 
weighted_df = pd.DataFrame({'distance': weighted_distance, 'decision': df1['decision']})'''

'''weighted_distance = (df2['distance'])
weighted_df = pd.DataFrame({'distance': weighted_distance, 'decision': df2['decision']})'''

'''weighted_distance = (df3['distance'])
weighted_df = pd.DataFrame({'distance': weighted_distance, 'decision': df3['decision']})'''

weighted_distance = (df1['distance'])
weighted_df = pd.DataFrame({'distance': weighted_distance, 'decision': df1['decision']})
'''weighted_distance = (df1['distance'] * 0.25) + (df2['distance'] * 0.15) + (df3['distance'] * 0.4) + (df4['distance'] * 0.2)
weighted_df = pd.DataFrame({'distance': weighted_distance, 'decision': df1['decision']})'''

weighted_df[weighted_df.decision == "Yes"].distance.plot.kde()
weighted_df[weighted_df.decision == "No"].distance.plot.kde()

tp_mean = round(weighted_df.loc[weighted_df.decision == "Yes", ['distance']].mean().values[0], 4)
tp_std = round(weighted_df.loc[weighted_df.decision == "Yes", ['distance']].std().values[0], 4)
fp_mean = round(weighted_df.loc[weighted_df.decision == "No", ['distance']].mean().values[0], 4)
fp_std = round(weighted_df.loc[weighted_df.decision == "No", ['distance']].std().values[0], 4)

sigma = 2
threshold = round(tp_mean + sigma * tp_std, 4)

print("True positive Mean and Standard deviation:",tp_mean, tp_std)
print("False positive Mean and Standard deviation:",fp_mean, fp_std)
print('Threshold: ', threshold)

correct_decisions = ((weighted_df['distance'] < threshold) == (weighted_df['decision'] == 'Yes')) | ((weighted_df['distance'] > threshold) == (weighted_df['decision'] == 'No'))
accuracy = correct_decisions.mean() * 100
print("Accuracy:", accuracy, "%")

print()
# Calculate true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN)
TP = ((weighted_df['distance'] < threshold) & (weighted_df['decision'] == 'Yes')).sum()
FP = ((weighted_df['distance'] < threshold) & (weighted_df['decision'] == 'No')).sum()
TN = ((weighted_df['distance'] >= threshold) & (weighted_df['decision'] == 'No')).sum()
FN = ((weighted_df['distance'] >= threshold) & (weighted_df['decision'] == 'Yes')).sum()

# Calculate true positive rate (TPR) and false positive rate (FPR)
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)

# Print the results
print("True Positives (TP):", TP)
print("False Positives (FP):", FP)
print("True Negatives (TN):", TN)
print("False Negatives (FN):", FN)
print("True Positive Rate (TPR):", TPR)
print("False Positive Rate (FPR):", FPR)
