import os
import itertools
import pandas as pd

def data_dir_loader(data_path):
    data_dict = {}
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                person_name = os.path.basename(root)
                if person_name not in data_dict:
                    data_dict[person_name] = []
                data_dict[person_name].append(os.path.join(root, file))
    return data_dict

def create_pairs(data_dict, num_pairs = None):
    positives = []
    for key, values in data_dict.items():
        for i in range(0, len(values) - 1):
            for j in range(i + 1, len(values)):
                positive = []
                positive.append(values[i])
                positive.append(values[j])
                positives.append(positive)

    positives = pd.DataFrame(positives, columns=["file_x", "file_y"])
    positives["type"] = "same"

    samples_list = list(data_dict.values())

    negatives = []
    for i in range(0, len(data_dict) - 1):
        for j in range(i + 1, len(data_dict)):
            cross_product = itertools.product(samples_list[i], samples_list[j])
            cross_product = list(cross_product)

            for cross_sample in cross_product:
                negative = []
                negative.append(cross_sample[0])
                negative.append(cross_sample[1])
                negatives.append(negative)

    negatives = pd.DataFrame(negatives, columns=["file_x", "file_y"])
    negatives["type"] = "different"

    if num_pairs is None:
        pairs = pd.concat([positives, negatives]).reset_index(drop=True)

    else:
        pairs = pd.concat([positives.sample(n= num_pairs, random_state=42), negatives.sample(n= num_pairs, random_state=42)]).reset_index(drop=True)
    return pairs



if __name__ == '__main__':
    data_dict = data_dir_loader('./datasets')
    pairs = create_pairs(data_dict, num_pairs= 10)
    print(pairs)