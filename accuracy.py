import argparse
from deepface import DeepFace
import dataloader
import matplotlib.pyplot as plt
import utils
from scipy.stats import gaussian_kde

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
    parser.add_argument('--n', type= int, help='Number of positive and negative pairs')
    parser.add_argument('--model', type=str, choices= models, default='Facenet512',
                        help='Choose a model from available options')
    parser.add_argument('--backend', type=str, choices= backends, default='retinaface',
                        help='Choose a backend from available options')
    parser.add_argument('--metric', type=str, choices= metrics, default='cosine',
                        help='Choose a metric from available options')

    args = parser.parse_args()

    dataset_path = args.i
    save_path = args.o
    num_pairs = args.n
    model = args.model
    backend = args.backend
    metric = args.metric

    data_dict = dataloader.data_dir_loader(data_path = dataset_path)
    pairs = dataloader.create_pairs(data_dict, num_pairs)

    resp_obj = DeepFace.verify_list(pairs['file_x'], pairs['file_y'], model_name=model,
                                    detector_backend=backend, distance_metric= metric, enforce_detection=False)

    instances = pairs[["file_x", "file_y"]].values.tolist()
    distances = []

    for i in range(0, len(instances)):
        distance = round(resp_obj[i]["distance"], 4)
        distances.append(distance)

    pairs["distance"] = distances
    print(pairs)

    tp_mean = round(pairs.loc[pairs.type == "same", ['distance']].mean().values[0], 4)
    tp_std = round(pairs.loc[pairs.type == "same", ['distance']].std().values[0], 4)
    fp_mean = round(pairs.loc[pairs.type == "different", ['distance']].mean().values[0], 4)
    fp_std = round(pairs.loc[pairs.type == "different", ['distance']].std().values[0], 4)

    sigma = 1
    threshold = round(tp_mean + sigma * tp_std, 4)
    # threshold = utils.find_intersection(pairs[pairs.type == "same"].distance.plot.kde(label="same"), pairs[pairs.type == "different"].distance.kde(label="different"), 0)

    print("True positive Mean and Standard deviation:", tp_mean, tp_std)
    print("False positive Mean and Standard deviation:", fp_mean, fp_std)
    print('Threshold: ', threshold)

    correct_decisions = ((pairs['distance'] < threshold) == (pairs['type'] == 'same')) | (
                (pairs['distance'] > threshold) == (pairs['type'] == 'different'))
    accuracy = correct_decisions.mean() * 100
    print("Accuracy:", accuracy, "%")

    plt.figure()
    print(pairs[pairs.type == "same"].distance.plot.kde(label="same"))
    pairs[pairs.type == "different"].distance.plot.kde(label="different")

    plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.legend()

    if save_path is not None:
        png_save_path = utils.save_path_gen(save_path, model, backend, 'accuracy', '.png')
        save_save_path = utils.save_path_gen(save_path, model, backend, 'accuracy', '.csv')
        plt.savefig(png_save_path)

    else:
        plt.show()
    plt.close()

