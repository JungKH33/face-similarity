import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import image
def calculate_similarity(embeddings: np.ndarray):
    # embeddings should be 2D numpy array with (batch size, embedding size)
    matrix = cosine_similarity(embeddings)

    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True)
    plt.title('Cosine Similarity Matrix')
    plt.xlabel('Images')
    plt.ylabel('Images')
    plt.show()

    # 대각 제외하고 최소 최대 평균 계산
    np.fill_diagonal(matrix, np.nan)

    min_value = np.nanmin(matrix)
    max_value = np.nanmax(matrix)
    min_index = np.unravel_index(np.nanargmin(matrix), matrix.shape)
    max_index = np.unravel_index(np.nanargmax(matrix), matrix.shape)
    average_value = np.nanmean(matrix)

    print("Average value:", average_value)

    print("Maximum value:", max_value)
    print("Index of maximum value:", max_index)

    print("Minimum value:", min_value)
    print("Index of minimum value:", min_index)


def plot_sim_matrix(matrix: np.ndarray):
    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True)
    plt.title('Cosine Similarity Matrix')
    plt.xlabel('Images')
    plt.ylabel('Images')
    plt.show()

    # 대각 제외하고 최소 최대 평균 계산
    np.fill_diagonal(matrix, np.nan)

    min_value = np.nanmin(matrix)
    max_value = np.nanmax(matrix)
    min_index = np.unravel_index(np.nanargmin(matrix), matrix.shape)
    max_index = np.unravel_index(np.nanargmax(matrix), matrix.shape)
    average_value = np.nanmean(matrix)

    return average_value, max_value, max_index, min_value, min_index

def show_images(img_list: list) -> None:
    num_images = len(img_list)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 5))
    for i, (img_name, img) in enumerate(img_list):
        axes[i].imshow(img)
        axes[i].set_title(f"{img_name}\n Index: {i}")
        axes[i].axis('off')  # Hide axis
    plt.tight_layout()
    plt.show()

def inference(model: callable, preprocess: callable, root_path: str):
    # List all files and folders in the current directory
    files_and_folders = os.listdir(root_path)

    # Initialize lists to store image paths and subfolder paths
    image_paths = []
    subfolder_paths = []

    # Iterate over files and folders
    for item in files_and_folders:
        item_path = os.path.join(root_path, item)
        if os.path.isfile(item_path):
            # If the item is a file, check if it's an image and store its path
            if any(item.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                image_paths.append(item_path)
        elif os.path.isdir(item_path):
            # If the item is a folder, recursively call the function
            subfolder_paths.append(item_path)

    if image_paths and len(image_paths) > 1:
        img_list = []
        all_imgs = []

        for index, img_path in enumerate(image_paths):
            img_name = os.path.basename(img_path)
            img = image.load_img(img_path)

            all_imgs.append([img_name, img])

            preprocess_img = preprocess(img)
            img_list.append(preprocess_img)

        input_image = np.array(img_list)
        embeddings = model(input_image)

        print()
        cos_sim_matrix = cosine_similarity(embeddings)
        average_value, max_value, max_index, min_value, min_index = plot_sim_matrix(cos_sim_matrix)

        print("Dataset path: ", root_path)
        print("Average Similarity: ", average_value)
        show_images(all_imgs)

        print("Maximum Similarity: ", max_value)
        show_images([all_imgs[max_index[0]], all_imgs[max_index[1]]])

        print("Minimum Similarity: ", min_value)
        show_images([all_imgs[min_index[0]], all_imgs[min_index[1]]])

        print()

    # Recursively explore subfolders
    for subfolder_path in subfolder_paths:
        inference(model, preprocess, subfolder_path)

# 예시
if __name__ == "__main__":
    embeddings = np.random.rand(10,50)
    calculate_similarity(embeddings)
