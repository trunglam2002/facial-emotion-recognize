import os
import pickle
import cv2
import numpy as np
from Delaunay import build_graph_with_master_node, extract_facial_landmarks, draw_graph_on_image
from transform import apply_gaussian_noise, apply_random_rotation, apply_random_flip


def augment_and_save(image, original_landmarks, num_augmentations=3):
    image = np.array(image)
    augmented_images_list = []
    augmented_landmarks_list = []

    # Ảnh gốc
    augmented_images_list.append(image)
    augmented_landmarks_list.append(original_landmarks)

    # Random noise ảnh gốc 3 lần
    for _ in range(num_augmentations):
        augmented_image_noise, augmented_landmarks_noise = apply_gaussian_noise(
            image, original_landmarks)
        augmented_images_list.append(augmented_image_noise)
        augmented_landmarks_list.append(augmented_landmarks_noise)

    # Random rotation ảnh gốc 3 lần
    for _ in range(num_augmentations):
        augmented_image_rotated, augmented_landmarks_rotated = apply_random_rotation(
            image, original_landmarks)
        augmented_images_list.append(augmented_image_rotated)
        augmented_landmarks_list.append(augmented_landmarks_rotated)

    # Flipping tất cả ảnh đã tạo
    original_length = len(augmented_images_list)
    for idx in range(original_length):
        augmented_image_flipped, augmented_landmarks_flipped = apply_random_flip(
            augmented_images_list[idx], augmented_landmarks_list[idx], image.shape[1])
        augmented_images_list.append(augmented_image_flipped)
        augmented_landmarks_list.append(augmented_landmarks_flipped)

    # Các bước augmenting khác không thay đổi
    return augmented_images_list, augmented_landmarks_list


def load_data(data_dir, augment=True, num_augmentations=3):
    data = {'train': {'images': [], 'labels': []},
            'valid': {'images': [], 'labels': []},
            'test': {'images': [], 'labels': []}}

    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(data_dir, split)
        image_folder = os.path.join(split_path, 'images')

        images, labels = [], []

        for idx, file_name in enumerate(os.listdir(image_folder)):
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_folder, file_name)

                label_file_path = os.path.join(
                    split_path, 'labels', file_name.replace('jpg', 'txt'))
                with open(label_file_path, 'r') as label_file:
                    label_str = label_file.readline().strip().split()[0]
                    label = int(label_str)

                facial_landmarks = np.array(
                    extract_facial_landmarks(image_path))

                if augment:
                    augumented_images_list, augmented_landmarks_list = augment_and_save(cv2.imread(image_path),
                                                                                        facial_landmarks, num_augmentations)
                    for augmented_landmarks_noise, agumented_images_noise in zip(augmented_landmarks_list, augumented_images_list):
                        augmented_graph = build_graph_with_master_node(
                            augmented_landmarks_noise, agumented_images_noise)
                        # 2 function below use to show graph on noise images, comment if you want to preprocess faster
                        # agumented_images_noise = (
                        #     agumented_images_noise * 255).astype(np.uint8)
                        # draw_graph_on_image(
                        #     augmented_graph, agumented_images_noise)
                        # break
                        #
                        images.append(augmented_graph)
                        labels.append(label)
                    print(f'Processed {idx + 1} of images in {split} set '
                          f'in 680/192/99 images')

        data[split]['images'] = images
        data[split]['labels'] = labels
    print('Finish preprocessing')
    return data


def main():
    data_dir = 'emotion_data'
    # data_dict_augmented = load_data(
    #     data_dir, augment=True, num_augmentations=3)

    # Load or preprocess data
    if os.path.exists('data_dict.pkl'):
        # Load previously processed data
        with open('data_dict.pkl', 'rb') as file:
            loaded_data_dict = pickle.load(file)
    else:
        # Preprocess and augment data
        data_dict_augmented = load_data(
            data_dir, augment=True, num_augmentations=3)

        loaded_data_dict = data_dict_augmented  # Use the freshly processed data
        with open('data_dict.pkl', 'wb') as file:
            pickle.dump(loaded_data_dict, file)
    # Example usage
    print(len(loaded_data_dict['train']['images']))


if __name__ == "__main__":
    main()
