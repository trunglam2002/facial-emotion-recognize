import numpy as np
import cv2


def apply_gaussian_noise(image, landmarks, sigma=1.0):
    height, width, channels = image.shape
    noise = np.random.normal(0, sigma, (height, width, channels))
    noisy_image = image + noise
    return noisy_image, landmarks


def apply_random_rotation(image, landmarks, angle_range=(-np.pi/10, np.pi/10)):
    height, width, _ = image.shape
    angle = np.random.uniform(angle_range[0], angle_range[1])

    # Create a 2x3 rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2), np.degrees(angle), 1.0)

    # Apply rotation to landmarks directly
    rotated_landmarks = cv2.transform(landmarks, rotation_matrix)

    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image, rotated_landmarks


def apply_random_flip(image, landmarks, image_width):
    flipped_image = cv2.flip(image, 1)  # 1 for horizontal flip
    landmarks_cp = np.copy(landmarks)

    # Adjust landmarks for flipping (only x is changed, y remains the same)
    landmarks_cp[:, :, 0] = image_width - landmarks_cp[:, :, 0]
    flipped_landmarks = np.array(landmarks_cp)
    return flipped_image, flipped_landmarks
