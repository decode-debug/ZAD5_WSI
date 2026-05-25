import numpy as np
from scipy.ndimage import maximum_filter, shift
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.ndimage import maximum_filter

class DataLoader:
    X_train = None
    y_train = None
    X_val = None
    y_val = None
    X_test = None
    y_test = None

    @classmethod
    def load_data(cls):
        """
        Loading class. Performs:
        1. Data retrieval (scale 0-16)
        2. Split into datasets
        3. Augmentation (ONLY for the training set)
        4. Normalization of all datasets (scale 0.0 - 1.0)
        """
        if cls.X_train is not None:
            return  # Data is already loaded and processed

        # 1. Data retrieval (scale 0-16)
        digits = load_digits()
        X, y = digits.data, digits.target

        # 2. Split into train (70%), val (15%), test (15%)
        X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        # 3. Augmentation ONLY on the training set (on data scaled 0-16)
        X_train_aug, y_train_aug = cls.augment_training_data(X_train_raw, y_train_raw)

        # 4. Normalization of all datasets (scale 0.0 - 1.0)
        cls.X_train = cls.normalize_images(X_train_aug)
        cls.y_train = y_train_aug

        cls.X_val = cls.normalize_images(X_val_raw)
        cls.y_val = y_val_raw

        cls.X_test = cls.normalize_images(X_test_raw)
        cls.y_test = y_test_raw

    @classmethod
    def get_train_data(cls):
        """Returns the augmented and normalized training set."""
        cls.load_data()
        return cls.X_train, cls.y_train

    @classmethod
    def get_val_data(cls):
        """Returns the normalized validation set."""
        cls.load_data()
        return cls.X_val, cls.y_val

    @classmethod
    def get_test_data(cls):
        """Returns the normalized test set."""
        cls.load_data()
        return cls.X_test, cls.y_test

    @classmethod
    def get_all_data(cls):
        """Returns all normalized datasets."""
        cls.load_data()
        return (cls.X_train, cls.y_train), (cls.X_val, cls.y_val), (cls.X_test, cls.y_test)

    # ---------------------------------------------------------
    # Static methods (Helpers)
    # ---------------------------------------------------------

    @staticmethod
    def pixel_dropout(image_1d, drop_count=1):
        img_copy = image_1d.copy()
        bright_pixels = np.where(img_copy > 5)[0]

        if len(bright_pixels) > drop_count:
            to_drop = np.random.choice(bright_pixels, drop_count, replace=False)
            img_copy[to_drop] = 0

        return img_copy

    @staticmethod
    def thicken_image(image_1d):
        image_2d = image_1d.reshape(8, 8)
        thickened_2d = maximum_filter(image_2d, size=2)
        thickened_1d = np.clip(thickened_2d.flatten(), 0, 16)
        return thickened_1d

    @staticmethod
    def shift_image(image_1d, dy, dx):
        """
        Shifts the image by dy (vertical) and dx (horizontal) pixels.
        Empty spaces are filled with 0 (black).
        """
        image_2d = image_1d.reshape(8, 8)
        shifted_2d = shift(image_2d, shift=[dy, dx], mode='constant', cval=0)
        return shifted_2d.flatten()

    @staticmethod
    def add_gaussian_noise(image_1d, std=1.5):
        """
        Adds random noise from a normal distribution.
        Keeps values clipped within the 0-16 bounds.
        """
        noise = np.random.normal(loc=0.0, scale=std, size=image_1d.shape)
        noisy_image = image_1d + noise
        return np.clip(noisy_image, 0, 16)

    @staticmethod
    def normalize_images(images):
        """Normalizes pixel values from [0, 16] to [0.0, 1.0]."""
        images_float = np.array(images, dtype=float)
        normalized_images = images_float / 16.0
        normalized_images = np.clip(normalized_images, 0.0, 1.0)
        return normalized_images

    @staticmethod
    def augment_training_data(X_train, y_train):
        """
        Augments the training set x4.
        Utilizes dropout, thickening, and noise addition.
        """
        augmented_images = []
        augmented_labels = []

        for img, label in zip(X_train, y_train):
            # 1. Original
            augmented_images.append(img)
            augmented_labels.append(label)

            # 2. Dropout
            dropout_img = DataLoader.pixel_dropout(img, drop_count=1)
            augmented_images.append(dropout_img)
            augmented_labels.append(label)

            # 3. Thicken
            thickened_img = DataLoader.thicken_image(img)
            augmented_images.append(thickened_img)
            augmented_labels.append(label)

            # 4. Shift (Randomly by 1 pixel in X and/or Y direction)
            # np.random.choice([-1, 0, 1]) picks a random shift direction
            dy, dx = np.random.choice([-1, 0, 1], size=2)
            # Ensure it actually shifts (if both are 0, force a shift)
            if dy == 0 and dx == 0:
                dy = 1
            shifted_img = DataLoader.shift_image(img, dy=dy, dx=dx)
            augmented_images.append(shifted_img)
            augmented_labels.append(label)

            # 5. Gaussian Noise
            noisy_img = DataLoader.add_gaussian_noise(img, std=1.5)
            augmented_images.append(noisy_img)
            augmented_labels.append(label)

        return np.array(augmented_images), np.array(augmented_labels)