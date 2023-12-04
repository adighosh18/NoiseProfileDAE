import numpy as np
import random
import cv2

def salt_and_pepper_noise(X, salt_prob, pepper_prob):
    """
    Add salt and pepper noise to 2D image data.
    salt_prob: Probability of adding salt (white) noise.
    pepper_prob: Probability of adding pepper (black) noise.
    """
    noisy_images = np.copy(X)
    num_images, num_rows, num_cols = X.shape

    for i in range(num_images):
        for row in range(num_rows):
            for col in range(num_cols):
                rand = random.random()
                if rand < salt_prob:
                    noisy_images[i, row, col] = 255  # Salt noise
                elif rand < salt_prob + pepper_prob:
                    noisy_images[i, row, col] = 0    # Pepper noise

    return noisy_images

def add_noise(X_train, X_test, noise_type, noise_factor):
    if noise_type.lower() == "gaussian":
        x_train_guassian = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0,
                                                                     size=X_train.shape)
        x_test_guassian = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0,
                                                                   size=X_test.shape)

        # Clip values to be between 0 and 1
        x_train_guassian = np.clip(x_train_guassian, 0., 1.)
        x_test_guassian = np.clip(x_test_guassian, 0., 1.)

        # Reshape the data for visualization
        # x_train_guassian_reshaped = x_train_guassian.reshape(-1, 28, 28)
        # x_test_guassian_reshaped = x_test_guassian.reshape(-1, 28, 28)

        num_pixels = x_train_guassian.shape[1] * x_test_guassian.shape[2]
        x_train_guassian = x_train_guassian.reshape(x_train_guassian.shape[0], num_pixels).astype(
            'float32')
        x_test_guassian = x_test_guassian.reshape(x_test_guassian.shape[0], num_pixels).astype(
            'float32')
        x_train_guassian = x_train_guassian / 255
        x_test_guassian = x_test_guassian / 255
        # print(MNIST_x_train_guassian.shape)
        # print(MNIST_x_test_guassian.shape)

        return x_train_guassian, x_test_guassian

    elif noise_type.lower() == "snp":
        # Adjust these probabilities as needed
        salt_prob = noise_factor  # Probability of salt noise
        pepper_prob = noise_factor  # Probability of pepper noise

        # Reshape MNIST data to 2D if it's not already
        x_train_reshaped = X_train.reshape(-1, 28, 28)
        x_test_reshaped = X_test.reshape(-1, 28, 28)

        # Add salt-and-pepper noise to the MNIST data
        x_train_snp = salt_and_pepper_noise(x_train_reshaped, salt_prob, pepper_prob)
        x_test_snp = salt_and_pepper_noise(x_test_reshaped, salt_prob, pepper_prob)
        # MNIST_X_train_noisy and MNIST_X_test_noisy contain the noisy images.
        return x_train_snp, x_test_snp

    elif noise_type.lower() == "uniform":
        uni_noise = np.zeros((28, 28), dtype=np.uint8)
        cv2.randu(uni_noise, 0, 255)
        uni_noise = (uni_noise * noise_factor).astype(np.uint8)
        x_train_uniform = []
        x_test_uniform = []
        for i in range(0, X_train.shape[0]):
            img = cv2.add(X_train[i], uni_noise)
            x_train_uniform.append(img)
        for i in range(0, X_test.shape[0]):
            img = cv2.add(X_test[i], uni_noise)
            x_test_uniform.append(img)
        x_train_uniform = np.array(x_train_uniform)
        x_test_uniform = np.array(x_test_uniform)

        num_pixels = x_train_uniform.shape[1] * x_train_uniform.shape[2]
        x_train_uniform = x_train_uniform.reshape(x_train_uniform.shape[0], num_pixels).astype(
            'float32')
        x_test_uniform = x_test_uniform.reshape(x_test_uniform.shape[0], num_pixels).astype('float32')
        x_train_uniform = x_train_uniform / 255
        x_test_uniform = x_test_uniform / 255
        return x_train_uniform, x_test_uniform

    elif noise_type.lower() == "random":
        x_train_random = []
        x_test_random = []
        intensity = noise_factor * 100
        for i in range(0, X_train.shape[0]):
            noisy_image = X_train[i].copy()
            noise = np.random.randint(-intensity, intensity + 1, noisy_image.shape)
            noisy_image = np.clip(noisy_image + noise, 0, 255).astype(np.uint8)
            x_train_random.append(noisy_image)
        for i in range(0, X_test.shape[0]):
            noisy_image = X_test[i].copy()
            noise = np.random.randint(-intensity, intensity + 1, noisy_image.shape)
            noisy_image = np.clip(noisy_image + noise, 0, 255).astype(np.uint8)
            x_test_random.append(noisy_image)
        x_train_random = np.array(x_train_random)
        x_test_random = np.array(x_test_random)
        num_pixels = x_train_random.shape[1] * x_train_random.shape[2]
        x_train_random = x_train_random.reshape(x_train_random.shape[0], num_pixels).astype('float32')
        x_test_random = x_test_random.reshape(x_test_random.shape[0], num_pixels).astype('float32')
        x_train_random = x_train_random / 255
        x_test_random = x_test_random / 255
        return x_train_random, x_test_random




