import matplotlib.pyplot as plt
from src.data.make_dataset import imageset
from src.features.build_features import add_noise
from src.models.train_model import train_model


def generate_images(autoencoder, x_test_noise, X_test):
    denoised_images = autoencoder.predict(x_test_noise)

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))  # 1 row, 3 columns

    # Display the first image in the original test set
    axes[0].imshow(X_test[0], cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')  # Turn off axis numbers and ticks

    # Display the first image in the noisy test set
    axes[1].imshow(x_test_noise[0].reshape(28, 28), cmap='gray')
    axes[1].set_title("Noisy Image")
    axes[1].axis('off')

    # Display the first image in the denoised test set
    axes[2].imshow(denoised_images[0].reshape(28, 28), cmap='gray')
    axes[2].set_title("Denoised Image")
    axes[2].axis('off')

    # Set the overall title for the figure
    plt.suptitle('Comparison of Original, Noisy, and Denoised Images', fontsize=16)

    # Show the plot with all three images
    plt.tight_layout()  # Adjust the layout so the titles and images don't overlap
    plt.show()


def final_kl_divergence(imageset_name, noise_type, noise_factor, no_of_encoder_layers, no_of_decoder_layers,
                        activation_function, no_of_epochs):
    X_train, y_train, X_test, y_test = imageset(imageset_name)
    x_train_noise, x_test_noise = add_noise(X_train, X_test, noise_type, noise_factor)
    autoencoder = train_model(x_train_noise, X_train, x_test_noise, X_test, no_of_encoder_layers, no_of_decoder_layers,
                              activation_function, no_of_epochs)
    generate_images(autoencoder, x_test_noise, X_test)
    kl_divergence = autoencoder.evaluate(x_test_noise, X_test, verbose=0)[1]
    return kl_divergence
