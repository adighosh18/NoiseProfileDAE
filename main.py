from src.models.predict_model import final_kl_divergence

imagesets = ["MNIST", "FASHION_MNIST", "CIFAR10"]
noise_profiles = ["Gaussian", "S&P", "Uniform", "Random"]
noise_factors = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
no_of_encoder_layers_arr = [4, 5, 6]
no_of_decoder_layers_arr = [4, 5, 6]
activation_functions = ["linear", "relu", "sigmoid"]
no_of_epochs_arr = [5, 10, 20]

if __name__ == "__main__":
    imageset_name = "FASHION_MNIST"
    noise_profile = "Uniform"
    noise_factor = 0.35
    no_of_encoder_layers = 4
    no_of_decoder_layers = 6
    activation_function = "sigmoid"
    no_of_epochs = 5

    print(final_kl_divergence(imageset_name, noise_profile, noise_factor, no_of_encoder_layers, no_of_decoder_layers,
                              activation_function, no_of_epochs))
