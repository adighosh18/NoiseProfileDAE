from src.models.predict_model import final_kl_divergence

noise_profiles = ["Gaussian", "S&P", "Uniform", "Random"]
noise_factors = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
no_of_encoder_layers = [4, 5, 6]
no_of_decoder_layers = [4, 5, 6]
activation_functions = ["linear", "relu", "sigmoid"]
no_of_epochs = [5, 10, 20]