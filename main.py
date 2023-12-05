from src.models.predict_model import final_kl_divergence
import os

imagesets = ["MNIST", "FASHION_MNIST"]
noise_profiles = ["Gaussian", "SnP", "Uniform", "Random"]
noise_factors = [0.25, 0.35, 0.45, 0.55]
no_of_encoder_layers_arr = [4, 5, 6]
no_of_decoder_layers_arr = [4, 5, 6]
activation_functions = ["linear", "relu", "sigmoid"]
no_of_epochs_arr = [5, 10, 15]


def find_project_root(start_dir, marker_filename):
    current_dir = start_dir

    # Continue searching parent directories until the marker file is found
    while True:
        marker_file = os.path.join(current_dir, marker_filename)

        if os.path.isfile(marker_file):
            return current_dir  # Found the marker file, so this is the project root

        # Move up one directory level
        parent_dir = os.path.dirname(current_dir)

        # Check if we have reached the root directory
        if parent_dir == current_dir:
            raise FileNotFoundError(f"Marker file '{marker_filename}' not found in project directory structure.")

        current_dir = parent_dir


if __name__ == "__main__":
    project_root = find_project_root(os.getcwd(), "main.py")

    for imageset_name in imagesets:
        for noise_profile in noise_profiles:
            for noise_factor in noise_factors:
                for no_of_encoder_layers in no_of_encoder_layers_arr:
                    for no_of_decoder_layers in no_of_decoder_layers_arr:
                        for activation_function in activation_functions:
                            for no_of_epochs in no_of_epochs_arr:
                                final_kl_divergence(project_root, imageset_name, noise_profile, noise_factor,
                                                    no_of_encoder_layers,
                                                    no_of_decoder_layers, activation_function, no_of_epochs)
