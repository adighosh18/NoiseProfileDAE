from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Reshape
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten

input_layer = Input(shape=(784,))


def encoder_decoder(no_of_encoder_layers, no_of_decoder_layers, activation_function):
    if activation_function != "relu" and activation_function != "sigmoid" and activation_function != "linear":
        activation_function = "relu"

    if no_of_encoder_layers == 4:
        x = Reshape((28, 28, 1))(input_layer)
        x = Conv2D(32, (3, 3), activation=activation_function, padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

    elif no_of_encoder_layers == 5:
        x = Reshape((28, 28, 1))(input_layer)
        x = Conv2D(32, (3, 3), activation=activation_function, padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation=activation_function, padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

    elif no_of_encoder_layers == 6:
        x = Reshape((28, 28, 1))(input_layer)
        x = Conv2D(32, (3, 3), activation=activation_function, padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation=activation_function, padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation=activation_function, padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

    if no_of_decoder_layers == 4:
        x = Conv2D(32, (3, 3), activation=activation_function, padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation=activation_function, padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Flatten()(x)
        decoded = Dense(784, activation='sigmoid')(x)
        #decoded = x
        return decoded

    elif no_of_decoder_layers == 5:
        x = Conv2D(32, (3, 3), activation=activation_function, padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation=activation_function, padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        #decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        x = Flatten()(x)
        decoded = Dense(784, activation='sigmoid')(x)
        return decoded

    elif no_of_decoder_layers == 6:
        x = Conv2D(32, (3, 3), activation=activation_function, padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation=activation_function, padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation=activation_function, padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        #decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        x = Flatten()(x)
        decoded = Dense(784, activation='sigmoid')(x)
        return decoded


def kl_divergence(y_true, y_pred):
    # Ensure values are in valid range (0, 1)
    y_true = tf.clip_by_value(y_true, 1e-10, 1.0 - 1e-10)
    y_pred = tf.clip_by_value(y_pred, 1e-10, 1.0 - 1e-10)

    # Calculate KL divergence
    kl = tf.reduce_sum(y_true * tf.math.log(y_true / y_pred), axis=-1)

    return kl


def model(no_of_encoder_layers, no_of_decoder_layers, activation_function):
    decoded = encoder_decoder(no_of_encoder_layers, no_of_decoder_layers, activation_function)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=[kl_divergence])
    return autoencoder


def train_model(x_train_noise, X_train, x_test_noise, X_test, no_of_encoder_layers, no_of_decoder_layers,
                activation_function, no_of_epochs):
    autoencoder = model(no_of_encoder_layers, no_of_decoder_layers, activation_function)
    if x_train_noise.shape[1] == 28 and x_train_noise.shape[2] == 28:
        x_train_noise = x_train_noise.reshape(x_train_noise.shape[0], x_train_noise.shape[1]*x_train_noise.shape[2])
    if x_test_noise.shape[1] == 28 and x_test_noise.shape[2] == 28:
        x_test_noise = x_test_noise.reshape(x_test_noise.shape[0], x_test_noise.shape[1]*x_test_noise.shape[2])
    if X_train.shape[1] == 28 and X_train.shape[2] == 28:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
    if X_test.shape[1] == 28 and X_test.shape[2] == 28:
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
    #x_train_noise = x_train_noise.reshape(x_train_noise.shape[0], x_train_noise.shape[1]*x_train_noise.shape[2])
    #X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
    autoencoder.fit(x_train_noise, X_train, epochs=no_of_epochs, batch_size=128, validation_data=(x_test_noise, X_test))
    return autoencoder
