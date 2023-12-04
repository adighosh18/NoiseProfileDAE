from keras.datasets import mnist, fashion_mnist, cifar10


def imageset(imageset_name) :
    if imageset_name == "MNIST" or imageset_name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        return X_train, y_train, X_test, y_test
    elif imageset_name == "FASHION_MNIST" or imageset_name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        return X_train, y_train, X_test, y_test
    elif imageset_name == "CIFAR10" or imageset_name == "cifar10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        return X_train, y_train, X_test, y_test
