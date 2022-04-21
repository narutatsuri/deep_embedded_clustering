from keras.datasets import mnist
from util import *
from util.functions import *
import numpy as np


def load_mnist_dataset():
    """
    Loads MNIST dataset.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    initializer_dataset = np.concatenate((x_train, x_test)).reshape((-1, mnist_dim))
    initializer_dataset_labels = np.concatenate((y_train, y_test))
    initializer_dataset, initializer_dataset_labels = sample_dataset(x = initializer_dataset,
                                                                    y = initializer_dataset_labels,
                                                                    no_samples = no_samples)
    initializer_dataset = initializer_dataset.reshape((-1, mnist_dim))/255
    
    return initializer_dataset, initializer_dataset_labels

def load_toy_dataset(dim,
                     big_number,
                     cluster_num):
    
    random_centers = []
    for index in range(cluster_num):
        center = np.zeros(dim)
        center[index] = big_number
        random_centers.append(center)
    points_per_cluster = int(no_samples/len(random_centers))
    x = []
    labels = []

    # Generate simple dataset
    for index, center in enumerate(random_centers):
        points = np.random.randn(dim, points_per_cluster)
        points = points.reshape((points_per_cluster,dim,))
        points += np.array(center)
        x += list(points)
        labels += [index] * points_per_cluster
    
    return np.array(x), labels, random_centers
