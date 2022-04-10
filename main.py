import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from util import *
from util.methods import *
from util.plotting import *
import numpy as np
from keras.datasets import mnist
import sys
from sklearn.manifold import TSNE


# Get MNIST dataset
print("Loading dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
initializer_dataset = np.concatenate((x_train, x_test)).reshape((-1, mnist_dim))
initializer_dataset_labels = np.concatenate((y_train, y_test))
initializer_dataset, initializer_dataset_labels = sample_dataset(x = initializer_dataset,
                                                                 y = initializer_dataset_labels,
                                                                 no_samples = 5000)
initializer_dataset = initializer_dataset.reshape((-1, mnist_dim))/255

# initializer_dataset_labels = np.concatenate((y_train, y_test))
print("Loaded dataset.")

# Initialize SAE
print("Initializing SAE...")
initializer = SAE(initializer_dataset)
print("Initialized SAE.")

# Get initialized embeddings
print("Initializing embeddings...")
embeddings = initializer.encoder_model.predict(initializer_dataset)
print("Initialized embeddings.")

print(embeddings)

print("Plotting initial points...")
initial_z = TSNE(n_components=representation_dim, 
              learning_rate='auto',
              init='random').fit_transform(embeddings)

# Plot embeddings and clusters
plot_embeddings(plot_result=True,
                embeddings=initial_z,
                cluster_centroids=None,
                labels=initializer_dataset_labels)

# Initialize DEC by training model on initial embeddings and running Lloyd's 
# algorithm on output of trained model
dec = DEC(dataset       =   initializer_dataset,
          initial_z     =   embeddings,
          data_dim      =   mnist_dim,
          cluster_num   =   cluster_num)

# Train DEC
dec.train(x=initializer_dataset,
          labels=initializer_dataset_labels)

tsne_z = TSNE(n_components=representation_dim, 
              learning_rate='auto',
              init='random').fit_transform(dec.z)

# Plot embeddings and clusters
plot_embeddings(plot_result=True,
                embeddings=tsne_z,
                cluster_centroids=dec.mu,
                labels=initializer_dataset_labels)

#* Applying t-SNE to the original dataset
# tsne_x = TSNE(n_components=representation_dim, 
#               learning_rate='auto',
#               init='random').fit_transform(initializer_dataset)

# plot_embeddings(plot_result=True,
#                 embeddings=tsne_x,
#                 cluster_centroids=None,
#                 labels=initializer_dataset_labels)