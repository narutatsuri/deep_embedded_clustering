import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from util import *
from util.methods import *
from util.plotting import *
from util.datasets import *
import numpy as np
import sys
from sklearn.manifold import TSNE


initializer_dataset, initializer_dataset_labels = load_mnist_dataset()
# initializer_dataset, initializer_dataset_labels, random_centers = load_toy_dataset(mnist_dim, 4, toy_clusters)
# print(bcolors.WARNING, "CHECK: Using toy dataset.", bcolors.ENDC)

# Initialize SAE
print("Initializing and training SAE...")
initializer = SAE(initializer_dataset)
print("Initialized and trained SAE.")

# Get initialized embeddings
print("Plotting initial points...")
embeddings = initializer.encoder_model.predict(initializer_dataset)
initial_z = TSNE(n_components=representation_dim, 
            learning_rate='auto',
            init='random').fit_transform(embeddings)

plot_embeddings(plot_result=True,
                embeddings=initial_z,
                cluster_centroids=None,
                labels=initializer_dataset_labels)

# Initialize DEC by training model on initial embeddings and running Lloyd's 
# algorithm on output of trained model
print("Initializing DEC...")
dec = DEC(x             =   initializer_dataset,
          dnn           =   initializer.encoder_model,
          cluster_num   =   cluster_num)
print("Initialized DEC.")

if use_preset_mu:
    preset_mus = np.array([np.zeros(embedding_dim), np.zeros(embedding_dim)])
    preset_mus[0][0] += 1
    preset_mus[1][1] += 1
    print(bcolors.WARNING, "CHECK: Using preset centroids. Centers are:", bcolors.ENDC)
    print(preset_mus)
else:
    preset_mus = None
print("Training DEC...")
dec.train(x=initializer_dataset,
          labels=initializer_dataset_labels,
          preset_mu=preset_mus,
          use_preset_mu=use_preset_mu)
print("Trained DEC.")

tsne_z = TSNE(n_components=representation_dim, 
              learning_rate='auto',
              init='random').fit_transform(dec.z)

# Plot embeddings and clusters
plot_embeddings(plot_result=True,
                embeddings=tsne_z,
                cluster_centroids=dec.mu,
                labels=initializer_dataset_labels)