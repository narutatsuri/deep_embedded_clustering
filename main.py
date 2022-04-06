from util import *
from util.methods import *
import numpy as np
from keras.datasets import mnist
import sys


#* Dummmy example
# x = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
# mu = [np.array([2,0,0]), np.array([0,0.5,0.5])]

# q = np.array([q_dist(x[0],mu), q_dist(x[1],mu), q_dist(x[2],mu)])
# p = p_dist(q)
# print(p,q, kl_div(p, q))

# Get MNIST dataset
print("Loading dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
initializer_dataset = x_train[:1000].reshape((-1, mnist_dim))
#initializer_dataset = np.concatenate((x_train, x_test)).reshape((-1, mnist_dim))
print("Loaded dataset.")

# Initialize SAE
print("Initializing SAE...")
initializer = SAE(initializer_dataset)
print("Initialized SAE.")

# Reshape dataset
# initializer_dataset = initializer_dataset.reshape((-1, mnist_dim))

# Get initialized embeddings
print("Initialize embeddings...")
embeddings = initializer.encoder_model.predict(initializer_dataset)
print("Initialized embeddings.")

# Initialize DEC by training model on initial embeddings and running Lloyd's 
# algorithm on output of trained model
dec = DEC(dataset=initializer_dataset,
          initial_z=embeddings,
          data_dim=mnist_dim)

# Initialize clusters
dec.predict_embedding(initializer_dataset)
dec.cluster()

# Train DEC
dec.train(x=initializer_dataset)

print(dec.z, dec.mu)