

#? Datasets-specific Parameters
# MNIST = 784
mnist_dim = 784

#? DEC-specific Parameters
embedding_dim = 10
dec_train_epochs = 100
mu_learning_rate = 0.0001

#? SAE-specific Parameters
sae_first_layer_dim = mnist_dim
sae_hidden_layer_1_dim = 500
sae_hidden_layer_2_dim = 300
sae_last_layer_dim = embedding_dim
sae_dropout_rate = 0.1
sae_train_epochs = 1

#? 
alpha = 1