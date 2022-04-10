

#? Datasets-specific Parameters
# MNIST = 784
mnist_dim = 784
mnist_clusters = 10

#? Visualization-specific Parameters
representation_dim = 2

#? DEC-specific Parameters
embedding_dim = 10
dec_train_epochs = 20
mu_learning_rate = 1
z_learning_rate = 1
cluster_num = mnist_clusters
dec_initialization_epochs = 100
dec_z_epochs = 100
save_fig_dir = "img_normalized/"

#? SAE-specific Parameters
sae_first_layer_dim = mnist_dim
sae_hidden_layer_1_dim = 500
sae_hidden_layer_2_dim = 300
sae_last_layer_dim = embedding_dim
sae_dropout_rate = 0.2
sae_train_epochs = 500
sae_full_train_epochs = 1000
lr_initial_value = 0.1
lr_decay_rate = 0.1

#? 
alpha = 1