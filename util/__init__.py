

#? Datasets-specific Parameters
# MNIST = 784
mnist_dim = 784
mnist_clusters = 10
toy_clusters = 4
no_samples = 750

#? Visualization-specific Parameters
representation_dim = 2

#? DEC-specific Parameters
embedding_dim = 10
dec_train_epochs = 20
mu_learning_rate = 1
z_learning_rate = 1
cluster_num = mnist_clusters
dec_initialization_epochs = 200
dec_z_epochs = 100
save_fig_dir = "img/"
use_preset_mu = False

#? SAE-specific Parameters
initialize_sae = True

sae_first_layer_dim = mnist_dim
sae_hidden_layer_1_dim = 500
sae_hidden_layer_2_dim = 500
sae_hidden_layer_3_dim = 2000
sae_last_layer_dim = embedding_dim
sae_dropout_rate = 0.2
sae_train_epochs = 20
sae_full_train_epochs = 40
lr_initial_value = 0.1
lr_decay_rate = 0.1

#? 
alpha = 1
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'