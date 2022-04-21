from util import *
from util.functions import *
from util.plotting import *
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import LearningRateScheduler
from keras import initializers
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment as linear_assignment
import sys


class DEC:
    """
    Deep Embedded Clustering implementation. 
    """
    def __init__(self, 
                 x,
                 dnn,
                 cluster_num):
        # Construct DNN
        self.cluster_num = cluster_num

        self.model = dnn

        self.model.compile(loss="mse", 
                           optimizer='adam')

        # Construct clustering
        print(bcolors.WARNING, "CHECK: Clustering with", cluster_num, "clusters. ", bcolors.ENDC)
        self.clustering = KMeans(n_clusters=cluster_num)
    
    def train(self, 
              x,
              labels,
              preset_mu = None,
              use_preset_mu=False):
        self.model.compile(loss="mse", 
                           optimizer='adam')
        # Initial z's
        self.z = self.predict_embedding(x)
        # Cluster and obtain mu's
        if use_preset_mu:
            self.mu = preset_mu
        else:
            self.clustering.fit(self.z)
            self.mu = self.clustering.cluster_centers_
        
        for epoch in range(dec_train_epochs):
            #* Update step
            # Obtain p and q distributions for current epoch
            self.q_dists = q_dist(self.z, self.mu)
            self.p_dists = p_dist(self.q_dists)
            # Update z's and mu's
            self.update_z(x)
            if not use_preset_mu:
                self.update_mu()
            # print("Cluster centroids at epoch ", epoch, ": ", self.mu)

            #* Visualization step
            # Print loss
            if not use_preset_mu:
                y = self.clustering.predict(self.z)
                acc, w = self.cluster_acc(labels, y)
                print("Epoch: ", epoch, 
                    "Loss: ", DEC.loss(self.p_dists, self.q_dists),
                    "Cluster Accuracy: ", str(acc) + "%")
            else:
                print("Epoch: ", epoch, 
                    "Loss: ", DEC.loss(self.p_dists, self.q_dists))
                            
            # Plot intermediate results
            tsne_z = TSNE(n_components=representation_dim, 
                          learning_rate='auto',
                          init='random').fit_transform(self.z)
            plot_embeddings(plot_result=False,
                embeddings=tsne_z,
                cluster_centroids=self.mu,
                labels=labels,
                name=str(epoch))
            
            #* Feedforward step
            # Obtain embeddings z for current epoch
            self.z = self.predict_embedding(x)
                
    def predict_embedding(self,
                          x):
        return self.model.predict(x)
    
    def predict_cluster(self,
                x):
        return self.clustering.predict(x)
    
    def update_z(self,
                 x):
        """
        """
        for i, z in enumerate(self.z):
            gradient = (alpha+1)/alpha * sum([(1 + np.linalg.norm(z-mu)**2/alpha)**-1 * (self.p_dists[i][j]-self.q_dists[i][j]) * (z-mu) for j, mu in enumerate(self.mu)])
            # print("before: ", updated_z[i])
            self.z[i] += gradient * z_learning_rate
            # print("after: ", updated_z[i])
            # print("gradient: ", gradient)
            
        self.model.fit(x        =   x,
                       y        =   self.z,
                       epochs   =   dec_z_epochs,
                       verbose=0)
        
    def update_mu(self):
        """
        """
        for j, mu in enumerate(self.mu):
            gradient = -(alpha+1)/alpha * sum([(1 + np.linalg.norm(z-mu)**2/alpha)**-1 * (self.p_dists[i][j]-self.q_dists[i][j]) * (z-mu) for i, z in enumerate(self.z)])    
            self.mu[j] += gradient * mu_learning_rate
    
    @staticmethod
    def loss(p, q):
        return kl_div(p, q)
    
    def cluster_acc(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max())+1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        count = 0
        for index, val in enumerate(ind[0]):
            count += w[val, ind[1][index]]
        return count*100/y_pred.size, w

    
    def cluster_accuracy(self, 
                         x,
                         labels):
        """
        Checks cluster accuracy. Cluster centroid class is determined by majority.
        Requires self.mu to be initialized.
        """
        correct = 0
        mu_votes = np.zeros((self.cluster_num, self.cluster_num))
        cluster_to_label = {}
        # Assign with centroid is which class
        embeddings = []
        if len(self.mu) != 0:
            z = self.predict_embedding(x)
            predictions = self.clustering.predict(z)
                        
            for index, prediction in enumerate(predictions):
                mu_votes[prediction][labels[index]] += 1
            for index, row in enumerate(mu_votes):
                cluster_to_label[index] = list(row).index(max(row))
                
            for index, prediction in enumerate(predictions):
                if cluster_to_label[prediction] == labels[index]:
                    correct += 1
                    
            
            return str(correct/len(labels) * 100)
    
class SAE:
    def __init__(self, 
                 dataset):
        """
        Constructor for SAE. Constructs SAE and trains invididual denoising
        autoencoders, then finetunes the combined model.
        INPUTS:     Dataset (Default is MNIST)
        RETURNS:    Encoder
        """
        # Construct loss monitor
        # self.callback = keras.callbacks.EarlyStopping(monitor="loss", 
        #                                               mode="min",
        #                                               patience=5,
        #                                               min_delta=0.01)
        print("Training individual layers...")
        # Train first layer
        first_layer_encoder, first_layer_decoder = self.train_autoencoder_layer(dataset,
                                                                                io_dim=sae_first_layer_dim,
                                                                                hidden_dim=sae_hidden_layer_1_dim,
                                                                                layer="first")
        # Get encoder outputs from first layer
        hidden_layer_dataset_1 = first_layer_encoder.predict(dataset)
        
        # Train first hidden layer
        hidden_layer_encoder_1, hidden_layer_decoder_1 = self.train_autoencoder_layer(hidden_layer_dataset_1,
                                                                                      io_dim=sae_hidden_layer_1_dim,
                                                                                      hidden_dim=sae_hidden_layer_2_dim,
                                                                                      layer="hidden")
        # Get encoder outputs from hidden layer
        hidden_layer_dataset_2 = hidden_layer_encoder_1.predict(hidden_layer_dataset_1)
        
        # Train second hidden layer
        hidden_layer_encoder_2, hidden_layer_decoder_2 = self.train_autoencoder_layer(hidden_layer_dataset_2,
                                                                                      io_dim=sae_hidden_layer_2_dim,
                                                                                      hidden_dim=sae_hidden_layer_3_dim,
                                                                                      layer="hidden")
        # Get encoder outputs from hidden layer
        final_layer_dataset = hidden_layer_encoder_2.predict(hidden_layer_dataset_2)
        
        # Train first layer
        last_layer_encoder, last_layer_decoder = self.train_autoencoder_layer(final_layer_dataset,
                                                                              io_dim=sae_hidden_layer_3_dim,
                                                                              hidden_dim=sae_last_layer_dim,
                                                                              layer="last")
        print("Trained individual layers.")
        
        print("Training final model...")
        #? Finetune model
        # Construct final model
        last_layer_encoder.layers[1].rate = 0
        hidden_layer_encoder_2.layers[1].rate = 0
        hidden_layer_encoder_1.layers[1].rate = 0
        first_layer_encoder.layers[1].rate = 0
        first_layer_decoder.layers[1].rate = 0
        hidden_layer_decoder_1.layers[1].rate = 0
        hidden_layer_decoder_2.layers[1].rate = 0
        last_layer_decoder.layers[1].rate = 0
        
        self.final_model_input = keras.Input(shape=(sae_first_layer_dim,))
        self.final_model_encoder = last_layer_encoder(hidden_layer_encoder_2(hidden_layer_encoder_1(first_layer_encoder(self.final_model_input))))
        self.final_model_decoder = first_layer_decoder(hidden_layer_decoder_1(hidden_layer_decoder_2(last_layer_decoder(self.final_model_encoder))))
        self.model = keras.Model(inputs=self.final_model_input, 
                                 outputs=self.final_model_decoder)
        
        
        # Construct encoder section
        self.encoder_model = keras.Model(inputs=self.final_model_input, 
                                         outputs=self.final_model_encoder)
        
        # Compile and train model
        # opt = keras.optimizers.Adam(learning_rate=lr_initial_value)
        # self.model.compile(loss="mse", 
        #                    optimizer=opt)
        self.model.compile(loss="mse", 
                           optimizer="adam")
        self.model.fit(x=dataset,
                       y=dataset,
                       epochs=sae_full_train_epochs,)
                    #    callbacks=[LearningRateScheduler(SAE.lr_scheduler, 
                    #                                     verbose=0)])
        print("Trained final model.")

    @staticmethod
    def lr_scheduler(epoch, lr):
        decay_rate = lr_decay_rate
        decay_step = 50
        if epoch % decay_step == 0 and epoch:
            return lr * decay_rate
        return lr
                
    def train_autoencoder_layer(self, 
                                dataset,
                                io_dim,
                                hidden_dim,
                                layer):
        """
        Adds first layer of autoencoder to model.
        Dataset default is MNIST dataset (784 dimensions).
        """
        x = keras.Input(shape=(io_dim,))
        x_tilde = keras.layers.Dropout(rate=sae_dropout_rate)(x)
        if layer=="last":
            h = keras.layers.Dense(hidden_dim,
                                   kernel_initializer=initializers.RandomNormal(stddev=0.01))(x_tilde)
        else:
            h = keras.layers.Dense(hidden_dim, 
                                   activation="relu",
                                   kernel_initializer=initializers.RandomNormal(stddev=0.01))(x_tilde)
                
        h_tilde = keras.layers.Dropout(rate=sae_dropout_rate)(h)
        if layer=="first":
            y = keras.layers.Dense(io_dim,
                                   kernel_initializer=initializers.RandomNormal(stddev=0.01))(h_tilde)
        else:
            y = keras.layers.Dense(io_dim, 
                                   activation="relu",
                                   kernel_initializer=initializers.RandomNormal(stddev=0.01))(h_tilde)
        
        # Construct model
        training_model = keras.Model(inputs=x, 
                                     outputs=y)
        
        # Train temporary model  
        # opt = keras.optimizers.Adam(learning_rate=lr_initial_value)
        # training_model.compile(loss="mse", 
        #                        optimizer=opt)
        # training_model.summary()
        training_model.compile(loss="mse", 
                               optimizer="adam")
        training_model.fit(x=dataset,
                           y=dataset,
                           epochs=sae_train_epochs,)
                           # callbacks=[LearningRateScheduler(SAE.lr_scheduler, 
                           #                                 verbose=0)])
                           
        h_input = keras.Input(shape=(hidden_dim,))
        decoder_dropout = training_model.layers[-2]
        decoder_dense = training_model.layers[-1]
        
        # Returns encoder and decoder part separately
        encoder = keras.Model(inputs=x, outputs=h)
        decoder = keras.Model(inputs=h_input, outputs=decoder_dense(decoder_dropout(h_input)))
        return encoder, decoder