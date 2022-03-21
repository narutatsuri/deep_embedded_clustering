from util.modules import *
from util import *
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.cluster import KMeans


class DEC:
    """
    Deep Embedded Clustering implementation. 
    """
    def __init__(self, 
                 dataset,
                 initial_z,
                 data_dim):
        #! What is the activation used?        
        self.model = Sequential()
        self.model.add(Dense(500, input_dim=data_dim, activation='relu'))
        self.model.add(Dense(500, input_dim=500, activation='relu'))
        self.model.add(Dense(2000, input_dim=500, activation='relu'))
        self.model.add(Dense(10, input_dim=2000, activation='relu'))

        self.model.compile(loss="mse", 
                           optimizer='adam', 
                           metrics=['accuracy'])
        self.model.fit(x=dataset,
                       y=initial_z)
    
    def train(self, 
              x):
        self.model.compile(loss=DEC.loss, 
                           optimizer='adam', 
                           metrics=['accuracy'])
        
        for _ in range(dec_train_epochs):
            self.predict_embedding(x)
            self.cluster()
            # Update model
            self.model.fit(x=x,
                           y=self.mu)
            # Update mu's
            self.update_mu()
    
    def predict_embedding(self,
                          x):
        self.z = self.model.predict(x)
    
    def predict_cluster(self,
                x):
        return self.clustering.predict(x)
    
    @staticmethod
    def loss(mu,
             z):
        q = q_dist(z, mu)
        p = p_dist(q)
        return kl_div(q,
                      p)
    
    def update_mu(self):
        updated_mu = self.mu
        for index_1, mu in enumerate(self.mu):
            gradient = -(alpha+1)/alpha * sum([(1 + np.linalg.norm(z, mu)**2/alpha)**-1 * (p_dist(q_dist(z, mu))-q_dist(z, mu)) * (z-mu) for z in self.z])    
            updated_mu[index_1] += gradient * mu_learning_rate
        self.mu = updated_mu
    
    def cluster(self):
        self.clustering = KMeans(n_clusters=10).fit(self.z)
        self.mu = self.clustering.cluster_centers_
    
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
        self.callback = keras.callbacks.EarlyStopping(monitor="loss", 
                                                      mode="min",
                                                      patience=5,
                                                      min_delta=0.01)
        # Train first layer
        first_layer_encoder, first_layer_decoder = self.train_autoencoder_layer(dataset,
                                                                                io_dim=sae_first_layer_dim,
                                                                                hidden_dim=sae_hidden_layer_1_dim,
                                                                                layer="first")
        # Get encoder outputs from first layer
        hidden_layer_dataset = first_layer_encoder.predict(dataset)
        
        # Train hidden layer
        hidden_layer_encoder, hidden_layer_decoder = self.train_autoencoder_layer(hidden_layer_dataset,
                                                                                  io_dim=sae_hidden_layer_1_dim,
                                                                                  hidden_dim=sae_hidden_layer_2_dim,
                                                                                  layer="hidden")
        # Get encoder outputs from hidden layer
        final_layer_dataset = hidden_layer_encoder.predict(hidden_layer_dataset)
        
        # Train first layer
        last_layer_encoder, last_layer_decoder = self.train_autoencoder_layer(final_layer_dataset,
                                                                              io_dim=sae_hidden_layer_2_dim,
                                                                              hidden_dim=sae_last_layer_dim,
                                                                              layer="last")
        
        #? Finetune model
        # Construct final model
        self.final_model_input = keras.Input(shape=(sae_first_layer_dim,))
        self.final_model_encoder = last_layer_encoder(hidden_layer_encoder(first_layer_encoder(self.final_model_input)))
        self.final_model_decoder = first_layer_decoder(hidden_layer_decoder(last_layer_decoder(self.final_model_encoder)))
        self.model = keras.Model(inputs=self.final_model_input, 
                                 outputs=self.final_model_decoder)
        
        # Construct encoder section
        self.encoder_model = keras.Model(inputs=self.final_model_input, 
                                         outputs=self.final_model_encoder)
        
        # Compile and train model
        self.model.compile(loss="mse", optimizer="rmsprop")
        self.model.fit(x=dataset,
                       y=dataset,
                       epochs=sae_train_epochs,
                       callbacks=[self.callback])
                
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
            h = keras.layers.Dense(hidden_dim,)(x_tilde)
        else:
            h = keras.layers.Dense(hidden_dim, activation="relu")(x_tilde)
                
        h_tilde = keras.layers.Dropout(rate=sae_dropout_rate)(h)
        if layer=="first":
            y = keras.layers.Dense(io_dim,)(h_tilde)
        else:
            y = keras.layers.Dense(io_dim, activation="relu")(h_tilde)
        
        # Construct model
        training_model = keras.Model(inputs=x, 
                                     outputs=y)
        
        # Train temporary model  
        training_model.compile(loss="mse", optimizer="rmsprop")
        training_model.summary()
        
        training_model.fit(x=dataset,
                           y=dataset,
                           epochs=sae_train_epochs,
                           callbacks=[self.callback])
                           
        h_input = keras.Input(shape=(hidden_dim,))
        decoder_dropout = training_model.layers[-2]
        decoder_dense = training_model.layers[-1]
        
        # Returns encoder and decoder part separately
        encoder = keras.Model(inputs=x, outputs=h)
        decoder = keras.Model(inputs=h_input, outputs=decoder_dense(decoder_dropout(h_input)))
        return encoder, decoder