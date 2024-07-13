import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D, concatenate, Flatten, Dense, Reshape
from tensorflow.keras.models import Model

class UNet(tf.keras.Model):
    def __init__(self, latent_dim=30):
        super(UNet, self).__init__()
        self.latent_dim = latent_dim
        
        self.index_list = []
        self.train_loss_list = []
        self.val_loss_list = []
        
        # Encoder
        self.encoder_block1 = self.build_encoder_block(16, 16)
        self.encoder_block2 = self.build_encoder_block(16, 32)
        
        # Bottleneck with latent space
        self.bottleneck_conv = tf.keras.Sequential([
            Conv1D(64, 3, activation='relu', padding='same'),
            Conv1D(64, 3, activation='relu', padding='same'),
            Flatten(),
            Dense(self.latent_dim, activation='relu'),  # Latent dimension
            
        ])
        self.bottleneck_dense = tf.keras.Sequential([
            Dense(64 * 20, activation='relu'),  # Adjust based on your input size
            Reshape((20, 64))  # Adjust based on your input size
        ])

        # Decoder
        self.decoder_block2 = self.build_decoder_block(64, 32)
        self.decoder_block1 = self.build_decoder_block(48, 16)  # Adjust input filters to match concatenation

        # Final layer to restore original dimensions
        self.final_conv = Conv1D(1, 1, activation='linear', padding='same')
        
        # Loss
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def build_encoder_block(self, input_filters, output_filters):
        return tf.keras.Sequential([
            Conv1D(input_filters, 3, activation='relu', padding='same'),
            Conv1D(output_filters, 3, activation='relu', padding='same'),
            MaxPooling1D(2, padding='same')
        ])
    
    def build_decoder_block(self, input_filters, output_filters):
        return tf.keras.Sequential([
            Conv1D(input_filters, 3, activation='relu', padding='same'),
            Conv1D(output_filters, 3, activation='relu', padding='same'),
            UpSampling1D(2)
        ])

    def encode(self, inputs):
        e1 = self.encoder_block1(inputs)
        e2 = self.encoder_block2(e1)
        
        # Bottleneck
        latent = self.bottleneck_conv(e2) 
        
        return e1, latent  # Return e1 for use in the decoder

    def decode(self, e1, latent):
        encoded = self.bottleneck_dense(latent) 
        d2 = self.decoder_block2(encoded)
        d1 = self.decoder_block1(concatenate([e1, d2], axis=-1))
        decoded = self.final_conv(d1)
        return decoded

    def call(self, inputs):
        e1, latent = self.encode(inputs)
        decoded = self.decode(e1, latent)
        return decoded

    def loss(self, inputs, reconstructed):
        return self.loss_fn(inputs, reconstructed)

