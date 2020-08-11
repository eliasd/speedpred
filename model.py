#!/Users/dguillen/opt/anaconda3/bin/python3

import tensorflow as tf


# TODO:
#   - try batch normalization
#   - try no normalization
#   - try no max pool

def get_speed_prediction_network(output_window_size=4):
    return SpeedPredictionNetwork(output_window_size)

class SpeedPredictionNetwork(tf.keras.Model):
    def __init__(self, output_window_size):
        super(SpeedPredictionNetwork, self).__init__()

        initializer = tf.initializers.VarianceScaling(scale=2.0)

        # Layer #1: Padding + Convolution + Batch Normalization + Relu + Dropout.
        self.conv_h1 = tf.keras.layers.Conv2D(filters=24, 
                                              kernel_size=5, 
                                              padding='same',
                                              kernel_initializer=initializer)
        self.bn_h1 = tf.keras.layers.LayerNormalization()
        self.relu_h1 = tf.keras.layers.ReLU()
        self.drop_h1 = tf.keras.layers.Dropout(0.30)
        
        # Layer #2: Padding + Convolution + Batch Normalization + RelU + Dropout.
        self.conv_h2 = tf.keras.layers.Conv2D(filters=32, 
                                              kernel_size=3,
                                              padding='same',
                                              kernel_initializer=initializer)
        self.bn_h2 = tf.keras.layers.LayerNormalization()
        self.relu_h2 = tf.keras.layers.ReLU()
        self.drop_h2 = tf.keras.layers.Dropout(0.30)

        # Layer #3: Max-Pool
        self.maxPool_h3 = tf.keras.layers.MaxPool2D()

        # Layer #4: Padding + Convolution + Batch Normalization + ReLU + Dropout.
        self.conv_h4 = tf.keras.layers.Conv2D(filters=64,
                                               kernel_size=3,
                                               padding='same',
                                               kernel_initializer=initializer)
        self.bn_h4 = tf.keras.layers.LayerNormalization()
        self.relu_h4 = tf.keras.layers.ReLU()
        self.drop_h4 = tf.keras.layers.Dropout(0.30)

        # Layer #5: Padding + Convolution + Batch Normalization + ReLu + Dropout.
        self.conv_h5 = tf.keras.layers.Conv2D(filters=128,
                                               kernel_size=3,
                                               padding='same',
                                               kernel_initializer=initializer)
        self.bn_h5 = tf.keras.layers.LayerNormalization()
        self.relu_h5 = tf.keras.layers.ReLU()
        self.drop_h5 = tf.keras.layers.Dropout(0.30)

        # Flatten.
        self.flatten = tf.keras.layers.Flatten()

        # Layer #6: Fully-Connected + ReLU + Batch Normalization + Dropout.
        self.fc_h6 = tf.keras.layers.Dense(128, kernel_initializer=initializer)
        self.bn_h6 = tf.keras.layers.LayerNormalization()
        self.relu_h6 = tf.keras.layers.ReLU()
        self.drop_h6 = tf.keras.layers.Dropout(0.30)

        # Layer #7: Fully-Connected + ReLU + Batch Normalization + Dropout.
        self.fc_h7 = tf.keras.layers.Dense(128, kernel_initializer=initializer)
        self.bn_h7 = tf.keras.layers.LayerNormalization()
        self.relu_h7 = tf.keras.layers.ReLU()
        self.drop_h7 = tf.keras.layers.Dropout(0.30)
 
        # Layer #8: Fully-Connected ==> Regression output of size output_window_size.
        self.fc_h8 = tf.keras.layers.Dense(output_window_size,
                                           kernel_initializer=initializer)
    
    def call(self, input_tensor, training=False):
        # Layer #1.
        h1 = self.conv_h1(input_tensor)
        h1 = self.bn_h1(h1, training=training)
        h1 = self.relu_h1(h1)
        h1 = self.drop_h1(h1, training=training)

        # Layer #2.
        h2 = self.conv_h2(h1)
        h2 = self.bn_h2(h2, training=training)
        h2 = self.relu_h2(h2)
        h2 = self.drop_h2(h2, training=training)

        # Layer #3.
        h3 = self.maxPool_h3(h2)

        # Layer #4
        h4 = self.conv_h4(h3)
        h4 = self.bn_h4(h4, training=training)
        h4 = self.relu_h4(h4)
        h4 = self.drop_h4(h4, training=training)

        # Layers #5.
        h5 = self.conv_h5(h4)
        h5 = self.bn_h5(h5, training=training)
        h5 = self.relu_h5(h5)
        h5 = self.drop_h5(h5, training=training)

        # Layer #6.
        h6 = self.flatten(h5)
        h6 = self.fc_h6(h6)
        h6 = self.bn_h6(h6, training=training)
        h6 = self.relu_h6(h6)
        h6 = self.drop_h6(h6, training=training)

        # Layer #7.
        h7 = self.fc_h7(h6)
        h7 = self.bn_h7(h7, training=training)
        h7 = self.relu_h7(h7)
        h7 = self.drop_h7(h7, training=training)

        # Layer #5.
        x = self.fc_h8(h7)
        
        return x
