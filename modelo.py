#import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf

from tensorflow.keras.layers import BatchNormalization


def build_cnn_model(input_shape): 
    
    model = models.Sequential() 
    
    # Primera capa convolucional 
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape)) 
    model.add(layers.MaxPooling2D((2, 2))) 
    
    # Segunda capa convolucional 
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same')) 
    model.add(layers.MaxPooling2D((2, 2))) 
    
    # Tercera capa convolucional
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same')) 
    model.add(layers.MaxPooling2D((2, 2))) 
    
    # Cuarta capa convolucional 
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same')) 
    model.add(layers.MaxPooling2D((2, 2))) 
    
    # Capa de aplanamiento y fully connected 
    model.add(layers.Flatten()) 
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))) 
    model.add(layers.Dropout(0.5)) 
    model.add(layers.Dense(1, activation='sigmoid')) 
    
    # Compilar el modelo 
    model.compile(optimizer=RMSprop(learning_rate=0.0005), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy']) 
    
    return model
 



