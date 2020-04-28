import numpy as np

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Flatten, Dense, Lambda, ELU, Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from keras.layers.normalization import BatchNormalization 

# Local libraries
from image_preprocessing import *
from dataset_utilities import *

# ------------------ Control variables ------------------

# Paths and names
DATA_PATH = '/home/workspace/CarND-Behavioral-Cloning-P3/Data/'
CSV_NAME = 'driving_log.csv'
MODEL_NAME = 'model_final_hsv_elu_pertubed_64.h5'

# Image Dimensions
IMAGE_TARGET_SIZE = (64, 64)
VERTICAL_CROP = (55, 135)
MODEL_INPUT_SHAPE = (64, 64, 3)

# Filetering
THRESHOLD = 0.01
FRACTION_TO_REMOVE = 0.7
TRAINING_SPLIT = 0.8

# CNN parameters
BATCH_SIZE = 64
LEARNING_RATE = 1.0e-4
EPOCHS = 50
ACTIVATION = 'elu'

# Correction and state
STEERING_CORRECTION = 0.25
RANDOM_STATE = 17


# ------------------ Generator and Model ------------------

def get_data_generator(data_path, data_frame, batch_size, steering_correction, vertical_crop, size):
    '''
    Generates data for the model on the fly.
    
    Parameters:
        - data_path: Path of the data (root)
        - data_frame: Dataframe that contains the training set
        - batch_size: Desired size of each batch to feed the CNN
        - steering_correction: Correction of the steering angle
        - vertical_crop: Top and bottom points to crop
        - size: New size of the image
        
    Output:
        - image: Perturbed image
        - steering: Steering value of the image
    '''
    
    N = data_frame.shape[0]
    batches_per_epoch = N // batch_size

    i = 0
    while(True):
        start = i * batch_size
        end = start + batch_size - 1
        
        # Batches to return
        X_batch = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
        y_batch = np.zeros((batch_size,), dtype = np.float32)

        j = 0

        # slice a `batch_size` sized chunk from the dataframe
        # and perturb data for each row in the chunk on the fly
        for index, row in data_frame.loc[start:end].iterrows():
            image, steering =  perturb_image(data_path, row, steering_correction, vertical_crop, size)
            X_batch[j]  = image
            y_batch[j] = steering
            j += 1

        i += 1
        # If epoch finished reset the index to start all over again
        if i == batches_per_epoch - 1:
            i = 0
            
        yield X_batch, y_batch


def get_model(input_shape, activation):
    """
    Generates the CNN model.
    
    Parameters:
        - input_shape: Input shape of the network
        - activation: Activation function to use

    Output:
        - CNN model
    """
    model = Sequential()
    
    model.add(Conv2D(3, 1, 1, subsample=(2, 2), input_shape = input_shape, activation = activation, name='cv0'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(16, 3, 3, activation = activation, name = 'cv1'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(name='maxPool_cv1'))
    model.add(Dropout(0.3, name='dropout_cv1'))

    model.add(Conv2D(32, 3, 3, activation = activation, name = 'cv2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(name='maxPool_cv2'))
    model.add(Dropout(0.5, name='dropout_cv2'))

    model.add(Conv2D(64, 3, 3, activation = activation, name = 'cv3'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(name='maxPool_cv3'))
    model.add(Dropout(0.5, name='dropout_cv3'))

    model.add(Flatten())

    model.add(Dense(1000, activation = activation, name = 'fc1'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5, name='dropout_fc1'))

    model.add(Dense(100, activation = activation, name = 'fc2'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5, name='dropout_fc2'))

    model.add(Dense(10, activation = activation, name = 'fc3'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5, name='dropout_fc3'))

    model.add(Dense(1, name='output'))
    
    return model



def main():
    """
    Main function
    """
    print('1.- Loading data....')
    data_frame = load_data(DATA_PATH + CSV_NAME, RANDOM_STATE)
    
    print('2.- Filtering and extending...')
    training_data_frame, validation_data_frame, _ = filter_dataset(data_frame, 
                                                                   THRESHOLD, 
                                                                   FRACTION_TO_REMOVE, 
                                                                   TRAINING_SPLIT,
                                                                   RANDOM_STATE)
    print('3.- Calling generators...')
    number_training_batches = training_data_frame.shape[0] // BATCH_SIZE
    training_generator = get_data_generator(DATA_PATH,
                                            training_data_frame, 
                                            BATCH_SIZE, 
                                            STEERING_CORRECTION,
                                            VERTICAL_CROP, 
                                            IMAGE_TARGET_SIZE) 
    
    number_validation_batches = validation_data_frame.shape[0] // BATCH_SIZE
    validation_generator = get_data_generator(DATA_PATH,
                                              validation_data_frame,
                                              BATCH_SIZE, 
                                              STEERING_CORRECTION,
                                              VERTICAL_CROP, 
                                              IMAGE_TARGET_SIZE)
   
    print('4.- Building the CNN model...')
    model = get_model(MODEL_INPUT_SHAPE, ACTIVATION)

    adam = Adam(lr = LEARNING_RATE, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay =0.0)
    model.compile(optimizer = 'adam', loss = 'mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

    print('5.- Training the CNN model...')
    model.fit_generator(training_generator, 
        validation_data = validation_generator, 
        epochs = EPOCHS, 
        callbacks=[save_weights], 
        steps_per_epoch = number_training_batches, 
        validation_steps = number_validation_batches)
    
    # Save the model
    print('6.- Saving the CNN model...')
    model.save(MODEL_NAME)
    print("*** Finished! ***")


if __name__ == '__main__':
    main()
