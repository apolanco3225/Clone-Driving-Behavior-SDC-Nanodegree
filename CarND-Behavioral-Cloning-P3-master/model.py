# import pakages
import cv2
import csv
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from keras.layers import ELU
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
import matplotlib.pyplot as plt
import matplotlib.colors as c

# fix error with Keras and TensorFlow
tf.python.control_flow_ops = tf

# set string values to the image folder and the csv
IMAGES_DIRECTORY = "data_1/IMG"
DRIVING_LOG = "data_1/driving_log.csv"

# get the information from the csv file
def load_driving_log(path):
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        rows = []
        for row in reader:
            row['center'] = row['center'].strip()
            row['left'] = row['left'].strip()
            row['right'] = row['right'].strip()
            row['throttle'] = float(row['throttle'])
            row['steering'] = float(row['steering'])
            row['brake'] = float(row['brake'])
            row['speed'] = float(row['speed'])
            rows.append(row)
    return rows


# normalize the image, that would make your training run faster and less likely to be stuck in local optima
def normalize(img):
    return img.astype("float") / 255

# since the raw image appears not valuable information like the front of the car and the sky is better to remove it
def crop_car_and_sky(img):
    return img[45:135, :, :]

# due to there is more samples of left curves is necesary to flip images to have the same representations in both sides
def horizontal_flip(img):
    return cv2.flip(img, 1)
    
# change the size of the image (new_shape[0], new_shape[1])
def resize(img, new_shape=(200, 66)):
    return cv2.resize(img, new_shape, interpolation=cv2.INTER_CUBIC)

# to simulate changes in the environment like shadows and light is generated some random bright changes
def apply_random_brightness(img):
    # change the color space to hsv
    img_hsv = c.rgb_to_hsv(img)
    # change the H and S channels
    img_hsv[:, :, 2] = img_hsv[:, :, 2] * (0.25 + np.random.uniform())
    # return to rgb
    return c.hsv_to_rgb(img_hsv)

# applying preprocessing steps
def pre_process(img, new_shape=(200, 66)):
    return normalize(resize(crop_car_and_sky(img), new_shape))

# selects a random image, modifies it and adds a value if the image is not from the front camera
def augment_single_row(row):
    camera_selection = np.random.choice(['center', 'left', 'right'])
    img_path = IMAGES_DIRECTORY + "/" + row[camera_selection]
    angle = row['steering']
    if camera_selection == 'left':
        angle += 0.275  # steer right to get back to the center
    elif camera_selection == 'right':
        angle -= 0.275  # steer left to get back to the center
    image = mpimg.imread(img_path)
    image = pre_process(apply_random_brightness(image))
    flip = np.random.choice(['yes', 'no'])
    if flip == 'yes':
        image = horizontal_flip(image)
        angle *= -1
    return image, angle

# augments a cluster or rows
def augment_batch(batch):
    X_batch, y_batch = [], []
    for row in batch:
        img, angle = augment_single_row(row)
        X_batch.append(img)
        y_batch.append(angle)
    return np.array(X_batch), np.array(y_batch)

# defines generator for the inputs
def data_generator(rows, batch_size=16):
    n_rows = len(rows)
    batches_per_epoch = n_rows // batch_size
    current_batch = 0
    while True:
        start = current_batch * batch_size
        end = start + batch_size
        batch = rows[start:end]
        X_batch, y_batch = augment_batch(batch)
        current_batch += 1
        if current_batch == batches_per_epoch:
            current_batch = 0
        yield X_batch, y_batch

# implements nvidia architecture, the ReLu activation function is modified by ELU since is better for negative weights
def get_model(image_shape):
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), init='normal', border_mode='valid', input_shape=image_shape))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), init='normal', border_mode='valid'))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), init='normal', border_mode='valid'))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init='normal', border_mode='valid'))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init='normal', border_mode='valid'))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(1164, init='normal'))
    model.add(ELU())

    model.add(Dense(100, init='normal'))
    model.add(ELU())

    model.add(Dense(50, init='normal'))
    model.add(ELU())

    model.add(Dense(1, init='normal')) 

    return model

# visualize the behavior of the loss and accuracy
def plot_history(h):
    print(h.history.keys())
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

# save the model
def save_model(m):
    model_json = m.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    m.save_weights("model.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    BATCH_SIZE = 64
    # load data.
    data = load_driving_log(DRIVING_LOG)
    # shuffle data
    np.random.shuffle(data)  
    TRAIN_SPLIT = 0.8  # 80% for training - 20% for validation
    split_point = int(len(data) * TRAIN_SPLIT)
    # split data
    train_data = data[:split_point]
    validation_data = data[split_point:]
    # get the model
    model = get_model(image_shape=(66, 200, 3))
    # set generator
    train_generator = data_generator(data, batch_size=BATCH_SIZE)
    validation_generator = data_generator(validation_data, batch_size=BATCH_SIZE)
    # compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    # train 
    history = model.fit_generator(train_generator, validation_data=validation_generator, nb_epoch=5,
                                  samples_per_epoch=50000, nb_val_samples=10000)

    save_model(model)
    plot_history(history)

