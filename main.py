import matplotlib.pyplot as plt
import keras, os, shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow
# from google.colab import drive
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import tensorflow as tf

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))


def mod():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7, activation='softmax'))
    return model


def graphs(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    perplexity = history.history['perplexity']
    val_perplexity = history.history['val_perplexity']
    #epochs = range(len(acc))
    print(len(acc))
    plt.plot(range(len(acc)), acc, 'r', label='Training acc')
    plt.plot(range(len(acc)), val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(range(len(loss)), loss, 'r', label='Training loss')
    plt.plot(range(len(loss)), val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.figure()
    plt.plot(range(len(perplexity)), perplexity, 'r', label='Perplexity')
    plt.plot(range(len(perplexity)), val_perplexity, 'b', label='Validation Perplexity')
    plt.title('Perplexity and Validation Perplexity')
    plt.legend()

    plt.show()


def perplexity(y_true, y_pred):
    cross_entropy = tensorflow.losses.categorical_crossentropy(y_true, y_pred)
    perplex = tensorflow.exp(tensorflow.reduce_mean(cross_entropy))
    return perplex


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cd = os.getcwd()
    print(cd)
    epochs = 60
    # train = os.path.join(cd, '\images\train')
    train = cd + '\\images\\train'
    validate = cd + '\\images\\validation'
    train_angry = train + '\\angry'
    train_disgust = train + '\\disgust'
    train_fear = train + '\\fear'
    train_happy = train + '\\happy'
    train_neutral = train + '\\neutral'
    train_sad = train + '\\sad'
    train_surprise = train + '\\surprise'
    validate_angry = validate + '\\angry'
    validate_disgust = validate + '\\disgust'
    validate_fear = validate + '\\fear'
    validate_happy = validate + '\\happy'
    validate_neutral = validate + '\\neutral'
    validate_sad = validate + '\\sad'
    validate_surprise = validate + '\\surprise'
    model = mod()
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy', perplexity])
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train,
                                                        target_size=(48, 48),
                                                        batch_size=64,
                                                        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(validate,
                                                            target_size=(48, 48),
                                                            batch_size=64,
                                                            class_mode='categorical')
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.1, min_lr=0.0001)
    filepath = 'model.weights11.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    callbacks = [checkpoint, reduce_lr]
    history = model.fit(
        train_generator,
        batch_size=250,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=110,
        callbacks=callbacks)
    model.save('.\modelSaved')
    graphs(history)
