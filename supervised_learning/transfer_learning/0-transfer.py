#!/usr/bin/env python3
"""Task 13 Keras"""
from matplotlib import pyplot as plt
import tensorflow.keras as K
import tensorflow as tf
import numpy as np
import time

def resize(X):
    """resize image"""
    return tf.image.resize(X, [224, 224])

def preprocess_data(X, Y):
    """Preprocess images to correct size"""
    X_new = K.applications.vgg16.preprocess_input(X)
    X_new = K.layers.Lambda(resize)(X_new)
    Y_new = K.utils.to_categorical(Y, num_classes=10)

    return X_new, Y_new


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    start_time = time.time()
    X_t, Y_t = preprocess_data(X_train[0:2000], Y_train[0:2000])
    print("--- %s seconds ---" % (time.time() - start_time))
    print(X_t.shape)
    start_time = time.time()
    X_v, Y_v = preprocess_data(X_train[3000:3500], Y_train[3000:3500])
    
    print("--- %s seconds ---" % (time.time() - start_time))
    print(X_v.shape)
    
    batch_size = 30
    num_classes = 10
    epochs = 10

    initializer = K.initializers.HeNormal()

    vgg_model = K.models.load_model("vgg16-train.h5")

    out = K.layers.Dense(512, activation='relu', kernel_initializer=initializer)(vgg_model.output)
    out = K.layers.Dense(512, activation='relu', kernel_initializer=initializer)(out)

    Y = K.layers.Dense(
        10,
        "softmax"
    )(out)

    model = K.models.Model(inputs=vgg_model.input, outputs=Y)

    model.compile(optimizer=K.optimizers.Adam(),
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    
    
    
    history = model.fit(x=X_t, y=Y_t,
                    validation_data=(X_v, Y_v),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)    

    

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    t = f.suptitle('Basic CNN Performance', fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)

    epoch_list = list(range(1,31))
    ax1.plot(epoch_list, history.history['acc'], label='Train Accuracy')
    ax1.plot(epoch_list, history.history['val_acc'], label='Validation Accuracy')
    ax1.set_xticks(np.arange(0, 31, 5))
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")

    ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
    ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
    ax2.set_xticks(np.arange(0, 31, 5))
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")

    f.savefig("accu.png")

    model.save("cifar10.h5")
