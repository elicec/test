from __future__ import absolute_import, division,print_function
import os
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

(train_images, train_labels),(test_images,test_labels)=tf.keras.datasets.mnist.load_data()
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1,28*28)/255.0
test_images = test_images[:1000].reshape(-1,28*28)/255.0

def create_mode():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512,activation=tf.keras.activations.relu,input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10,activation=tf.keras.activations.softmax)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    return model

model = create_mode()
model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=5)
model.save('my_model.h5')

new_model = keras.models.load_model('my_model.h5')
new_model.summary()

loss,acc = new_model.evaluate(test_images,test_labels)
print("Restored model,accuracy:{:5.6f}%".format(100*acc))
