""" Vanilla Teacher Student Net
"""
from keras.datasets import mnist
from keras.layers import *
from keras import Model, utils
from sklearn.metrics import accuracy_score
import numpy as np


from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)


# >> Teacher net: 3-layer CNN model
def teacher_net():
    input_ = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), padding="same")(input_)
    x = Activation("relu")(x)
    print(x)
    x = MaxPool2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPool2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPool2D((2, 2))(x)
    x = Flatten()(x)
    out = Dense(10, activation="softmax")(x)
    model = Model(inputs=input_, outputs=out)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    model.summary()
    return model


# >> Student net: one 64 FC layer.
def student_net():
    input_ = Input(shape=(28, 28, 1))
    x = Flatten()(input_)
    x = Dense(64, activation="sigmoid")(x)  # 512
    out = Dense(10, activation="softmax")(x)
    model = Model(inputs=input_, outputs=out)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    model.summary()
    return model


def teacher_student_net(teacher_out, student_model, data_train, data_test, label_test):
    print('\nModel: Teacher-Student net')
    t_out = teacher_out

    s_model = student_model
    for l in s_model.layers:
        l.trainable = True

    label_test = utils.to_categorical(label_test)

    model = Model(s_model.input, s_model.output)
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam")
    model.fit(data_train, t_out, batch_size=64, epochs=10)

    s_predict = np.argmax(model.predict(data_test), axis=1)
    s_label = np.argmax(label_test, axis=1)
    print('teacher-student net acc: ', accuracy_score(s_predict, s_label))


# >> Load data
(data_train, label_train), (data_test, label_test) = mnist.load_data()
data_train = np.expand_dims(data_train, axis=3)
data_test = np.expand_dims(data_test, axis=3)

# >> Teacher net
t_model = teacher_net()
t_model.fit(data_train, label_train, batch_size=64, epochs=2, validation_data=(data_test, label_test))

# >> Student net
s_model = student_net()
s_model.fit(data_train, label_train, batch_size=64, epochs=10, validation_data=(data_test, label_test))

# >> Teacher-Student net
t_out = t_model.predict(data_train)
teacher_student_net(t_out, student_net(), data_train, data_test, label_test)  # no pretrain on Student net
# teacher_student_net(t_out, s_model, data_train, data_test, label_test)  #Pretrained Student net
