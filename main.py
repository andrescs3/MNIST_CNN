from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
import pandas as pd
import random 
from keras.models import load_model
import json
import csv


def GenerateModel(sample, fixed_items):
    (x_train, y_train), (x_test, y_test ) = mnist.load_data()

    batch_size = 128
    num_classes = 10
    epochs = 12
    size_from = 0
    size_to = sample 
    size_test = 10000

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)



    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')


    size = size_to-size_from

    if fixed_items:



        y_train = y_train[size_from:size_to]
        y_train=  y_train

        x_train = x_train[size_from:size_to]
        x_train = x_train.reshape(size, 28,28, 1)
    else:


        indexes = random.sample(range(60000), size_to)
        index = 0
        result = []
        result=np.array(result, dtype='uint8')


        result2 = []
        result2=np.array(result2, dtype='uint8')
        result2=result2.reshape(0, 28,28, 1)
        k=0
        for i in indexes:
            temp = y_train[i:i+1]
            result = np.append(result,  temp, axis=0)

            temp2 = x_train[i:i+1]
            result2 = np.append(result2,  temp2, axis=0)            
            



        
        y_train = result
        #y_train = random.sample(y_train,size_to)
        y_train=  y_train

        x_train = result2
        x_train = x_train.reshape(size, 28,28, 1)
    

    

    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

    type = "random"
    if fixed_items:
        type = "fixed"

    filename = type+""+str(sample)
    model.save(filename+".h5")
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("accuracy"+filename)
    plt.clf()

    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("loss"+filename)
    plt.clf()
    plt.close()

    



def EvaluateModel(sample, fixed_items, test_size, fixed_training):
    (x_train, y_train), (x_test, y_test ) = mnist.load_data()

    batch_size = 128
    num_classes = 10
    epochs = 12
    size_from = 0
    size_to = sample 
    size_test = 10000

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)



    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')


    size = size_to-size_from

    if fixed_items:



        y_test = y_test[size_from:size_to]
        y_test=  y_test
        y_true=y_test

        x_test = x_test[size_from:size_to]
        x_test = x_test.reshape(size, 28,28, 1)
    else:
        


        indexes = random.sample(range(10000), size_to)
        index = 0
        result = []
        result=np.array(result, dtype='uint8')


        result2 = []
        result2=np.array(result2, dtype='uint8')
        result2=result2.reshape(0, 28,28, 1)
        k=0
        for i in indexes:
            temp = y_test[i:i+1]
            result = np.append(result,  temp, axis=0)

            temp2 = x_test[i:i+1]
            result2 = np.append(result2,  temp2, axis=0)            
            



        
        y_test = result
        #y_train = random.sample(y_train,size_to)
        y_test=  y_test
        y_true=y_test

        x_test = result2
        x_test = x_test.reshape(size, 28,28, 1)
    

    

    x_train /= 255
    x_test /= 255


    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    training_type = "random"
    if fixed_training:
        training_type = "fixed"

    type = "random"
    if fixed_items:
        type = "fixed"
    

    model_name = training_type+""+str(test_size)+".h5"
    print("Name")
    print(model_name)
    # load model
    model = load_model(model_name)
    # summarize model.
    #model.summary()

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Train Size:',test_size,'--Test size:',sample,'--Test accuracy:', score[1],'--Test loss:', score[0] )

    classes=[0,1,2,3,4,5,6,7,8,9]

    y_pred=model.predict_classes(x_test)
    con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    
    con_mat_df = pd.DataFrame(con_mat_norm,
                        index = classes, 
                        columns = classes)

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    file_csv = "Testing/"+type+"_"+training_type+str(test_size)+"_"+str(sample)
    print (file_csv)
    plt.savefig(file_csv)             
    plt.clf()
    plt.close()

    with open('Testing/Testing_'+type+'_'+training_type+'.csv', mode='a+', newline='') as testing_file:
        testing_writter = csv.writer(testing_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        testing_writter.writerow([test_size, sample, score[1], score[0]])        

for i in range(10000,70000,10000):
    GenerateModel(i, True)

for i in range(10000,70000,10000):
    for j in range(1000,11000,1000):
        EvaluateModel(j,False,i, True) 

for i in range(10000,70000,10000):
    for j in range(1000,11000,1000):
        EvaluateModel(j,True,i, True)

for i in range(10000,70000,10000):
    GenerateModel(i, False)

for i in range(10000,70000,10000):
    for j in range(1000,11000,1000):
        EvaluateModel(j,False,i, False)

for i in range(10000,70000,10000):
    for j in range(1000,11000,1000):
        EvaluateModel(j,True,i, False)