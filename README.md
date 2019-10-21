# OneFourthLabs-Internship-Task
One Fourth Labs Internship Interview Round 1 -- Programming Assessment

For solving this problem, I had used [TensorFlow](https://www.tensorflow.org) A Machine Learning Library.

I had attached my step-by-step jupyter notebook solution for this problem [here](https://github.com/MALLI7622/OneFourthLabs-Internship-Task/blob/master/oflinterntask.ipynb). 

Below is the Deep Learning Model, I created for solving this problem

After applying different batch sizes, neural networks, and epochs. I had got the following results in trainig
##### model

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(784, activation = 'relu'),
    keras.layers.Dense(256, activation = 'relu'),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dense(47, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

##### Training


model.fit(train_images, labels, epochs = 10)

##### Accuracies in each eopch


Train on 112800 samples
Epoch 1/10
112800/112800 [==============================] - 33s 291us/sample - loss: 0.8618 - accuracy: 0.7323
Epoch 2/10
112800/112800 [==============================] - 32s 286us/sample - loss: 0.5216 - accuracy: 0.8219
Epoch 3/10
112800/112800 [==============================] - 32s 282us/sample - loss: 0.4448 - accuracy: 0.8451
Epoch 4/10
112800/112800 [==============================] - 33s 288us/sample - loss: 0.4004 - accuracy: 0.8579
Epoch 5/10
112800/112800 [==============================] - 32s 287us/sample - loss: 0.3679 - accuracy: 0.8663
Epoch 6/10
112800/112800 [==============================] - 32s 283us/sample - loss: 0.3422 - accuracy: 0.8741
Epoch 7/10
112800/112800 [==============================] - 31s 279us/sample - loss: 0.3236 - accuracy: 0.8806
Epoch 8/10
112800/112800 [==============================] - 32s 280us/sample - loss: 0.3064 - accuracy: 0.8859
Epoch 9/10
112800/112800 [==============================] - 32s 285us/sample - loss: 0.2940 - accuracy: 0.8889
Epoch 10/10
112800/112800 [==============================] - 34s 302us/sample - loss: 0.2811 - accuracy: 0.8922


##### Testing Accuracy

18800/1 - 3s - loss: 122.0796 - accuracy: 0.8414

Test accuracy: 0.841383





