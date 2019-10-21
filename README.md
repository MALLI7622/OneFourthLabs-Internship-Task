# OneFourthLabs-Internship-Task
One Fourth Labs Internship Interview Round 1 -- Programming Assessment

For solving this problem, I had used [TensorFlow](https://www.tensorflow.org) A Machine Learning Library developed by Google.

I had attached my step-by-step jupyter notebook solution for this problem [here](https://github.com/MALLI7622/OneFourthLabs-Internship-Task/blob/master/oflinterntask_model6.ipynb). 

I had applied different Deep Learning models mentioned below:

##### 1 ) input : 784 ---> hidden : 256 ----> output : 47  

Training Accuracy at last epoch is:


Epoch 10/10

112800/112800 [==============================] - 34s 302us/sample - loss: 0.2811 - accuracy: 0.8922

Test accuracy: 0.84271276




##### 2 ) input : 784 ---> hidden_1 : 256 -----> hidden_2 : 128 ----> output : 47 

Training Accuracy at the Last epoch: 

Epoch 10/10
112800/112800 [==============================] - 35s 309us/sample - loss: 0.2643 - accuracy: 0.8976

Test accuracy: 0.8381915


##### 3 ) input : 784 ---> hidden_1 : 256 -----> hidden_2 : 64 ----> output : 47 

Training Accuracy at the Last epoch: 

Epoch 10/10
112800/112800 [==============================] - 30s 264us/sample - loss: 0.2579 - accuracy: 0.8994

Test accuracy: 0.8376596


##### 4 ) input : 784 ---> hidden_1 : 128 -----> hidden_2 : 64 ----> output : 47 

Training Accuracy at the Last epoch: 

Epoch 10/10
112800/112800 [==============================] - 31s 276us/sample - loss: 0.2651 - accuracy: 0.8973


Test accuracy: 0.8424468


##### 5 ) input : 784 ---> hidden_1 : 256 -----> hidden_2 : 128 ----> hidden_3 : 64 ----> output : 47


Training Accuracy at last epoch is:


Epoch 10/10
112800/112800 [==============================] - 34s 302us/sample - loss: 0.2811 - accuracy: 0.8922


Test accuracy: 0.841383


##### 6 ) input : 784 ---> hidden_1 : 256 -----> hidden_2 : 128 ----> hidden_3 : 64 ----> output : 47 , batch_size = 32

Training Accuracy at last epoch is:

Epoch 10/10
112800/112800 [==============================] - 34s 305us/sample - loss: 0.2796 - accuracy: 0.8936

Test accuracy: 0.845

##### 7 ) input : 784 ---> hidden : 256 ----> output : 47  

Epoch 10/10
112800/112800 [==============================] - 32s 287us/sample - loss: 0.2337 - accuracy: 0.9070

Test accuracy: 0.8383511

I had explained my solution to the problem step-by-step in the [oflinterntask_model6.ipynb](https://github.com/MALLI7622/OneFourthLabs-Internship-Task/blob/master/oflinterntask_model6.ipynb). I had uploaded it to the GitHub repository

#### Instructions to run the code

I had also uploaded [oflinterntask_model6.py](https://github.com/MALLI7622/OneFourthLabs-Internship-Task/blob/master/oflinterntask_model6.py) file. You can download it and run it in your own system.

If you have tensorflow environment in your system. You can activate by the following command
$ conda activate tensorflow

Go to the Directory where your file was located 
$ cd [Directory name]

Now run the file

$ python oflinterntask.py

#### or 

You can download [oflinterntask_model6.ipynb](https://github.com/MALLI7622/OneFourthLabs-Internship-Task/blob/master/oflinterntask_model6.ipynb) file run each cell in your jupyter notebook for better understanding.


## Thank You

