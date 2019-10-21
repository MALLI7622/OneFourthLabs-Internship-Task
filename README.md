# OneFourthLabs-Internship-Task
One Fourth Labs Internship Interview Round 1 -- Programming Assessment

For solving this problem, I had used [TensorFlow](https://www.tensorflow.org) A Machine Learning Library developed by Google.

I had attached my step-by-step jupyter notebook solution for this problem [here](https://github.com/MALLI7622/OneFourthLabs-Internship-Task/blob/master/oflinterntask.ipynb). 

I had applied different Deep Learning models mentioned below:

1 ) input : 784 ---> hidden : 256 ----> output : 47  

2 ) input : 784 ---> hidden_1 : 256 -----> hidden_2 : 128 ----> output : 47

3 ) input : 784 ---> hidden_1 : 256 -----> hidden_2 : 64 ----> output : 47

4 ) input : 784 ---> hidden_1 : 256 -----> hidden_2 : 64 ----> output : 47

5 ) input : 784 ---> hidden_1 : 256 -----> hidden_2 : 128 ----> hidden_3 : 64 ----> output : 47

After applying these models I found  

input : 784 ---> hidden_1 : 256 -----> hidden_2 : 128 ----> hidden_3 : 64 ----> output : 47

this network produces better accuracy results. Below I had mentioned my better accuaracy and loss results. 

##### Training Accuracy

Training Accuracy at last epoch is:


Epoch 10/10
112800/112800 [==============================] - 34s 302us/sample - loss: 0.2811 - accuracy: 0.8922

##### Testing Accuracy

18800/1 - 3s - loss: 122.0796 - accuracy: 0.8414

Test accuracy: 0.841383

I had explained my solution to the problem step-by-step in the [oflinterntask.ipynb](https://github.com/MALLI7622/OneFourthLabs-Internship-Task/blob/master/oflinterntask.ipynb). I had uploaded it to the GitHub repository

##### Instructions to run the code

I had also uploaded [oflinterntask.py](https://github.com/MALLI7622/OneFourthLabs-Internship-Task/blob/master/oflinterntask.py) file. You can download it and run it in your own system.

If you have tensorflow environment in your system. You can activate by the following command
$ conda activate tensorflow

Go to the Directory where your file was located 
$ cd [Directory name]

Now run the file

$ python oflinterntask.py

###### or 
You can download [oflinterntask.ipynb](https://github.com/MALLI7622/OneFourthLabs-Internship-Task/blob/master/oflinterntask.ipynb) file run each cell in your jupyter notebook for better understanding.


## Thank You

