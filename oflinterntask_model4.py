#!/usr/bin/env python
# coding: utf-8

# # One Fourth Labs Internship Interview Round 1 -- Programming Assessment
#

# ###### Importing libraries

# In[1]:


from zipfile import ZipFile
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from collections import OrderedDict


# ###### Extracting Dataset

# In[2]:


data = "Character + Digits data.zip"
with ZipFile(data, 'r') as zip:
    zip.printdir()
    print(" Extracting all the files now...")
    zip.extractall()
    print("Done...!")


# ###### Extracting training data

# In[3]:


train_data = "Character + Digits data/characters-digits-train.zip"


# In[4]:


with ZipFile(train_data, 'r') as zip:
    zip.printdir()
    print(" Extracting all the files now...")
    zip.extractall()
    print("Done...!")


# In[5]:


# Reading trainied data file
train_csv_data = pd.read_csv("characters-digits-train.csv")


# In[6]:


# Let's see the training data
#train_csv_data


# In[7]:


# In pandas DataFrame first row of data named as columns names of Dataframe. So, I had made a first row of traine_csv_data file
# to the duplicate_data.csv file and then append to the train_data
duplicate_data = pd.read_csv("duplicate_data .csv")


# In[8]:


## Changing column names in both trained_csv_data and duplicate_data then only we have to append otherwise data will be noisy
for i in range(785):
        duplicate_data.rename( columns = {duplicate_data.columns[i]: i},inplace = True )
        train_csv_data.rename( columns = { train_csv_data.columns[i]:i}, inplace = True)
duplicate_data.rename( columns = {duplicate_data.columns[0]: "Label_ID" }, inplace = True)
train_csv_data.rename( columns = {train_csv_data.columns[0]: "Label_ID" }, inplace = True)


# In[9]:


#Let's see train_csv_data now
#train_csv_data


# In[10]:


#Let's see duplicate_data
#duplicate_data


# In[11]:


#appending dataframes to train_data Dataframe
train_data = duplicate_data.append(train_csv_data)


# In[12]:


#train_data


# In[13]:


# Now let's see the length of the train_data
#len(train_data)


# In[14]:


# For training we have to seperate Label_ID's and dataset. For that one I assigned to labels. Later I will drop
# train_data["Label_ID"] from train_data
labels = train_data["Label_ID"]


# In[15]:


#labels


# In[16]:


# Now I'm removing train_data["Label_ID"] from train_data
train_data = train_data.drop(["Label_ID"], axis = 1)


# In[17]:


# For training a model we don't has to use DataFrame for that one I'm converting to np.array
train_data = np.array(train_data)


# In[18]:


# Let's see the shape of the array
#train_data.shape


# In[19]:


# For viewing the image we have convert it to 28x28 image size
train = np.reshape(train_data, (train_data.shape[0], 28,28))


# In[20]:


# Now shape of the train is
#train.shape


# In[21]:


# Also converting Label data also to np.array
labels = np.array(labels)


# In[22]:


# Label Data
#labels


# In[23]:


# Now lets see the first image of our data
plt.figure()
plt.imshow(train[0])
plt.xlabel(labels[0])
plt.colorbar()
plt.grid(False)
plt.show()


# In[24]:


# Scaling these values between 0 to 1. So, divide values by 255
train_images = train / 255


# In[25]:


# Let's see the first 25 images and their labels
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(labels[i])
plt.show()


# ### Building the neural network model
#
# I had used TensorFlow and keras as backend for building neural network model

# In[26]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(784, activation = 'relu'),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dense(47, activation='softmax')
])


# In[27]:


## Now I'm compiling my model with optimizer, loss and metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[28]:


## Training the model
model.fit(train_images, labels, epochs = 10)


# ## Testing the Deep Learning model

# In[29]:


## Reading test data from characters-digits-test.csv
test_csv_data = pd.read_csv("characters-digits-test.csv")


# In[30]:


## Now let's see test_csv_data
#test_csv_data


# In[31]:


# As mentioned above the first row is named as column names
test_duplicate = pd.read_csv("test_duplicate.csv")


# In[32]:


# Let's see the test_duplicate
#test_duplicate


# In[33]:


# Changing column names both in test_csv_data and test_duplicate
for i in range(785):
        test_duplicate.rename( columns = {test_duplicate.columns[i]: i},inplace = True )
        test_csv_data.rename( columns = { test_csv_data.columns[i]:i}, inplace = True)
test_duplicate.rename( columns = {test_duplicate.columns[0]: "Label_ID" }, inplace = True)
test_csv_data.rename( columns = {test_csv_data.columns[0]: "Label_ID" }, inplace = True)


# In[34]:


# Let's see the test_dulpicate
#test_duplicate


# In[35]:


# Let's see test_csv_data
#test_csv_data


# In[36]:


# Now appending to test_data
test_data = test_duplicate.append(test_csv_data)


# In[37]:


# Let's see the test_data
#test_data


# In[38]:


# Getting test labels from test_data
test_label = test_data["Label_ID"]


# In[39]:


# Dropping test_data["Label_ID"] from test_data
test_data = test_data.drop(["Label_ID"], axis = 1)


# In[40]:


# Converting numpy array, because test_data is a DataFrame
test_data = np.array(test_data)


# In[41]:


# Converting numpy array, because test_label is a DataFrame
test_label = np.array(test_label)


# In[42]:


# Now let's see test_data type and shape
type(test_data),test_data.shape


# In[43]:


# Now reshaping to 18800x28x28
test_data = np.reshape(test_data,(test_data.shape[0],28,28))


# ###### Now test data is ready for testing the model

# In[44]:


# Finding accuracy of the model
test_loss, test_acc = model.evaluate(test_data,  test_label, verbose=2)

print('\nTest accuracy:', test_acc)


# In[45]:


# Finally predcting the model with test_data
predictions = model.predict(test_data)


# In[46]:


# Let's see predictions[1] data
#predictions[1]


# ###### np.argmax returns predicted label of test_dat. So, we have to convert it to either digits or letter using "chr" function

# In[47]:


# Predictions returns the numpy array. So, I had used np.argmax function returns maximum value along the axis
np.argmax(predictions[1])


# In[49]:


## Reading characters-digits-mapping.txt file and assigning it to labels dictionary
with open('characters-digits-mapping.txt', 'r') as file:
    labels = {}
    for line in file:
        line = line.split()
        if not line:
            continue
        labels[line[0]] = line[1:]


# In[52]:


# Let's see one of predicted data out of 18800
#d = chr(int(labels[str(np.argmax(predictions[random.randint(1,18800)]))][0]))


# In[53]:


#d


# In[54]:


# I had created digit and vowel list for distinguishing
digit = []
for i in range(48,58):
    digit.append(i)
vowel = ['a','e','i','o','u','A','E','I','U']


# ## Logic for Task 1, Task 2 and Task 3

# ###### By running the below cell repeatedly my model will predict
# ###### Task 1 : Letter / Digit
# ###### Task 2 : Even / Odd and Vowel / Consonant
# ###### Task3 : Character Classifier

# In[55]:


i = random.randint(1,18800)
predicted_data = np.argmax(predictions[i])
print("================================ TASK 1 ----------- Letter or Digit ------------ ================================ \n\n")
if int(labels[str(predicted_data)][0]) in digit:
    print("The Given image is -------> is Digit \n\n\n")

else:
    print("The Given image is -------> is a Letter \n\n\n")
print(" --------------------------------------////////\\\\\\\\\-------------------------------------------- \n\n\n\n")



print("=================== TASK 2 --------- Vowel or Consonant and Even or Odd Classifier --------- ===================\n\n ")

if int(labels[str(predicted_data)][0]) in digit:
    if int(chr(int(labels[str(predicted_data)][0]))) % 2 == 0:
        print("The Given image is ------->  is a Digit and it's a even number \n\n\n")

    else:
        print("The Given image is -------> is a Digit and it's a odd number \n\n\n")

elif chr(int(labels[str(predicted_data)][0])) in vowel:
    print("The Given image is -------> is a Letter and is a Vowel \n\n\n")

else:
    print("The Given image is ------->", chr(int(labels[str(predicted_data)][0])), "is a Letter and is a Consonant \n\n\n")

print(" --------------------------------------////////\\\\\\\\\-------------------------------------------- \n\n\n\n")


print("================================== Task 3 ------ Character Classifier ------ ==================================== \n\n")

if int(labels[str(predicted_data)][0]) in digit:
    if int(chr(int(labels[str(predicted_data)][0]))) % 2 == 0:
        print("The Given image is -------> ",chr(int(labels[str(predicted_data)][0]))," is a Digit and it's a even number \n\n")
        plt.imshow(test_data[i])
    else:
        print("The Given image is -------> ", chr(int(labels[str(predicted_data)][0]))," is a Digit and it's a odd number \n\n")
        plt.imshow(test_data[i])
elif chr(int(labels[str(predicted_data)][0])) in vowel:
    print("The Given image is ------->  ", chr(int(labels[str(predicted_data)][0])),"    is a Letter and is a Vowel \n\n")
    plt.imshow(test_data[i])
else:
    print("The Given image is ------->  ", chr(int(labels[str(predicted_data)][0])), "     is a Letter and is a Consonant \n\n")
    plt.imshow(test_data[i])


# ###### Task: Class-wise accuracy

# In[56]:


## Appending test_duplicate and test_csv_data to test_dupl for class-wise accuracy
test_dupl = test_duplicate.append(test_csv_data)
## Getting test_dupl
test_dup = test_dupl["Label_ID"]
## Converting to list
test_list = list(test_dup)
## Removing duplicates
li = list(OrderedDict.fromkeys(test_list))
## Sorting for good understanding
li.sort()


# In[62]:


### Calculating class-wise accuracy

for i in li:
    test_i = test_dupl[test_dupl.Label_ID == i]
    test_i_label = test_i["Label_ID"]
    test_i = test_i.drop(["Label_ID"], axis = 1)
    test_i = np.array(test_i)
    test_i_label = np.array(test_i_label)
    test_i = np.reshape( test_i,(len(test_i),28,28))

    test_loss_i, test_acc_i = model.evaluate(test_i,  test_i_label, verbose=2)

    print('\nTest accuracy of Label  ------>',chr(int(labels[str(i)][0])),"is", test_acc_i,"\n\n")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
