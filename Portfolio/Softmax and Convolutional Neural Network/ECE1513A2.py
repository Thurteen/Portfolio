import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import string
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# ################################PREAMBLE######################################
# This script was written cooperatively by Dylan Johnston (1003852690) And Omar Ismail (999467660).
# It optimizes the notMNIST dataset using two methods: a conventional Neural Net using Numpy and a powerful Convolutional Neural Network with TensorFlow.
# When the script is run in the interpreter, the output will be the training and test accuracy and loss for each method, along with the time required for the computation.
# Graphs of the accuracy and loss  are created and displayed. Variables can be changed in the function globalVariableDefine.



def neuralNumpy():
####################################################################################
#                 SUMMARY OF FINAL DATA (NEURAL NUMPY)                             #
# LAYER   TIME(s)   TRAIN LOSS/ACCURACY     VAL LOSS/ACCURACY  TEST LOSS/ACCURACY  #
# 100      27.56       0.142640/96.21       0.362000/89.83      0.388943/89.90   #
# 500     129.44       0.129509/96.72       0.354060/90.37      0.383086/90.05   #
# 1000    232.41       0.122815/96.93       0.345485/90.32      0.377778/90.35   #
# 2000    426.98       0.121036/96.82       0.338872/90.67      0.367983/90.38   #
####################################################################################

    start1 = time.time()
    W1, W2, v1, v2 = globalVariableDefine()

    fig, (ax1, ax2) = plt.subplots(2, 1)

    for i in range(Numpy_Epochs):
        # Calculate and Store Training, Validation, and Test Error and Accuracy
        x2, x1, s1, x0, W2 = forwardPass(trainData, W1, W2)
        store_training_error[i], store_training_accuracy[i] = CE(np.transpose(trainTarget), x2)

        x2Trash, x1Trash, s1Trash, x0Trash, W2Trash = forwardPass(validData, W1, W2)
        store_valid_error[i], store_valid_accuracy[i] = CE(np.transpose(validTarget), x2Trash)

        x2Trash, x1Trash, s1Trash, x0Trash, W2Trash = forwardPass(testData, W1, W2)
        store_test_error[i], store_test_accuracy[i] = CE(np.transpose(testTarget), x2Trash)

        # Calculate new gradients using backwards propogation
        W1grad, W2grad = backwardPass(trainTarget, x2, x1, s1, x0, W2)

        # Update bias terms. This is done wihout momentum as the problem set had the update rule for the weights only
        W1[0,:] = W1[0,:] - learn_rate*W1grad[0,:]
        W2[0,:] = W2[0,:] - learn_rate*W2grad[0,:]

        # Update momentum term
        v1 = gamma*v1 + learn_rate*W1grad[1:,:]
        v2 = gamma*v2 + learn_rate*W2grad[1:,:]

        # Update weights
        W1[1:,:] = W1[1:,:] - v1
        W2[1:,:] = W2[1:,:] - v2


        # Live update of accuracy and loss graphs
        ax1.cla()
        ax1.plot(store_training_error[0:i])
        ax1.plot(store_valid_error[0:i])
        ax1.plot(store_test_error[0:i])

        ax2.cla()
        ax2.plot(store_training_accuracy[0:i])
        ax2.plot(store_valid_accuracy[0:i])
        ax2.plot(store_test_accuracy[0:i])
        plt.pause(0.05)

    Loss_and_Acc_Numpy = np.array([store_training_error, store_valid_error, store_test_error, store_training_accuracy, store_valid_accuracy, store_test_accuracy])
    print('Time Taken to Train, Validate, and Test Numpy Neural = ', np.around(time.time() - start1,2))

    return Loss_and_Acc_Numpy

def forwardPass(x, W1,W2):
    ones = np.ones((x.shape[0],1))
    x0 = np.append(ones,x, axis = 1)
    x0 = np.transpose(x0)  #Now this is a 785 by N matrix
    s1 = computeLayer(x0,W1) #This is a 1000 by N matrix
    x1 = relu(s1)
    x1 =np.append(np.ones((1,x1.shape[1])),x1,axis=0)
    s2 = computeLayer(x1,W2)
    x2 = softmax(s2) #probability labels
    return x2, x1, s1, x0, W2

def backwardPass(y, x2, x1, s1, x0, W2):
    y = np.transpose(y)
    delta2 = gradCE(y,x2)
    gradW2 = np.matmul(delta2,np.transpose(x1))
    gradW2 = np.transpose(gradW2)
    delta1 = np.multiply(np.matmul(W2[1:,:],delta2),reluPrime(s1))
    gradW1 = np.matmul(delta1,np.transpose(x0))
    gradW1 = np.transpose(gradW1)
    return gradW1, gradW2

def relu(x):
   return np.maximum(x, 0, x)

def softmax(x):
    return (np.exp(x) / np.sum(np.exp(x), axis=0))

def computeLayer(X, W):
   return np.matmul(np.transpose(W),X)

def CE(target, prediction):
    accuracy = np.mean(np.argmax(target,axis = 0) ==  np.argmax(prediction,axis=0))*100
    loss =  np.multiply(-1/target.shape[1],np.trace(np.matmul(np.log(prediction),np.transpose(target))))
    return loss, accuracy

def gradCE(target, prediction):
    return np.subtract(prediction,target)

def reluPrime(x):
     x[x<=0] = 0
     x[x>0] = 1
     return x


def playGame(x,y, W1, W2):  # For Neural Net Numpy only! Send the data, the labels, and W1, and W2 and it will make predictions for you!
    dictionary = dict(enumerate(string.ascii_uppercase, 0))
    for i in range(x.shape[0]):
        x2, x1, s1, x0, W2 = forwardPass(x[i,None], W1,W2)
        predict = np.asscalar(np.argmax(x2,axis=0))
        plt.imshow(np.array(x[i]).reshape(28,28), cmap = 'gray') # Show each row as a picture
        plt.show()
        print("The system has predicted ", dictionary[predict])
        input() # Press anything so you can see the next picture


def globalVariableDefine():
    #Define for Numpy Section
    global trainData, validData, testData, trainTarget, validTarget, testTarget, W1, W2, v1, v2, Numpy_Epochs, gamma, learn_rate, store_training_error, store_valid_error, store_test_error, store_training_accuracy, store_valid_accuracy, store_test_accuracy
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    Numpy_Epochs = 200
    layer0Units = 784
    layer1Units = 1000
    layer2Units = 10
    W1 = np.random.normal(0,2/(layer0Units+layer1Units),(layer0Units+1,layer1Units))
    W2 = np.random.normal(0,2/(layer1Units+layer2Units),(layer1Units+1,layer2Units))
    v1 = np.full((layer0Units,layer1Units),1e-5)
    v2 = np.full((layer1Units,layer2Units),1e-5)
    gamma = 0.95
    learn_rate = 1e-5
    store_training_error = np.zeros((Numpy_Epochs,1))
    store_valid_error= np.zeros((Numpy_Epochs,1))
    store_test_error= np.zeros((Numpy_Epochs,1))
    store_training_accuracy = np.zeros((Numpy_Epochs,1))
    store_valid_accuracy = np.zeros((Numpy_Epochs,1))
    store_test_accuracy = np.zeros((Numpy_Epochs,1))


    #Define for TensorFlow Section
    global regulaizer, batch_size, trainData4D,testData4D, validData4D, alpha, SGD_training_epochs, keep_rate
    trainData4D = np.reshape(trainData, (-1,28,28,1))
    testData4D = np.reshape(testData, (-1,28,28,1))
    validData4D = np.reshape(validData, (-1,28,28,1))
    regulaizer = 0.0 #[0.01,0.1,0.5]
    keep_rate = 1 #[0.9=overfit 0.75=overfit 0.5=overfit]
    batch_size = 32
    SGD_training_epochs = 50
    alpha = 1e-4

    return W1, W2, v1, v2

def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]

        trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)
    return reshapeData(trainData), reshapeData(validData), reshapeData(testData),trainTarget, validTarget, testTarget

def reshapeData(dataset):
    img_h = img_w = 28             # MNIST images are 28x28
    global img_size_flat
    img_size_flat = img_h * img_w  # 28x28=784, the total number of pixels
    dataset = dataset.reshape((-1,img_size_flat)).astype(np.float64)
    return dataset

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest

def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target

def buildGraph():
    globalVariableDefine()
    tf.reset_default_graph()
    tf.set_random_seed(421)

    weights = {
    'kernel': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
    'w1': tf.get_variable('W1', shape=( 28*28*32,784), initializer=tf.contrib.layers.xavier_initializer()),
    'w2': tf.get_variable('W2', shape=(784,10), initializer=tf.contrib.layers.xavier_initializer()),
}
    biases = {
    'b0': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'b1': tf.get_variable('B1', shape=(784), initializer=tf.contrib.layers.xavier_initializer()),
    'b2': tf.get_variable('B2', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
}

    labels = tf.placeholder(shape=(None, 10), dtype='int32')
    reg = tf.placeholder(tf.float32,None, name='regulaizer')
    keeper = tf.placeholder(tf.float32, shape=(), name="keeper")

    # Step 1 - Input Layer
    trainingInput = tf.placeholder(shape=(None, 28, 28,1), dtype='float32')

    # Step 2 - Convolutional Layer
    conv1 = tf.nn.conv2d(trainingInput, weights['kernel'], strides=[1, 1, 1, 1], padding="SAME")
    conv1 = tf.nn.bias_add(conv1, biases['b0'])

    # Step 3 - ReLU Activation
    x = tf.nn.relu(conv1)

    # Step 4 - Normalization layer.....which axes do I get the mean from?for so-called "global normalization", used with convolutional filters with shape [batch, height, width, depth], pass axes=[0, 1, 2]. for simple batch normalization pass axes=[0]
    mean, variance = tf.nn.moments(x, [0])
    x_norm = tf.nn.batch_normalization(x,mean,variance,None,None,1e-20)

    # Step 5 - Pooling Layer
    pool1 = tf.nn.max_pool(x_norm, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME")

    # Step 6 - Flatten layer
    pool1_flat = tf.reshape(pool1, [-1,28*28*32])

    # Step 7 - Fully Connected Layer
    layer_784 = tf.nn.bias_add(tf.matmul(pool1_flat, weights['w1']), biases['b1'])

    # Step 8 - ReLU Activation
    dropLayer = tf.nn.relu(tf.nn.dropout(layer_784, rate = keeper))

    # Step 9 - Fully Connected Layer
    logits = tf.nn.bias_add(tf.matmul(dropLayer, weights['w2']), biases['b2'])

    # Step 10 - SoftMax Output
    outputClass = tf.argmax(tf.nn.softmax(logits), axis=1)

    # Step 11 - Cross Entropy Loss....Do we need to L2 the Biases?
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)) +reg*(tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2']))

    # Step 12 - Calculate Prediction Accuracy
    correct_prediction = tf.equal(outputClass, tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))*100

    # Step 13 - Define Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = alpha).minimize(loss) #variables defined above and changed through console

    return optimizer, loss, trainingInput, labels, reg, accuracy, keeper

def CNN():
####################################################################################
#                SUMMARY OF FINAL DATA (CNN - NO VARYING)                          #
#         TIME(s)   TRAIN LOSS/ACCURACY   VAL LOSS/ACCURACY  TEST LOSS/ACCURACY    #
#         116.65      0.098641/96.88       0.973543/92.10       1.003803/91.18     #
####################################################################################
#                SUMMARY OF FINAL DATA (CNN - VARY LAMBDA)                         #
#  L      TIME(s)   TRAIN LOSS/ACCURACY   VAL LOSS/ACCURACY  TEST LOSS/ACCURACY    #
# 0.01    115.83      0.205066/96.88       0.432190/92.68       0.466788/91.92     #
# 0.1     118.22      0.487105/90.65       0.503893/92.10       0.512228/92.18     #
# 0.5     115.58      0.902568/84.34       0.835178/90.28       0.835641/90.35     #
####################################################################################
#                SUMMARY OF FINAL DATA (CNN - VARY KEEP RATE)                      #
#  P      TIME(s)   TRAIN LOSS/ACCURACY   VAL LOSS/ACCURACY  TEST LOSS/ACCURACY    #
# 0.5     115.76      0.012504/100         0.759497/93.13       0.766280/92.69     #
# 0.75    117.73      0.021895/100         0.871609/92.98       0.925613/92.37     #
# 0.9     116.19      0.056570/96.88       0.951308/92.27       0.970033/92.00     #
####################################################################################

    start1 = time.time()
    optimizer, loss, trainingInput, labels, reg, accuracy, keeper = buildGraph()

    #-----------------------Initialize Storage Variables-----------------------#
    store_training_error = np.zeros((SGD_training_epochs,1))
    store_valid_error= np.zeros((SGD_training_epochs,1))
    store_test_error= np.zeros((SGD_training_epochs,1))
    store_training_accuracy = np.zeros((SGD_training_epochs,1))
    store_valid_accuracy = np.zeros((SGD_training_epochs,1))
    store_test_accuracy = np.zeros((SGD_training_epochs,1))

    init = tf.global_variables_initializer()   # Initialize session
    fig, (ax1, ax2) = plt.subplots(2, 1)

    with tf.Session() as sess:
        sess.run(init)
        batch_number = int(trainTarget.shape[0]/batch_size) # Calculate batch number

        for i in range(SGD_training_epochs): # Loop across SGD_training_epochs
            batch_index = np.random.permutation(trainData4D.shape[0]) # Reshuffle training data
            X_split = np.array_split(trainData4D[batch_index],batch_number) # Split into the number of batches
            Y_split = np.array_split(trainTarget[batch_index],batch_number) # Split into the number of batches

            for j in range(len(X_split)): # Loop through each batch
                _, store_training_error[i], store_training_accuracy[i] = sess.run([optimizer,loss,accuracy], feed_dict = {trainingInput: X_split[j], labels: Y_split[j], reg: regulaizer, keeper: (1-keep_rate)}) # Let us OPTIMIZE!
            store_valid_error[i], store_valid_accuracy[i] = sess.run([loss,accuracy], feed_dict = {trainingInput: validData4D, labels: validTarget, reg: regulaizer, keeper: 0}) # Store validation error and accuracy
            store_test_error[i], store_test_accuracy[i] = sess.run([loss, accuracy], feed_dict = {trainingInput: testData4D, labels: testTarget, reg: regulaizer, keeper: 0}) # Store test error and accuracy


            ax1.cla()
            ax1.plot(store_training_error[0:i])
            ax1.plot(store_valid_error[0:i])
            ax1.plot(store_test_error[0:i])

            ax2.cla()
            ax2.plot(store_training_accuracy[0:i])
            ax2.plot(store_valid_accuracy[0:i])
            ax2.plot(store_test_accuracy[0:i])
            plt.pause(0.05)


        Loss_and_Acc_adam = np.array([store_training_error, store_valid_error, store_test_error, store_training_accuracy, store_valid_accuracy, store_test_accuracy])

        print('Time Taken to Train, Validate, and Test TensorFlow Neural = ', np.around(time.time() - start1,2))
        sess.close()
        return Loss_and_Acc_adam


def plotting_time(a,b,c):
    plt.figure(1)
    plt.clf()
    plt.plot(a[0,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
    plt.plot(a[1,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
    plt.plot(a[2,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
    plt.title('TensorFlow Neural Network - 0.5 Regularization - Loss Plots', fontsize = 32)
    plt.xlabel('Epoch', fontsize = 30)
    plt.xlim(0,50)
    plt.ylabel('Loss Value', fontsize = 30)
    plt.legend(ncol=1, loc='upper right', fontsize = 16)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(which = 'both', axis = 'both')
    plt.show()
    plt.ylim(0)


    plt.figure(2)
    plt.clf()
    plt.plot(a[3,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
    plt.plot(a[4,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
    plt.plot(a[5,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
    plt.title('TensorFlow Neural Network - 0.5 Regularization - Accuracy Plots', fontsize = 32)
    plt.xlabel('Epoch', fontsize = 30)
    plt.xlim(0,50)
    plt.ylim(80,100)

    plt.ylabel('Accuracy (%)', fontsize = 30)
    plt.legend(ncol=1, loc='upper left', fontsize = 16)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(which = 'both', axis = 'both')
    plt.show()

#
#    plt.figure(3)
#    plt.clf()
#    plt.plot(b[0,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
#    plt.plot(b[1,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
#    plt.plot(b[2,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
#    plt.title('Numpy Neural Network - 500 Layers - Loss Plots', fontsize = 32)
#    plt.xlabel('Epoch', fontsize = 30)
#    plt.xlim(0,200)
#    plt.ylabel('Loss Value', fontsize = 30)
#    plt.legend(ncol=1, loc='upper right', fontsize = 16)
#    plt.xticks(fontsize = 20)
#    plt.yticks(fontsize = 20)
#    plt.grid(which = 'both', axis = 'both')
#    plt.show()
#
#
#    plt.figure(4)
#    plt.clf()
#    plt.plot(b[3,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
#    plt.plot(b[4,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
#    plt.plot(b[5,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
#    plt.title('Numpy Neural Network - 500 Layers - Accuracy Plots', fontsize = 32)
#    plt.xlabel('Epoch', fontsize = 30)
#    plt.xlim(0,200)
#    plt.ylabel('Accuracy (%)', fontsize = 30)
#    plt.legend(ncol=1, loc='upper left', fontsize = 16)
#    plt.xticks(fontsize = 20)
#    plt.yticks(fontsize = 20)
#    plt.grid(which = 'both', axis = 'both')
#    plt.show()
#
#
#    plt.figure(5)
#    plt.clf()
#    plt.plot(c[0,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
#    plt.plot(c[1,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
#    plt.plot(c[2,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
#    plt.title('Numpy Neural Network - 1000 Layers - Loss Plots', fontsize = 32)
#    plt.xlabel('Epoch', fontsize = 30)
#    plt.xlim(0,200)
#    plt.ylabel('Loss Value', fontsize = 30)
#    plt.legend(ncol=1, loc='upper right', fontsize = 16)
#    plt.xticks(fontsize = 20)
#    plt.yticks(fontsize = 20)
#    plt.grid(which = 'both', axis = 'both')
#    plt.show()
#
#
#    plt.figure(6)
#    plt.clf()
#    plt.plot(c[3,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
#    plt.plot(c[4,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
#    plt.plot(c[5,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
#    plt.title('Numpy Neural Network - 1000 Layers - Accuracy Plots', fontsize = 32)
#    plt.xlabel('Epoch', fontsize = 30)
#    plt.xlim(0,200)
#    plt.ylabel('Accuracy (%)', fontsize = 30)
#    plt.legend(ncol=1, loc='upper left', fontsize = 16)
#    plt.xticks(fontsize = 20)
#    plt.yticks(fontsize = 20)
#    plt.grid(which = 'both', axis = 'both')
#    plt.show()
#
#
#    plt.figure(7)
#    plt.clf()
#    plt.plot(d[0,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
#    plt.plot(d[1,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
#    plt.plot(d[2,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
#    plt.title('Numpy Neural Network - 2000 Layers - Loss Plots', fontsize = 32)
#    plt.xlabel('Epoch', fontsize = 30)
#    plt.xlim(0,200)
#    plt.ylabel('Loss Value', fontsize = 30)
#    plt.legend(ncol=1, loc='upper right', fontsize = 16)
#    plt.xticks(fontsize = 20)
#    plt.yticks(fontsize = 20)
#    plt.grid(which = 'both', axis = 'both')
#    plt.show()
#
#
#    plt.figure(8)
#    plt.clf()
#    plt.plot(d[3,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
#    plt.plot(d[4,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
#    plt.plot(d[5,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
#    plt.title('Numpy Neural Network - 2000 Layers - Accuracy Plots', fontsize = 32)
#    plt.xlabel('Epoch', fontsize = 30)
#    plt.xlim(0,200)
#    plt.ylabel('Accuracy (%)', fontsize = 30)
#    plt.legend(ncol=1, loc='upper left', fontsize = 16)
#    plt.xticks(fontsize = 20)
#    plt.yticks(fontsize = 20)
#    plt.grid(which = 'both', axis = 'both')
#    plt.show()
##
#    plt.figure(9)
#    plt.clf()
#    plt.plot(a[2,:], 'k-', label=r'0.01', linewidth = 4, color = 'blue')
#    plt.plot(b[2,:], 'b-', label=r'0.1', linewidth = 4, color = 'orange')
#    plt.plot(c[2,:], 'r-', label=r'0.5', linewidth = 4, color = 'green')
##    plt.plot(d[2,:], 'r-', label=r'2000', linewidth = 4, color = 'black')
#    plt.title('TensorFlow Neural Network - Compare Regularization Loss', fontsize = 32)
#    plt.xlabel('Epoch', fontsize = 30)
#    plt.xlim(0,50)
#    plt.ylabel('Loss Value', fontsize = 30)
#    plt.legend(ncol=1, loc='upper right', fontsize = 16)
#    plt.xticks(fontsize = 20)
#    plt.yticks(fontsize = 20)
#    plt.grid(which = 'both', axis = 'both')
#    plt.show()
#
#
#    plt.figure(10)
#    plt.clf()
#    plt.plot(a[5,:], 'k-', label=r'0.01', linewidth = 4, color = 'blue')
#    plt.plot(b[5,:], 'b-', label=r'0.1', linewidth = 4, color = 'orange')
#    plt.plot(c[5,:], 'r-', label=r'0.5', linewidth = 4, color = 'green')
##    plt.plot(d[5,:], 'r-', label=r'2000', linewidth = 4, color = 'black')
#
#    plt.title('TensorFlow Neural Network - Compare Regularization Accuracy', fontsize = 32)
#    plt.xlabel('Epoch', fontsize = 30)
#    plt.xlim(0,50)
#    plt.ylabel('Accuracy (%)', fontsize = 30)
#    plt.ylim(80,100)
#    plt.legend(ncol=1, loc='upper left', fontsize = 16)
#    plt.xticks(fontsize = 20)
#    plt.yticks(fontsize = 20)
#    plt.grid(which = 'both', axis = 'both')
#    plt.show()


Training = "True"
Plotting = "True"


#if __name__ == "__main__":
#    main()
