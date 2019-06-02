import sys
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ~ ~ ~ ~ ~ ~ ~ ~  Preamble ~ ~ ~ ~ ~ ~ ~ ~ #

# This script was written cooperatively by Dylan Johnston (1003852690) And Omar Ismail (999467660).
# It optimizes the notMNIST dataset using a gradient descent algoirthm. It executes this using the Linear Regression, Logistic Regression, Normal Equation and Adam Optimizer methods.
# When the script is run in the interpreter, the output will be the training and test accuracy and loss for each method, along with the time required for the computation. Graphs of the
# accuracy and loss of Linear regression, Logistic regression and Adam optimzation are created and displayed. Variables can be changed in the function globalVariableDefine. For
# Adam optimzation, the loss type can be changed by changing the argument of the function to 'CE'.

# ~ ~ ~ ~ ~ ~ ~ ~  Main Body of Code ~ ~ ~ ~ ~ ~ ~ ~ #
# Function that calls optimization scheme, accuracy and loss calculations, and plotting function.

def main():
    # Define variables required for the optmization, accuracy and loss calculations
    globalVariableDefine()
    # Toggle whether weights and biases should be trained on program start
    if Training == "True":

        # Record the length of time required to train weights and biases
        start1 = time.time()
        global WT_linreg, bT_linreg
        # Calls linear regression optimization, which minimizes mean square error (MSE) during gradient descent, returns optimized weights and bias.
        WT_linreg, bT_linreg = grad_descent(W, b, trainData, trainTarget, learning_rate, training_epochs, regularization, error_tol, LossType = "MSE", Multibatch = "False")

        global Loss_linreg_train, Acc_linreg_train, Loss_linreg_valid, Acc_linreg_valid, Loss_linreg_test, Acc_linreg_test
        # Use the optimized weight and bias at each epoch to vectorize the MSE loss calculation, returns an accuracy and loss value vector with length = training_epochs
        Loss_linreg_train, Acc_linreg_train = MSE(WT_linreg, bT_linreg, trainData, trainTarget, regularization)
        Loss_linreg_valid, Acc_linreg_valid = MSE(WT_linreg, bT_linreg, validData, validTarget, regularization)
        Loss_linreg_test, Acc_linreg_test = MSE(WT_linreg, bT_linreg, testData, testTarget, regularization)
        # Concatenate the data from all three data sets, format: (train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc)
        Loss_and_Acc_LinReg = np.array([Loss_linreg_train, Loss_linreg_valid, Loss_linreg_test, Acc_linreg_train, Acc_linreg_valid, Acc_linreg_test])
        print('Linear Regression Train Time = ', np.around(time.time() - start1,2))
        print('Final Training Loss for Linear Regression = ', np.around(Loss_linreg_train[-1],2))
        print('Final Training Accuracy for Linear Regression = ', np.around(Acc_linreg_train[-1],2), '%')
        print('Final Test Loss for Linear Regression = ', np.around(Loss_linreg_test[-1],2))
        print('Final Test Accuracy for Linear Regression = ', np.around(Acc_linreg_test[-1],2), '%')
        print('')

        start2 = time.time()
        global WT_logreg, bT_logreg
        # Calls logistic regression optimization, which minimizes cross entropy during gradient descent, returns optimized weights and bias.
        WT_logreg, bT_logreg = grad_descent(W, b, trainData, trainTarget, learning_rate, training_epochs, regularization, error_tol, LossType = "CE", Multibatch = "False")
        # Same buisness, for the Cross-Entropy cost function weights and biases
        global Loss_logreg_train, Acc_logreg_train, Loss_logreg_valid, Acc_logreg_valid, Loss_logreg_test, Acc_logreg_test
        Loss_logreg_train, Acc_logreg_train = CE(WT_logreg, bT_logreg, trainData, trainTarget, regularization)
        Loss_logreg_valid, Acc_logreg_valid = CE(WT_logreg, bT_logreg, validData, validTarget, regularization)
        Loss_logreg_test, Acc_logreg_test = CE(WT_logreg, bT_logreg, testData, testTarget, regularization)
        Loss_and_Acc_LogReg = np.array([Loss_logreg_train, Loss_logreg_valid, Loss_logreg_test, Acc_logreg_train, Acc_logreg_valid, Acc_logreg_test])
        print('Logistic Regression Train Time = ', np.around(time.time() - start2,2))
        print('Final Training Loss for Logistic Regression = ', np.around(Loss_logreg_train[-1],2))
        print('Final Training Accuracy for Logistic Regression = ', np.around(Acc_logreg_train[-1],2), '%')
        print('Final Test Loss for Logistic Regression = ', np.around(Loss_logreg_test[-1],2))
        print('Final Test Accuracy for Logistic Regression = ', np.around(Acc_logreg_test[-1],2), '%')
        print('')

        start3 = time.time()
        global WT_Normal_Function, bT_Normal_Function
        # Calls normal function equation, which solves for the optimal weights and bias analytically using the gradient of the MSE cost function, returns optimized weights and bias.
        Normal_Function_Loss, Normal_function_Accuracy = normalEquation(b, trainData, trainTarget, regularization)
        print('Normal Equation Evaluation Time = ', np.around(time.time() - start3,2))
        print('Final Training Loss for Normal Equation = ', np.around(Normal_Function_Loss[-1],2))
        print('Final Training Accuracy for Normal Equation = ', np.around(Normal_function_Accuracy[-1],2), '%')
        print('')

        start4 = time.time()
        global WT_Adam, bT_Adam, Loss_and_Acc_adam
        # Loss function can be changed to Cross Entropy by changing argument of Adam function to 'CE'
        WT_Adam, bT_Adam, Loss_and_Acc_adam = Adam('MSE')
        print('Adam Optimization Evaluation Time = ', np.around(time.time() - start4,2))
        print('Final Training Loss for Adam Optimization = ', np.around(Loss_and_Acc_adam[0,-1],2))
        print('Final Training Accuracy for Adam Optimization = ', np.around(Loss_and_Acc_adam[3,-1],2), '%')
        print('Final Test Loss for Adam Optimization = ', np.around(Loss_and_Acc_adam[2,-1],2))
        print('Final Test Accuracy for Adam Optimization = ', np.around(Loss_and_Acc_adam[5,-1],2), '%')


    # Toggle plotting the data. This function (unmodified) accepts three np.arrays that have columns according to concatenation format above, and plots the first three columns of each np.array on one plot, and the second 3 columns on a second plot
    if Plotting == "True":
        plotting_time(Loss_and_Acc_LinReg, Loss_and_Acc_LogReg, Loss_and_Acc_adam)

def grad_descent(W, b, x, y, alpha, epochs, r, er, LossType = "None", Multibatch = "None"):
    weights = np.zeros((len(W), epochs+1))
    weights[:,0,None] = W
    biases = np.zeros((1, epochs+1))
    biases[:,0,None] = b

    if Multibatch == "True":
        for epoch in range(epochs):
            x, y = randomize(x, y)
            n_batches = int(len(x) / batch_size)
            for batch in range(n_batches):
                start = batch * batch_size
                end = (batch + 1) * batch_size
                x_batch, y_batch = next_batch(x, y, start, end)

                if LossType == "MSE":
                    grad_bias, grad_weight = gradMSE(weights[:,epoch,None], biases[:,epoch,None], x_batch, y_batch, r)
                    weights[:, epoch+1, None] = np.subtract(weights[:,epoch,None], alpha*grad_weight)
                    biases[:, epoch+1, None] = np.subtract(biases[:,epoch,None], alpha*grad_bias)

                if LossType == "CE":
                    grad_bias, grad_weight = gradCE(weights[:,epoch,None], biases[:,epoch,None], x_batch, y_batch, r)
                    weights[:, epoch+1, None] = np.subtract(weights[:,epoch,None], alpha*grad_weight)
                    biases[:, epoch+1, None] = np.subtract(biases[:,epoch,None], alpha*grad_bias)

            if (epoch + 1) % 100 == 0:
                print("Epoch", (epoch+1))

    else:
        if LossType == "MSE":
            for epoch in range(epochs):
                grad_bias, grad_weight = gradMSE(weights[:,epoch,None], biases[:,epoch,None], x, y, r)
                weights[:, epoch+1, None] = np.subtract(weights[:,epoch,None], alpha*grad_weight)
                biases[:, epoch+1, None] = np.subtract(biases[:,epoch,None], alpha*grad_bias)
                if (np.abs(weights[:,epoch+1,None] - weights[:,epoch,None])).all() <= error_tol:
                    break

        if LossType == "CE":
            for epoch in range(epochs):
                grad_bias, grad_weight = gradCE(weights[:,epoch,None], biases[:,epoch,None], x, y, r)
                weights[:, epoch+1, None] = np.subtract(weights[:,epoch,None], alpha*grad_weight)
                biases[:, epoch+1, None] = np.subtract(biases[:,epoch,None], alpha*grad_bias)
                if (np.abs(weights[:,epoch+1,None] - weights[:,epoch,None])).all() <= error_tol:
                    break

    return weights, biases

# ~ ~ ~ ~ ~ ~ ~ ~ Linear Regression Functions ~ ~ ~ ~ ~ ~ ~ ~ #
# Functions used when loss function is MSE

def gradMSE(W, b, x, y, reg):
    h = np.matmul(x,W)+b
    #Calculate gradient
    grad_bias = (1/y.shape[0])*np.sum(h-y)
    grad_Weight = (1/y.shape[0])*np.matmul(np.transpose(x),(h-y)) + reg*W
    return grad_bias, grad_Weight

def MSE(W, b, x, y, r):
    h = np.add(np.matmul(x, W), b)
    # This returns a 5001 x 5001 matrix, where the diagonal elements are the loss at each epoch
    loss = np.add(np.divide(np.matmul(np.transpose(h - y), (h - y)), (2.*len(y))), np.multiply(r/2, np.matmul(np.transpose(W), W)))
    acc = np.sum(np.equal(np.around(h), np.repeat(y, W.shape[1], axis = 1)), axis = 0) / len(y)
    return np.diagonal(loss), acc*100

def normalEquation(b, x, y, r):
    # Put wieghts and biases in expanded dimension notation
    b = np.ones((x.shape[0],1))
    x = np.append(b,x, axis = 1)
    partial_identity = np.identity(x.shape[1])
    partial_identity[0,0] = 0
    weight = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x),x) + r*partial_identity),np.transpose(x)),y)
    h = np.matmul(x,weight)
    loss = np.add(np.divide(np.matmul(np.transpose(h - y), (h - y)), (2.*len(y))), np.multiply(r/2, np.matmul(np.transpose(weight), weight)))
    acc = np.sum(np.equal(np.around(h), np.repeat(y, W.shape[1], axis = 1)), axis = 0) / len(y)
    return loss, acc*100

# ~ ~ ~ ~ ~ ~ ~ ~ Logistic Regression Functions ~ ~ ~ ~ ~ ~ ~ ~ #
# Functions used when Cross-Entropy loss function is used

def sigmoid(z):
    return np.divide(1,1+np.exp(-z))

def CE(W, b, x, y, r):
    z = np.add(np.matmul(x, W), b)
    h = sigmoid(z)
    loss = np.add(np.divide((-np.matmul(np.transpose(y), np.log(h)) - np.matmul(np.transpose(1 - y), np.log(1 - h))), y.shape[0]), np.multiply(r/2, np.matmul(np.transpose(W), W)))
    acc = np.sum(np.equal(np.around(h), np.repeat(y, W.shape[1], axis = 1)), axis = 0) / len(y)

    return np.diagonal(loss), acc*100

def gradCE(W, b, x, y, reg):
    z = (np.matmul(x,W))+b
    h = sigmoid(z)
    grad_bias = (1/y.shape[0])*np.sum(h-y)
    grad_Weight = (1/y.shape[0])*np.matmul(np.transpose(x),(h-y)) + reg*W

    return grad_bias, grad_Weight

# ~ ~ ~ ~ ~ ~ ~ ~  SGD ~ ~ ~ ~ ~ ~ ~ ~ #

def buildGraph(loss_type):
    #--------------Define Equation Paramater Placeholders----------------------#

    tf.set_random_seed(421)
    X = tf.placeholder(tf.float32, [None, trainTarget.shape[0]])
    Y_target = tf.placeholder(tf.float32, [None, 1])
    reg = tf.placeholder(tf.float32,None, name='regulariser')
    W = tf.Variable(tf.truncated_normal(shape=[trainTarget.shape[0],1], stddev=0.5), name='weights')
    b = tf.Variable(tf.truncated_normal(shape=[1,1], stddev=0.5), name='biases')

    model = tf.matmul(X,W) + b

    if loss_type == "MSE":
        #use TensorFlow function to simplify expression
        MSE = tf.losses.mean_squared_error(labels = Y_target, predictions = model) + reg*tf.nn.l2_loss(W)
        lossfn = tf.reduce_mean(MSE)
        #variables defined above and changed through console
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = Beta1, beta2 = Beta2, epsilon = eps).minimize(lossfn)
        #Any value less than 0 becomes 0, any value greater than 1 becomes 1, and then any value in between is rounded to the nearest integer
        Y_predicted = tf.round(tf.clip_by_value(model,0,1))
        correct = tf.cast(tf.equal(Y_predicted, Y_target), dtype=tf.float32)
        accuracy = tf.reduce_mean(correct)*100
        return  X, Y_target, W,b, lossfn, optimizer, accuracy, reg

    elif loss_type == "CE":
        #use TensorFlow function to simplify expression
        CE = tf.nn.sigmoid_cross_entropy_with_logits(logits = model,labels = Y_target) + reg*tf.nn.l2_loss(W)
        lossfn = tf.reduce_mean(CE)
        #variables defined above and changed through console
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = Beta1, beta2 = Beta2, epsilon = eps).minimize(lossfn)

        Y_predicted = tf.round(tf.nn.sigmoid(model))
        correct = tf.cast(tf.equal(Y_predicted, Y_target), dtype=tf.float32)
        accuracy = tf.reduce_mean(correct)*100
        return  X, Y_target, W,b, lossfn, optimizer, accuracy, reg

def Adam(loss_type):
    X, Y_target, W, b, lossfn, optimizer, accuracy, reg = buildGraph(loss_type)   # Building the graph

    #-----------------------Initialize Storage Variables-----------------------#
    store_training_error = np.zeros((SGD_training_epochs,1))
    store_valid_error= np.zeros((SGD_training_epochs,1))
    store_test_error= np.zeros((SGD_training_epochs,1))
    store_training_accuracy = np.zeros((SGD_training_epochs,1))
    store_valid_accuracy = np.zeros((SGD_training_epochs,1))
    store_test_accuracy = np.zeros((SGD_training_epochs,1))

    init = tf.global_variables_initializer()   # Initialize session

    with tf.Session() as sess:
        sess.run(init)
        batch_number = int(trainTarget.shape[0]/batch_size) # Calculate batch number

        for i in range(SGD_training_epochs): # Loop across SGD_training_epochs
            batch_index = np.random.permutation(trainData.shape[0]) # Reshuffle training data
            Y_split = np.split(trainTarget[batch_index],batch_number) # Split into the number of batches
            X_split = np.split(trainData[batch_index],batch_number) # Split into the number of batches

            for j in range(len(X_split)): # Loop through each batch
                sess.run([optimizer], feed_dict = {X: X_split[j], Y_target: Y_split[j], reg: regularization}) # Let us OPTIMIZE!
                store_training_error[i], store_training_accuracy[i] = sess.run([lossfn, accuracy], feed_dict =  {X: X_split[j], Y_target: Y_split[j], reg: regularization}) # Store training error and accuracy

            store_valid_error[i], store_valid_accuracy[i] = sess.run([lossfn,accuracy], feed_dict = {X: validData, Y_target: validTarget, reg: regularization}) # Store validation error and accuracy
            store_test_error[i], store_test_accuracy[i] = sess.run([lossfn, accuracy], feed_dict = {X: testData, Y_target: testTarget, reg: regularization}) # Store test error and accuracy

        Final_Weight, Final_Bias = sess.run([W,b]) # Stores Final Weights and Biases in a form that is not a Tensor Object

        y_tested_final_accuracy = sess.run(accuracy, feed_dict = {X: testData, Y_target: testTarget}) # Calculate the final prediction accuracy on the test set
        # print("Final Testing Accuracy: {0:.2f}%".format(y_tested_final_accuracy))  # Print the final prediction accuracy on the test s

    dataframe = pd.DataFrame({'Training Error': [store_training_error], 'Validation Error': [store_valid_error], 'Test Error': [store_test_error],
                              'Training Accuracy': [store_training_accuracy], 'Validation Accuracy': [store_valid_accuracy], 'Test Accuracy':[store_test_accuracy],
                              'Weights': [Final_Weight],'Bias': [Final_Bias], 'Final Test Accuracy':[y_tested_final_accuracy]}) # Combine Data to one DataFrame
    dataframe.to_json('{}_A{}_L{}_B{}_B1{}_B2{}_E{}'.format(loss_type,learning_rate,regularization, batch_size,Beta1,Beta2,eps)) # Save the data as a JSON file

    Loss_and_Acc_adam = np.array([store_training_error, store_valid_error, store_test_error, store_training_accuracy, store_valid_accuracy, store_test_accuracy])

    return Final_Weight, Final_Bias, Loss_and_Acc_adam

# Added a function where you can show a picture from the data, and if the trained weights and bias matrices are sent to it, then it will tell you what the model predicts
# We don't know why, but this only works if you have your graphics printing inside the console, and not as a pop-up
# We promise, this is a "fascinating" game!

def playGame(W, b, x):  # Send the final weight and bias, as well as X_train, valid or test
    play_index = np.random.choice(x.shape[0],x.shape[0]) # Shuffle data randomly

    for i in play_index:
        predict = predictLogistic(W, b, np.transpose(x[i,:,None]))
        plt.imshow(np.array(x[i]).reshape(28,28), cmap = 'gray') # Show each row as a picture
        plt.show()
        if predict == True:
            print("The system has predicted C")
        elif predict == False:
            print("The system has predicted J")
        input() # Press anything so you can see the next picture

def predictLogistic(W, b, x):
    return sigmoid(np.matmul(x,W)+b) >= 0.5

# ~ ~ ~ ~ ~ ~ ~ ~  Data Handling Functions ~ ~ ~ ~ ~ ~ ~ ~ #

def loadData():
    with np.load('notMNIST.npz') as data:
        Data, Target = data['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target == posClass) + (Target == negClass)
        Data = Data[dataIndx] / 255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target == posClass] = 1
        Target[Target == negClass] = 0
        np.random.seed(555)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return reshapeData(trainData), reshapeData(validData), reshapeData(testData), trainTarget, validTarget, testTarget

def reshapeData(dataset):
    img_h = img_w = 28             # MNIST images are 28x28
    global img_size_flat
    img_size_flat = img_h * img_w  # 28x28=784, the total number of pixels
    dataset = dataset.reshape((-1,img_size_flat)).astype(np.float64)
    return dataset

def globalVariableDefine():
    global trainData, validData, testData, trainTarget, validTarget, testTarget
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    global loss_type, learning_rate, training_epochs, error_tol, regularization, regtf, batch_size, SGD_training_epochs
    learning_rate = 0.005
    training_epochs = 5000
    error_tol = 0.0000001
    regularization = 0.
    batch_size = int(500)
    SGD_training_epochs = 700

    global W, b, Beta1, Beta2, eps
    b = np.random.randn(1,1)
    W = np.random.normal(0,0.5,(trainData.shape[1],1))
    Beta1 = 0.9
    Beta2 = 0.999
    eps = 1e-8

def plotting_time(a,b,c):
    plt.figure(1)
    plt.clf()
    plt.plot(a[0,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
    plt.plot(a[1,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
    plt.plot(a[2,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
    plt.title('Linear Regression Loss', fontsize = 32)
    plt.xlabel('Epoch', fontsize = 30)
    plt.ylabel('Loss Value', fontsize = 30)
    plt.legend(ncol=1, loc='upper right', fontsize = 16)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(which = 'both', axis = 'both')

    plt.figure(2)
    plt.clf()
    plt.plot(a[3,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
    plt.plot(a[4,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
    plt.plot(a[5,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
    plt.title('Linear Regression Accuracy', fontsize = 32)
    plt.xlabel('Epoch', fontsize = 30)
    plt.ylabel('Accuracy (%)', fontsize = 30)
    plt.legend(ncol=1, loc='upper left', fontsize = 16)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(which = 'both', axis = 'both')

    plt.figure(3)
    plt.clf()
    plt.plot(b[0,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
    plt.plot(b[1,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
    plt.plot(b[2,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
    plt.title('Logistic Regression Loss', fontsize = 32)
    plt.xlabel('Epoch', fontsize = 30)
    plt.ylabel('Loss Value', fontsize = 30)
    plt.legend(ncol=1, loc='upper right', fontsize = 16)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(which = 'both', axis = 'both')

    plt.figure(4)
    plt.clf()
    plt.plot(b[3,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
    plt.plot(b[4,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
    plt.plot(b[5,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
    plt.title('Logistic Regression Accuracy', fontsize = 32)
    plt.xlabel('Epoch', fontsize = 30)
    plt.ylabel('Accuracy (%)', fontsize = 30)
    plt.legend(ncol=1, loc='upper left', fontsize = 16)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(which = 'both', axis = 'both')

    plt.figure(5)
    plt.clf()
    plt.plot(c[0,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
    plt.plot(c[1,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
    plt.plot(c[2,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
    plt.title('Adam Optimizer Loss', fontsize = 32)
    plt.xlabel('Epoch', fontsize = 30)
    plt.ylabel('Loss Value', fontsize = 30)
    plt.legend(ncol=1, loc='upper right', fontsize = 16)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(which = 'both', axis = 'both')

    plt.figure(6)
    plt.clf()
    plt.plot(c[3,:], 'k-', label=r'Training Set', linewidth = 4, color = 'blue')
    plt.plot(c[4,:], 'b-', label=r'Validation Set', linewidth = 4, color = 'orange')
    plt.plot(c[5,:], 'r-', label=r'Test Set', linewidth = 4, color = 'green')
    plt.title('Adam Optimizer Accuracy', fontsize = 32)
    plt.xlabel('Epoch', fontsize = 30)
    plt.ylabel('Accuracy (%)', fontsize = 30)
    plt.legend(ncol=1, loc='upper left', fontsize = 16)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(which = 'both', axis = 'both')
    plt.show()

Training = "True"
Plotting = "True"

if __name__ == "__main__":
    main()
