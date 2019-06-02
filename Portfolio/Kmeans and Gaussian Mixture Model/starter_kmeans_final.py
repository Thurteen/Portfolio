import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

####################################################################
#       QUESTION 2 (10,000 DATASET)        #      QUESTION 3       #
#  K    K1%     K2%    K3%    K4%   K5%    #  VALIDATION SET LOSS  #   
#  1    100                                #     12,870.10         #
#  2    50.5   49.5                        #      2,960.67         #        
#  3    38.2   23.8   38.0                 #      1,629.21         #
#  4    37.1   12.1   37.3   13.5          #      1,054.54         #
#  5    36.8   11.1   37.0    7.6   7.5    #        907.21         #
####################################################################
#                     100 DIMENSION QUESTION                       #   
#  K         TRAINING SET LOSS K MEANS    TRAINING SET LOSS MoG    #
#  5               215,509                     1,091,210           #   
#  10              215,268                       834,024           #
#  15              215,361                       834,038           #
#  20              212,945                       486,583           #
#  30              211,200                       484,477           #
####################################################################

def loadData(valid = False):
    global k, epochs, trainData, validData, num_pts, dim, trainLoss, validLoss
    k = 3
    epochs = 350

#    trainData = np.load('data100D.npy')
    trainData = np.load('data2D.npy')
    [num_pts, dim] = np.shape(trainData)
    trainLoss = np.full((epochs, 1), np.inf)

    if valid:
        valid_batch = int(num_pts / 3.0)
        np.random.seed(45689)
        rnd_idx = np.arange(num_pts)
        np.random.shuffle(rnd_idx)
        validData = trainData[rnd_idx[:valid_batch]]
        trainData = trainData[rnd_idx[valid_batch:]]
        [num_pts, dim] = np.shape(trainData)
        validLoss = np.full((epochs, 1), np.inf)

def buildGraph():
    tf.reset_default_graph() # Clear any previous junk
    tf.set_random_seed(45689)

    trainingInput = tf.placeholder(tf.float32, shape=(None, dim))
    centroid = tf.get_variable('mean', shape=(k,dim), initializer=tf.initializers.random_normal())
    distanceSquared = distanceFunc(trainingInput,centroid) # Finds the euclidean norm
    loss = tf.math.reduce_sum(tf.math.reduce_min(distanceSquared,0))
    optimizer =tf.train.AdamOptimizer(learning_rate= 0.01, beta1=0.9, beta2=0.99,epsilon=1e-5).minimize(loss)
    return optimizer, loss,  distanceSquared, centroid, trainingInput

def plotter(valid, iteration):
    plt.figure(1)
    plt.cla()
    plt.title("K = %i Loss vs Epoch" % k, fontsize = 32)
    plt.ylabel("Loss", fontsize = 30)
    plt.xlabel("Epoch", fontsize = 30)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(which = 'both', axis = 'both')

    if valid == False:
        plt.plot(trainLoss[0:iteration])
    else:
        plt.plot(trainLoss[0:iteration]/trainData.shape[0])
        plt.plot(validLoss[0:iteration]/validData.shape[0])
    plt.pause(0.001)


def scatter(X, cluster, mU):
    if dim == 2:
        plt.figure()
        plt.title("K = %i Scatter Plot" % k, fontsize = 32)
        plt.xlabel("$x_1$", fontsize = 30)
        plt.ylabel("$x_2$", fontsize = 30)
        plt.scatter(X[:, 0], X[:, 1], c= cluster, s=1, cmap='viridis')
        plt.scatter(mU[:, 0], mU[:, 1], c='black', s=50, alpha=0.5)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)

        plt.pause(0.001)

def kMeans(valid=False):
    loadData(valid)
    optimizer, loss,  distanceSquared, centroid, trainingInput = buildGraph()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(0,epochs):
            _, trainLoss[i], dist, mU = sess.run([optimizer, loss,  distanceSquared, centroid], feed_dict = {trainingInput:trainData})

            if valid:
                validLoss[i],distV, mUV = sess.run([loss,  distanceSquared, centroid], feed_dict = {trainingInput: validData})
        plotter(valid,i)
        assign = np.argmin(dist,0)
        inCluster = np.mean(np.eye(k)[assign],0)
        scatter(trainData, assign, mU)

        if valid:
            assignV = np.argmin(distV,0)
            inClusterV = np.mean(np.eye(k)[assignV],0)
            scatter(validData, assignV, mUV)
            return mU, mUV, inCluster, inClusterV
    return mU, None, inCluster, None



def distanceFunc(X, mu): # Returns distance squared
    expandPoints = tf.expand_dims(X, 0)
    expandCentroid = tf.expand_dims(mu, 1)
    return tf.reduce_sum(tf.square(tf.subtract(expandPoints, expandCentroid)), 2)

mU, mUV, inCluster, inClusterV = kMeans(False)
