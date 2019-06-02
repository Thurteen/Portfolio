import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import math
from matplotlib.patches import Ellipse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

####################################################################
#              QUESTION 1 (BEST MODEL PARAMATERS)                  #
# μ1   -1.10     -3.31         var1  0.0427         pi1  0.334     #                    #
# μ2    1.30      0.31         var2  0.0388         pi2  0.333     #
# μ3    0.11     -1.52         var3  0.9818         pi3  0.333     #
####################################################################
#                 QUESTION 2 (VALIDATION DATASET)                  #
#  K    K1%     K2%    K3%    K4%   K5%      VALIDATION SET LOSS   #   
#  1    100                                     11,651.40          #
#  2    65.6   34.4                              7,987.79          #        
#  3    35.1   33.6   31.3                       5,629.62          #
#  4     1.6   33.7   29.3   35.4                5,629.64          #
#  5     6.3   33.7   24.6   35.4   0.0          5,629.58          #
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
    epochs = 500

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

    trainingInput =tf.placeholder(tf.float32, shape=(None,dim))
    centroid, variance, logPi = initializeVars()
    logPDF = log_GaussPDF(trainingInput,centroid,variance)
    logPosterior  = log_posterior(logPDF, logPi)  #What dimension should this be?
    loss = -1.0*tf.reduce_sum(hlp.reduce_logsumexp(logPDF + logPi,keep_dims=True))
    optimizer =tf.train.AdamOptimizer(learning_rate= 0.01, beta1=0.9, beta2=0.99,epsilon=1e-5).minimize(loss)
    return optimizer, loss, tf.exp(logPosterior), centroid, variance, tf.exp(logPi), trainingInput

def initializeVars():
    centroid = tf.get_variable('mean', shape=(k,dim), initializer = tf.initializers.random_normal())
    phi = tf.get_variable('phi', shape=(1, k), initializer = tf.initializers.random_normal())
    variance = tf.math.exp(phi)
    gamma =  tf.get_variable('gamma', shape=(1,k), initializer = tf.initializers.random_normal())
    logPi = tf.math.log( tf.nn.softmax(gamma,axis=1))
    return centroid, variance, logPi

def log_GaussPDF(X, mu, variance):
    distanceSquared = distanceFunc(X,mu)
    return -1.0*tf.divide(tf.transpose(distanceSquared),2*variance) - dim/2*tf.log(2*math.pi*variance)

def log_posterior(log_PDF, log_Pi):
    lagrange = tf.add(log_PDF, log_Pi)
    return tf.subtract(lagrange,hlp.reduce_logsumexp(lagrange,keep_dims=True))

# Returns distance squared
def distanceFunc(X, mu):
    expandPoints = tf.expand_dims(X, 0)
    expandCentroid = tf.expand_dims(mu, 1)
    return tf.reduce_sum(tf.square(tf.subtract(expandPoints, expandCentroid)), 2)



def GMM(valid = False):
    loadData(valid)
    optimizer, loss, logPosterior, centroid, variance, weight, trainingInput = buildGraph()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(0,epochs):
            _,  trainLoss[i], mU, var, weights, probs= sess.run([optimizer, loss, centroid, variance, weight, logPosterior], feed_dict = {trainingInput: trainData})

            if valid:
                validLoss[i], mUV, varV, weightsV, probsV = sess.run([loss, centroid, variance, weight, logPosterior], feed_dict = {trainingInput: validData})

        plotter(valid,i)

        assign = np.argmax(probs,1)
        inCluster = np.mean(np.eye(k)[assign],0)
        scatter(trainData, assign, mU, np.transpose(var), weights)

        if valid:
            assignV = np.argmax(probsV,1)
            inClusterV = np.mean(np.eye(k)[assignV],0)
            scatter(validData, assignV, mUV, np.transpose(varV), weightsV)
            return mU, mUV, var, varV, weights, weightsV, probs, probsV, inCluster, inClusterV

    return mU, None, var, None, weights, None, probs, None, inCluster, None


def plotter(valid, iteration):
    plt.figure(1)
    plt.cla()
    plt.title("K = %i Loss vs Epoch" % k, fontsize = 32)
    plt.ylabel("Loss", fontsize = 30)
    plt.xlabel("Epoch", fontsize = 30)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(which = 'both', axis = 'both')
    plt.xlim((0,epochs))
    
    if valid == False:
        plt.plot(trainLoss[0:iteration])
    else:
        plt.plot(trainLoss[0:iteration]/trainData.shape[0])
        plt.plot(validLoss[0:iteration]/validData.shape[0])
    plt.pause(0.001)



def scatter(X, cluster, mU, var, weights):
    if dim ==2:
        var =  np.expand_dims(var,1)
        covariance = np.eye(dim)*var
        weights = np.reshape(weights,k)
    
        plt.figure()
        plt.title("K = %i Scatter Plot" % k, fontsize = 32)
        plt.xlabel("$x_1$", fontsize = 30)
        plt.ylabel("$x_2$", fontsize = 30)
        plt.scatter(X[:, 0], X[:, 1], c= cluster, s=1, cmap='viridis')
        plt.scatter(mU[:, 0], mU[:, 1], c='black', s=50, alpha=0.5)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)

        w_factor = 0.2 / weights.max()
        
        for pos, covar, w in zip(mU, covariance, weights.tolist()):
            drawEllipse(pos, covar, alpha= w * w_factor)
        plt.pause(0.001)

def drawEllipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))






mU, mUV, var, varV, weights, weightsV, probs, probsV, inCluster, inClusterV = GMM(False)
