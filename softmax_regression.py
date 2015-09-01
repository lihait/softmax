#!/usr/bin/env python

import sys
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

cost = 0.0
lrate = .1
grad = np.array([])
lambda_factor = 0.0
nclasses = 2

def vector_to_collumn(vec):
    length = len(vec)
    A = np.array(vec)
    for i in range(length):
        A[i] = vec[i]
    A.reshape(len(A), 1)
    return A

def vector_to_row(vec):
    A = vector_to_collumn(vec)
    A = A.transpose()
    return A

def rows_collumns(mat):
    return (len(mat), len(mat[0]))

def vector_to_matrix(vec):

    rows = len(vec[0])
    cols = len(vec)
    mat = np.zeros(shape=(rows, cols))
    for i in xrange(rows):
        for j in xrange(cols):
            mat[i][j] = vec[j][i]

    return mat

def update_costFunction_gradient(mat_x, row, weights, lambda_factor):

    nsamples = len(mat_x[0])
    nfeatures = len(mat_x)
    theta = np.asarray(weights)

    ### print stats for data structures ###
    #print "################ INTERMEDIARY STATS ####################"
    #print "weights: col", len(weights[0]), "row", len(weights), '\n'
    #print "mat_x: col", len(mat_x[0]), "row", len(mat_x), '\n'
    #print "theta: col", len(theta[0]), "row", len(theta), '\n'
    #print "\ntheta:", theta
    #print "########################################################"
    M = np.dot(theta,mat_x)

    max = np.amax(M, axis=0) #get max element in the collumn
    temp = np.tile(max, (nclasses, 1))

    M = np.subtract(M, temp)
    M = np.exp(M)
    mat_sum = np.sum(M, axis=0) #returns an array of sums across collumns
    temp = np.tile(mat_sum, (nclasses, 1))

    M = M / temp

    groundTruth = np.zeros(shape=(nclasses, nsamples))
    for i in xrange(len(row)):
        a = row[i]
        groundTruth[a][i] = 1

    temp = groundTruth * np.log(M)
    sum_cost = np.sum(temp)
    cost = -sum_cost / nsamples
    cost += np.dot(np.sum(theta**2), (lambda_factor/2))

    #calc gradient
    temp = np.subtract(groundTruth, M)
    temp = np.dot(temp, mat_x.transpose())
    grad = -temp / nsamples
    grad += np.dot(lambda_factor, theta)

def calculate(mat_x, weights):
    theta = np.asarray(weights)
    M = np.dot(theta, mat_x)
    M_max = np.amax(M, axis=0)
    temp = np.tile(M_max, (nclasses, 1))
    M = np.subtract(M, temp)
    M = np.exp(M)
    mat_sum = np.sum(M, axis=0)
    temp = np.tile(mat_sum, (nclasses, 1))
    M = M / temp
    M = np.log(M)
    res = np.zeros(len(M[0]))
    for i in xrange(len(M[0])):
        maxele = -sys.maxint
        which = 0
        for j in xrange(len(M)):
            if M[j][i] > maxele:
                maxele = M[j][i]
                which = j
        res[i] = which
    return res

def softmax(vecX, vecY, testX, testY):
    nsamples = len(vecX)
    nfeatures = len(vecX[0])

    # change vecX and vecY into matrix or vector
    #print "################ INTERMEDIARY STATS ####################"
    #print "type of vecX:", type(vecX)
    #print "type of vecY:", type(vecY)
    #print "length of vecX:", len(vecX), '\n'
    #print "length of vecY:", len(vecY), '\n'
    y = vector_to_row(vecY)
    x = vector_to_matrix(vecX)

    init_epsilon = 0.12

    weights = np.zeros(shape=(nclasses, nfeatures))
    weights = [[random.uniform(0, 1) for num in list] for list in weights]
    weights = np.dot(weights,2 * init_epsilon)
    weights -= init_epsilon

    grad = np.zeros(shape=(nclasses, nfeatures))
    # Gradient Checking (remember to disable this part after you're sure the 
    # cost function and dJ function are correct)

    update_costFunction_gradient(x, y, weights, lambda_factor);
    dJ = np.matrix(grad);
    dJ = np.asarray(grad)
    print "\ntest!!!!\n"
    epsilon = 1e-4
    for i in range(len(weights)):
        for j in range(len(weights[0])):
            memo = weights[i][j]
            weights[i][j] = memo + epsilon;
            update_costFunction_gradient(x, y, weights, lambda_factor);
            value1 = cost;
            weights[i][j] = memo - epsilon;
            update_costFunction_gradient(x, y, weights, lambda_factor);
            value2 = cost;
            tp = (value1 - value2) / (2 * epsilon)
            weights[i][j] = memo;


    converge = 0
    lastcost = 0.0
    while converge < 5000:
        update_costFunction_gradient(x, y, weights, lambda_factor)
        weights -= lrate * grad

        if abs((cost - lastcost)) <= 5e-6 and converge > 0:
            break
        lastcost = cost
        converge += 1
    print "########### result ############\n"

    yT = vector_to_row(testY)
    xT = vector_to_matrix(testX)
    res = calculate(xT, weights)
    error = yT - res
    correct = len(error)
    for i in range(len(error)):
        if error[i] != 0:
            correct -= 1
    print "correct: {}, total: {}, accuracy: {}".format(correct,len(error),(correct/len(error)))

if __name__ == "__main__":

    file1 = "trainX.txt"
    file2 = "trainY.txt"
    file3 = "testX.txt"
    file4 = "testY.txt"
    #np.set_printoptions(precision=3)         
    # train with file 1
    with open(file1, "r+") as f1:

        numofX = 30
        counter = 0
        array = []
        vecX = [[]]
        tpdouble = 0
        try:
            for line in f1:
                line_nums = line.split()
                for num in line_nums:
                    array.append(eval(num))
        except:
            print('f1) Ignoring: malformed line: "{}"'.format(line))

        for tpdouble in array:
            if counter/numofX >= len(vecX):
                tpvec = []
                vecX.append(tpvec)
            vecX[counter/numofX].append(tpdouble)
            counter += 1
    f1.close()
    # train with file 2
    with open(file2, "r+") as f2 :

        vecY, array = [], []
        try:
            for line in f2:
                line_nums = line.split()
                for num in line_nums:
                    array.append(eval(num))
        except:
            print('f2) Ignoring: malformed line: "{}"'.format(line))
        for tpdouble in array:
            vecY.append(tpdouble)
        f2.close()
    for i in range(1, len(vecX)):
        if len(vecX[i]) != len(vecX[i - 1]):
            sys.exit(0)

    assert len(vecX) == len(vecY)
    if len(vecX) != len(vecY):
        sys.exit(0)

    #test against file 3
    with open(file3, "r") as f3:
        vecTX, array = [[]], []
        counter = 0
        try:
            for line in f3:
                line_nums = line.split()
                for num in line_nums:
                    array.append(eval(num))
        except:
            print('f3) Ignoring: malformed line: "{}"'.format(line))
        for tpdouble in array:
            if counter/numofX >= len(vecTX):
                tpdouble = []
                vecTX.append(tpvec)
            vecTX[counter/numofX].append(tpdouble)
            counter += 1
        f3.close()

    # test against file 4
    with open(file4, "r") as f4:

        vecTY, array = [], []
        try:
            for line in f4:
                line_nums = line.split()
                for num in line_nums:
                    array.append(eval(num))
        except:
            print('f4) Ignoring: malformed line: "{}"'.format(line))

        for tpdouble in array:
            vecTY.append(tpdouble)
        f4.close()
        start = time.clock()
        softmax(vecX, vecY, vecTX, vecTY)
        end = time.clock()
        sys.exit(0)

