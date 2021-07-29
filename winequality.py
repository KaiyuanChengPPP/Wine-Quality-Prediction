import pandas as pd
import numpy as np
import matplotlib as plt
import math
import random
import decimal
from sympy import symbols, diff

wine = pd.read_csv("winequality-red.csv",delimiter=",")

def get_slope():
        slope = []
        for i in range (0,11):
                slope.append(float(decimal.Decimal(random.randrange(-30, 30))/100))
        return slope

def pre_qual(intercept,slope):
        predict_qual = []
        for j in range(0,1599):
                predict_qual.append(intercept + np.sum(np.multiply(slope,wine.iloc[j,:-1])))
        return predict_qual

def observed_qual():
        observe = []
        for i in range(0,1599):
                observe.append(wine.iloc[:,11][i])
        return observe

def get_mse(observe,predict):
        residual = np.subtract(predict,observe)
        square = np.multiply(residual,residual)
        sum = np.sum(square)
        return sum/1600

# def get_partials(observe,predict):
#         p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, pi = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#         for i in range(0, 1599):
#                 partial_xi =
#                 pi = pi + partial_xi
#                 partial_x1 = np.subtract(observe,predict)*wine.iloc[i,:-1][0]
#                 p1 = p1 +


def get_partials(slope,intercept,observe):
        p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,pi=0,0,0,0,0,0,0,0,0,0,0,0
        for i in range (0, 1599):
                partial_xi = ((intercept + np.sum(np.insert(list(np.multiply(slope,wine.iloc[i,:-1])),0,intercept)))-observe[i])/1600
                pi = pi+partial_xi
        for i in range(0, 1599):
                partial_x1 = (((intercept + np.sum(
                        np.insert(list(np.multiply(slope, wine.iloc[i, :-1])), 0, intercept))) - observe[i]) \
                             * wine.iloc[i, :-1][0])/1600
                p1 = p1 + partial_x1
        for i in range(0, 1599):
                partial_x2 = (((intercept + np.sum(
                        np.insert(list(np.multiply(slope, wine.iloc[i, :-1])), 0, intercept))) - observe[i]) \
                             * wine.iloc[i, :-1][1])/1600
                p2 = p2 + partial_x2
        for i in range(0, 1599):
                partial_x3 = (((intercept + np.sum(
                        np.insert(list(np.multiply(slope, wine.iloc[i, :-1])), 0, intercept))) - observe[i]) \
                             * wine.iloc[i, :-1][2])/1600
                p3 = p3 + partial_x3
        for i in range(0, 1599):
                partial_x4 = (((intercept + np.sum(
                        np.insert(list(np.multiply(slope, wine.iloc[i, :-1])), 0, intercept))) - observe[i]) \
                             * wine.iloc[i, :-1][3])/1600
                p4 = p4 + partial_x4
        for i in range(0, 1599):
                partial_x5 = (((intercept + np.sum(
                        np.insert(list(np.multiply(slope, wine.iloc[i, :-1])), 0, intercept))) - observe[i]) \
                             * wine.iloc[i, :-1][4])/1600
                p5 = p5 + partial_x5
        for i in range(0, 1599):
                partial_x6 = (((intercept + np.sum(
                        np.insert(list(np.multiply(slope, wine.iloc[i, :-1])), 0, intercept))) - observe[i]) \
                             * wine.iloc[i, :-1][5])/1600
                p6 = p6 + partial_x6
        for i in range(0, 1599):
                partial_x7 = (((intercept + np.sum(
                        np.insert(list(np.multiply(slope, wine.iloc[i, :-1])), 0, intercept))) - observe[i]) \
                             * wine.iloc[i, :-1][6])/1600
                p7 = p7 + partial_x7
        for i in range(0, 1599):
                partial_x8 = (((intercept + np.sum(
                        np.insert(list(np.multiply(slope, wine.iloc[i, :-1])), 0, intercept))) - observe[i]) \
                             * wine.iloc[i, :-1][7])/1600
                p8 = p8 + partial_x8
        for i in range(0, 1599):
                partial_x9 = (((intercept + np.sum(
                        np.insert(list(np.multiply(slope, wine.iloc[i, :-1])), 0, intercept))) - observe[i]) \
                             * wine.iloc[i, :-1][8])/1600
                p9 = p9 + partial_x9
        for i in range(0, 1599):
                partial_x10 = (((intercept + np.sum(
                        np.insert(list(np.multiply(slope, wine.iloc[i, :-1])), 0, intercept))) - observe[i]) \
                             * wine.iloc[i, :-1][9])/1600
                p10 = p10 + partial_x10
        for i in range(0, 1599):
                partial_x11 = (((intercept + np.sum(
                        np.insert(list(np.multiply(slope, wine.iloc[i, :-1])), 0, intercept))) - observe[i]) \
                             * wine.iloc[i, :-1][10])/1600
                p11 = p11 + partial_x11

        partials = [pi,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11]
        return partials

def update_step(partials,intercept,slope):
        learning_rate = 0.01
        lr = []
        for i in range (0,12):
              lr.append(learning_rate)
        step_size = np.multiply(lr,partials)
        # print(step_size)
        newintercept = intercept - step_size[0]
        newslope = np.subtract(slope,step_size[1:])
        return newintercept,newslope

def gradient_decent(intercept,slope,observe, predict):

        for i in range (0,1000):
                partials = get_partials(slope,intercept,observe)
                mse = get_mse(observe,predict)
                intercept,slope = update_step(partials,intercept,slope)
                print(intercept)
                print(slope)
                print("This is iteration: ", i, "The mse is ", mse)
                predict = pre_qual(intercept,slope)

def main():
        intercept = 1.0
        slope = get_slope()
        predict = pre_qual(1,slope)
        # print(predict)
        observe = observed_qual()
        # print(slope)
        # partials = get_partials(slope, intercept, observe)
        # print(partials)
        # newintercept,newslope = update_step(partials,intercept,slope)
        # print(newintercept,newslope)
        gradient_decent(intercept,slope,observe,predict)


if __name__ == '__main__':
        main()