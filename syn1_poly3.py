import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import decimal
from sympy import symbols, diff
s1 = pd.read_csv("synthetic-1.csv",delimiter=",")
def get_slope_2():
        slope1 = float(decimal.Decimal(random.randrange(-30, 30))/100)
        slope2 = float(decimal.Decimal(random.randrange(-30, 30))/100)
        slope3 = float(decimal.Decimal(random.randrange(-30, 30))/100)
        return [slope1,slope2,slope3]


def get_xy():
    x1 = list(s1.iloc[:, 0])
    x2 = np.multiply(list(s1.iloc[:, 0]), list(s1.iloc[:, 0]))
    x3 = np.multiply(x1,x2)
    y = list(s1.iloc[:, 1])
    return [x1,x2,x3], y

def predict_num(x,slope,intercept):
    predict = []
    for i in range(0, 100):
        predict.append(intercept + slope[0] * x[0][i] + slope[1] * x[1][i]+slope[2] * x[2][i])
    return predict

def get_mse(y, predict):
    residual = np.subtract(predict, y)
    square = np.multiply(residual, residual)
    sum = np.sum(square)
    return sum / 100


def get_partials(x, y, slope, intercept):
    p1, p2, p3, pi = 0, 0, 0, 0
    for i in range(0, 100):
        partial_x3 = (slope[0]*x[0][i] + slope[1]*x[1][i] + slope[2]*x[2][i] +intercept-y[i]) * x[2][i]
        p3 = p3 + partial_x3
        partial_x2 = (slope[0] * x[0][i] + slope[1] * x[1][i]+ slope[2]*x[2][i] + intercept - y[i]) * x[1][i]
        p2 = p2 + partial_x2
        partial_x1 = (slope[0] * x[0][i] + slope[1] * x[1][i]+ slope[2]*x[2][i] + intercept - y[i]) * x[0][i]
        p1 = p1 + partial_x1
        partial_xi = (slope[0] * x[0][i] + slope[1] * x[1][i]+ slope[2]*x[2][i] + intercept - y[i])
        pi = pi + partial_xi
    return pi, [p1,p2,p3]

def update_step(pi, p, intercept,slope):
    learning_rate = 0.001
    stepi = pi * learning_rate
    step1 = p[0] * learning_rate
    step2 = p[1] * learning_rate
    step3 = p[2] * learning_rate
    newintercept = intercept - stepi
    newslope1 = slope[0] - step1
    newslope2 = slope[1] - step2
    newslope3 = slope[2] - step3
    return newintercept, [newslope1, newslope2, newslope3]

def gradient_decent(x, y, slope, intercept, predict):
    for i in range(0, 1000):
        pi, p = get_partials(x,y,slope,intercept)
        print(pi, p)
        mse = get_mse(y, predict)
        print(mse)
        intercept, slope = update_step(pi, p, intercept, slope)
        predict = predict_num(x,slope,intercept)
    print("The final slope1 is: ", slope[0], ", slope2 is: ", slope[1],  ", slope3 is: ", slope[2], ", intercept is: ", intercept)

def teo_function(x):
    return -3.9566059201317394+11.14927219219195*x + 0.8770271113243374* x ** 2 - 3.832922908642564 * x**3

def main():
    intercept = 1.5
    slope = get_slope_2()
    xlist,observed = get_xy()
    predict = predict_num(xlist,slope, intercept)
    gradient_decent(xlist, observed, slope, intercept, predict)
    vecfunc = np.vectorize(teo_function)
    d = np.arange(-2, 2, 0.01)
    T = vecfunc(d)
    plt.plot(d, T, 'bo', d, T, 'k')
    plt.plot(xlist[0], observed, 'g+')
    plt.title('sythetic data 1 polynomial 3')
    plt.show()

if __name__ == '__main__':
        main()