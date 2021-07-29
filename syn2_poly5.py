import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import decimal
from sympy import symbols, diff
s2 = pd.read_csv("synthetic-2.csv",delimiter=",")
def get_slope_2():
        slope1 = float(decimal.Decimal(random.randrange(-20, 20))/100)
        slope2 = float(decimal.Decimal(random.randrange(-20, 20))/100)
        slope3 = float(decimal.Decimal(random.randrange(-20, 20))/100)
        slope4 = float(decimal.Decimal(random.randrange(-20, 20)) / 100)
        slope5 = float(decimal.Decimal(random.randrange(-20, 20)) / 100)
        return [slope1,slope2,slope3,slope4,slope5]


def get_xy():
    x1 = list(s2.iloc[:, 0])
    x2 = np.multiply(list(s2.iloc[:, 0]), list(s2.iloc[:, 0]))
    x3 = np.multiply(x1,x2)
    x4 = np.multiply(x1,x3)
    x5 = np.multiply(x1,x4)
    y = list(s2.iloc[:, 1])
    return [x1,x2,x3,x4,x5], y

def predict_num(x,slope,intercept):
    predict = []
    for i in range(0, 100):
        predict.append(intercept + slope[0] * x[0][i] + slope[1] * x[1][i]+slope[2] * x[2][i]+ slope[3] * x[3][i]+slope[4] * x[4][i])
    return predict

def get_mse(y, predict):
    residual = np.subtract(predict, y)
    square = np.multiply(residual, residual)
    sum = np.sum(square)
    return sum / 100


def get_partials(x, y, slope, intercept):
    p1, p2, p3, p4, p5, pi = 0, 0, 0, 0, 0, 0
    for i in range(0, 100):
        partial_x5 = (slope[0]*x[0][i] + slope[1]*x[1][i] + slope[2]*x[2][i]+ slope[3]*x[3][i] + slope[4] * x[4][i] +intercept - y[i]) * x[4][i]
        p5 = p5 + partial_x5
        partial_x4 = (slope[0] * x[0][i] + slope[1] * x[1][i] + slope[2] * x[2][i] + slope[3] * x[3][i] + slope[4] *
                      x[4][i] + intercept - y[i]) * x[3][i]
        p4 = p4 + partial_x4
        partial_x3 = (slope[0] * x[0][i] + slope[1] * x[1][i] + slope[2] * x[2][i] + slope[3] * x[3][i] + slope[4] *
                      x[4][i] + intercept - y[i]) * x[2][i]
        p3 = p3 + partial_x3
        partial_x2 = (slope[0] * x[0][i] + slope[1] * x[1][i] + slope[2] * x[2][i] + slope[3] * x[3][i] + slope[4] *
                      x[4][i] + intercept - y[i]) * x[1][i]
        p2 = p2 + partial_x2
        partial_x1 = (slope[0] * x[0][i] + slope[1] * x[1][i] + slope[2] * x[2][i] + slope[3] * x[3][i] + slope[4] *
                      x[4][i] + intercept - y[i]) * x[0][i]
        p1 = p1 + partial_x1
        partial_xi = (slope[0] * x[0][i] + slope[1] * x[1][i] + slope[2] * x[2][i] + slope[3] * x[3][i] + slope[4] *
                      x[4][i] + intercept - y[i])
        pi = pi + partial_xi

    return pi, [p1,p2,p3,p4,p5]

def update_step(pi, p, intercept,slope):
    learning_rate = 0.0001
    stepi = pi * learning_rate
    step1 = p[0] * learning_rate
    step2 = p[1] * learning_rate
    step3 = p[2] * learning_rate
    step4 = p[3] * learning_rate
    step5 = p[4] * learning_rate
    newintercept = intercept - stepi
    newslope1 = slope[0] - step1
    newslope2 = slope[1] - step2
    newslope3 = slope[2] - step3
    newslope4 = slope[3] - step4
    newslope5 = slope[4] - step5
    return newintercept, [newslope1, newslope2, newslope3, newslope4, newslope5]

def gradient_decent(x, y, slope, intercept, predict):
    for i in range(0, 10000):
        pi, p = get_partials(x,y,slope,intercept)
        print(pi, p)
        mse = get_mse(y, predict)
        print(mse)
        intercept, slope = update_step(pi, p, intercept, slope)
        predict = predict_num(x,slope,intercept)
    print("The final slope1 is: ", slope[0], ", slope2 is: ", slope[1], ", slope3 is: ", slope[2],
          ", slope4 is: ", slope[3], ", slope5 is: ", slope[4], ", intercept is: ", intercept)

def teo_function(x):
    return 0.4849634201571061 -0.36258587292335454 * x  -0.5302531430643763 * x ** 2  +0.31373139937715117 * x ** 3 + \
           0.1107395213835789* x**4 -0.06330736235589908 *x**5

def main():
    intercept = 1.5
    slope = get_slope_2()
    xlist,observed = get_xy()
    predict = predict_num(xlist,slope, intercept)
    gradient_decent(xlist, observed, slope, intercept, predict)
    vecfunc = np.vectorize(teo_function)
    d = np.arange(-2.2, 2.2, 0.01)
    T = vecfunc(d)
    plt.plot(d, T, 'bo', d, T, 'k')
    plt.plot(xlist[0], observed, 'g+')
    plt.title('sythetic data 2 polynomial 5')
    plt.show()


if __name__ == '__main__':
        main()