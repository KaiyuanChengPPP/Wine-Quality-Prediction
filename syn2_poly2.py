import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import decimal
from sympy import symbols, diff
s2 = pd.read_csv("synthetic-2.csv",delimiter=",")
def get_slope_2():
        slope1 = float(decimal.Decimal(random.randrange(-30, 30))/100)
        slope2 = float(decimal.Decimal(random.randrange(-30, 30))/100)
        return slope1,slope2

def get_x1_x2_y():
        x1 = list(s2.iloc[:,0])
        x2 = np.multiply(list(s2.iloc[:,0]),list(s2.iloc[:,0]))
        y = list(s2.iloc[:,1])
        return x1,x2,y

def predict_num(x1,x2,slope1,slope2,intercept):
        predict = []
        for i in range(0,100):
                predict.append(intercept+slope1*x1[i]+slope2*x2[i])
        return predict

def get_mse(y,predict):
        residual = np.subtract(predict,y)
        square = np.multiply(residual,residual)
        sum = np.sum(square)
        return sum/100

def get_partials(x1,x2,y,slope1,slope2,intercept):
        p1,p2,pi = 0,0,0
        for i in range(0,100):
                partial_x2 = (slope1*x1[i]+slope2*x2[i]+intercept-y[i])*x2[i]
                p2 = p2+partial_x2
                partial_x1 = (slope1 * x1[i] + slope2 * x2[i] + intercept - y[i]) * x1[i]
                p1 = p1 + partial_x1
                partial_xi = (slope1 * x1[i] + slope2 * x2[i] + intercept - y[i])
                pi = pi+ partial_xi
        return pi,p1,p2

def update_step(pi,p1,p2,intercept,slope1,slope2):
        learning_rate = 0.005
        stepi = pi * learning_rate
        step1 = p1 * learning_rate
        step2 = p2 * learning_rate
        newintercept = intercept - stepi
        newslope1 = slope1 - step1
        newslope2 = slope2 - step2
        return newintercept,newslope1,newslope2

def gradient_decent(x1,x2,y,slope1,slope2,intercept,predict):
        for i in range (0,1000):
                pi,p1,p2 = get_partials(x1,x2,y,slope1,slope2,intercept)
                print(pi,p1,p2)
                mse = get_mse(y,predict)
                print(mse)
                intercept,slope1,slope2 = update_step(pi,p1,p2,intercept,slope1,slope2)
                predict = predict_num(x1,x2,slope1,slope2,intercept)
        print("The final slope1 is: ", slope1, ", slope2 is: ", slope2, ", intercept is: ", intercept)
def teo_function(x):
        return 0.3702494372870867-0.0478094705500752*x + -0.17884705518681787* x ** 2

def main():
        intercept =1.5
        slope1,slope2 = get_slope_2()
        x1,x2,y = get_x1_x2_y()
        print(intercept,slope1,slope2)
        predict = predict_num(x1,x2,slope1,slope2,intercept)
        pi,p1,p2 = get_partials(x1,x2,y,slope1,slope2,intercept)
        # newintercept, newslope1, newslope2 = update_step(pi,p1,p2,intercept,slope1,slope2)
        # print(newintercept, newslope1, newslope2)
        gradient_decent(x1, x2, y, slope1, slope2, intercept, predict)
        vecfunc = np.vectorize(teo_function)
        d = np.arange(-2, 2, 0.01)
        T = vecfunc(d)
        plt.plot(d, T, 'bo', d, T, 'k')
        plt.plot(x1, y, 'g+')
        plt.title('sythetic data 2 polynomial 2')
        plt.show()
if __name__ == '__main__':
        main()