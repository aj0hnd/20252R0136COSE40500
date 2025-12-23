import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(func, x):
    h = 1e-4
    return (func(x + h) - func(x - h)) / (2 * h)

def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

if __name__ == '__main__':
    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)
    
    try:
        point = int(input("choose point of f(x) : "))
    except:
        point = 5 # default

    x_max, y_max = 20.0, 7.0
    plt.xlim([0.0, x_max])
    plt.ylim([0.0, y_max])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y, label='original')

    diff = numerical_diff(func=function_1, x=point)
    y2 = diff * (x - point) + function_1(point)
    plt.plot(x, y2, color='r', label='tangent')
    
    plt.scatter(point, function_1(point), color='black')
    plt.axvline(x=point, ymax=function_1(point)/y_max, color='gray', linestyle='--')
    plt.axhline(y=function_1(point), xmax=point/x_max, color='gray', linestyle='--')  
    plt.legend()
    plt.show()
