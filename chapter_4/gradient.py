import numpy as np
import matplotlib.pyplot as plt

# just for 1-dim array
def numerical_gradient(func, x: np.ndarray):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        val = x[idx]
        
        x[idx] = val + h
        f_right = func(x)
        x[idx] = val - h
        f_left = func(x)

        grad[idx] = (f_right - f_left) / (2 * h)
        x[idx] = val
    return grad

def gradient_descent(func, x: np.ndarray, lr: float = 0.01, step_num: int = 1000):
    result = []
    for i in range(step_num):
        grad = numerical_gradient(func=func, x=x)
        x = x - lr * grad
        temp_value = func(x)
        result.append(x.tolist() + [temp_value])
    return x, np.array(result)

def function_2(x):
    return x[0]**2 + x[1]**2

if __name__ == '__main__':
    init_x = np.array([-9.0, 8.0])
    optimized_x, logs = gradient_descent(func=function_2, x=init_x, lr=0.01, step_num=1000)

    print(f"Inital input x: {init_x}")
    print(f"Optimized input x: {optimized_x}")
    print(logs)
    step = np.arange(-10, 10, 0.01)
    X, Y = np.meshgrid(step, step)
    Z = function_2([X, Y])

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3)
    ax.scatter(logs[::20, 0], logs[::20, 1], logs[::20, 2], color='r', linewidth=3)

    ax.set_title('training process of numerical gradient')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()
    plt.close()