import numpy as np
import matplotlib.pyplot as plt

def show_sgd_drawback():
    x, y = np.arange(-10, 10, step=0.05), np.arange(-10, 10, step=0.05)
    X, Y = np.meshgrid(x, y)
    Z = 0.05 * X ** 2 + Y ** 2

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_title('3D shape of f(x, y)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')

    fig.suptitle('Difficulty of SGD')

    ax = fig.add_subplot(1, 2, 2)
    ax.contour(X, Y, Z, levels=50, cmap='viridis', linewidths=1)
    ax.set_title('Contour of f(x, y)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.show()

show_sgd_drawback()