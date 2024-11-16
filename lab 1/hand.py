import numpy as np
import matplotlib.pyplot as plt

#%%
def quadratic_loss_function(X, Y):
    return X**2 + Y**2

def schaffer_n2_loss_function(x , y):
    return np.sin(x) * np.exp((1 - np.cos(y)) ** 2) + np.cos(y) * np.exp((1 - np.sin(x)) ** 2) + (x - y) ** 2

def loss_function(x, y):
    return quadratic_loss_function(x, y)
'''
    Косяк в вычислении частных производных методом числовой аппроксимации 
'''
#%%
def numerical_partial_derivative(f, x, y, var='x', method='central', h=1e-5):
    """
    Compute numerical partial derivative of a function f(x, y)

    Parameters:
    - f: Function that takes two arguments (x, y)
    - x, y: Point at which to evaluate the partial derivative
    - var: Variable to differentiate with respect to ('x' or 'y')
    - method: Differentiation method ('forward', 'backward', or 'central')
    - h: Step size for approximation

    Returns:
    Approximated partial derivative
    """
    if var == 'x':
        if method == 'forward':
            return (f(x + h, y) - f(x, y)) / h
        elif method == 'backward':
            return (f(x, y) - f(x - h, y)) / h
        else:  # central difference
            return (f(x + h, y) - f(x - h, y)) / (2 * h)
    elif var == 'y':
        if method == 'forward':
            return (f(x, y + h) - f(x, y)) / h
        elif method == 'backward':
            return (f(x, y) - f(x, y - h)) / h
        else:  # central difference
            return (f(x, y + h) - f(x, y - h)) / (2 * h)
    else:
        raise ValueError("var must be 'x' or 'y'")
#%%
def grad_func(x, y):
    d_x = numerical_partial_derivative(loss_function, x, y, var='x')
    d_y = numerical_partial_derivative(loss_function, x, y, var='y')
    return np.array([d_x, d_y])

#%%
def gradient_descent(f, grad_f, x0, y0, learning_rate=0.001, num_iterations=100000):
    x, y = x0, y0
    history = [(x, y, f(x, y))]

    for _ in range(num_iterations):
        dx, dy = grad_f(x, y)
        x = x - learning_rate * dx
        y = y - learning_rate * dy
        z = f(x, y)
        history.append((x, y, z))

    return x, y, history

#%%
def plot_results(history):
    # Convert list of tuples into separate lists for x, y, and loss
    x_vals = [point[0] for point in history]
    y_vals = [point[1] for point in history]
    loss_vals = [point[2] for point in history]

    # Grid for contour plots
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = loss_function(X, Y)

    fig = plt.figure(figsize=(12, 10))

    # First plot (3D surface) in upper left
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(x_vals, y_vals, loss_vals, 'r-', linewidth=2)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Loss')

    # Second plot (3D contour) in upper right
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.contour(X, Y, Z, levels=50)
    ax2.plot(x_vals, y_vals, 'r.-')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Loss')

    # Third plot (2D contour) at the bottom
    ax3 = fig.add_subplot(212)
    ax3.contour(X, Y, Z, levels=50)
    ax3.plot(x_vals, y_vals, 'r.-')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')

    plt.tight_layout()
    plt.show()

#%%
start_points = [
    (4.0, 4.0),
    (-3.0, -3.0),
    (0.0, 0.0),
    (2.0, -2.0),
]

for start_x, start_y in start_points:
    x_min, y_min, history = gradient_descent(loss_function, grad_func, start_x, start_y)
    plot_results(history)
    print(f"Начальная точка: ({start_x}, {start_y})")
    print(f"Минимум найден в точке: ({x_min:.4f}, {y_min:.4f})")
    print(f"Значение функции в минимуме: {loss_function(x_min, y_min):.4f}\n")
