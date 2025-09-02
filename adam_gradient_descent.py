import numpy as np
import matplotlib.pyplot as plt


def quadratic_loss(x, y):
    return x**2 + 10*y**2

def quadratic_grad(x, y):
    dx = 2*x
    dy = 20*y

    return np.array([dx, dy])



def batch_gradient_descent(loss_func, gradient_func, eta, epochs, start_point):
    x, y = start_point
    path = [(x, y)]

    losses = [loss_func(x, y)]

    for _ in range(epochs):
        grad = gradient_func(x, y)
        x -= eta*grad[0]
        y -= eta*grad[1]
        path.append((x, y))
        losses.append(loss_func(x, y))
    
    return np.array(path), losses


def momentum_gradient_descent(loss_func, gradient_func, eta, beta, epochs, start_point):
    x, y = start_point
    v = np.array([0, 0])
    path = [(x, y)]
    losses = [loss_func(x, y)]

    for _ in range(epochs):
        grad = gradient_func(x, y)
        v = beta*v + (1 - beta) * grad

        x -= eta*v[0]
        y -= eta*v[1]

        path.append((x, y))
        losses.append(loss_func(x, y))
    
    return np.array(path), losses

def adam_gradient_descent(loss_func, gradient_func, eta, beta1, beta2, epochs, start_point):
    x, y = start_point
    path = [(x, y)]
    m = np.array([0.0, 0.0])
    v = np.array([0.0, 0.0])
    losses = [loss_func(x, y)]
    epsilon = 10**-8

    for t in range(1, epochs + 1):
        grad = gradient_func(x, y)
        m = beta1* m + (1 - beta1) * grad
        v = beta2* v + (1 - beta2) * (grad**2)

        m_hat = m / (1 - (beta1 ** t))
        v_hat = v / (1 - (beta2 ** t))

        x -= eta*m_hat[0] /(np.sqrt(v_hat[0] ) + epsilon)
        y -= eta*m_hat[1] /(np.sqrt(v_hat[1] ) + epsilon)

        path.append((x, y))
        losses.append(loss_func(x, y))
    
    return np.array(path), losses


def plot_paths(function, paths, labels, title):
    X, Y = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 2, 400))
    Z = function(X, Y)
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50, cmap='jet')

    # Only label Start and End once
    first = True
    for path, label in zip(paths, labels):
        plt.plot(path[:, 0], path[:, 1], label=label)
        plt.scatter(path[0, 0], path[0, 1], color='green', label="Start")
        plt.scatter(path[-1, 0], path[-1, 1], color='red', label="End")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def plot_losses(losses, labels, title):
    plt.figure(figsize=(8, 6))
    for loss, label in zip(losses, labels):
        plt.plot(loss,label=label)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def main():
    eta_bgd = 0.1
    eta_momentum = 0.1
    beta1 = 0.90
    beta2 = 0.999
    epochs = 50
    start_point = (1.5, 1.5)


    path_bgd, losses_bgd = batch_gradient_descent(quadratic_loss, quadratic_grad, eta_bgd, epochs, start_point)
    path_rms, losses_rms = adam_gradient_descent(quadratic_loss, quadratic_grad, eta_momentum, beta1, beta2, epochs, start_point)

    plot_paths(quadratic_loss, [path_bgd, path_rms],
            ["Batch Gradient Descent", "ADAM"] ,
            "Oscillations in BGD vs ADAM ")
    
    plot_losses([losses_bgd, losses_rms], ["Batch Gradient Descent",
            "ADAM" ] ,
            "Loss vs Epoch for  BGD & ADAM")
    

if __name__ == "__main__":
    main()