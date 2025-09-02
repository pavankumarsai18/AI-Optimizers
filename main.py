from gradient_descent import *
from stochastic_gradient_descent import *
import time
import matplotlib.pyplot as plt

def run_gradient_descent(gradient_descent_func, small_dataset, large_dataset):
    start_time = time.time()
    theta_bgd_small, losses_small = gradient_descent_func(small_dataset["x"], small_dataset["y"])
    end_time = time.time()
    print(f"time taken is {end_time - start_time} sec")
    
    start_time = time.time()
    theta_bgd_large, losses_large = gradient_descent_func(large_dataset["x"], large_dataset["y"])
    end_time = time.time()
    print(f"time taken is {end_time - start_time} sec")
    return losses_small, losses_large
    
def main():
    small_dataset = generate_data(100)
    large_dataset = generate_data(100_000_000)

    losses_small_stoch, losses_large_stoch = run_gradient_descent(stochastic_gradient_descent, small_dataset, large_dataset)
    losses_small_batch, losses_large_batch = run_gradient_descent(batch_gradient_descent, small_dataset, large_dataset)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(losses_small_batch,
    label="Batch GD (Small Dataset)")
    plt.plot(losses_small_stoch, label="Stochastic GD (Small Dataset)")
    plt .xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend( )
    plt. title("Small Dataset")
    plt.subplot(1, 2, 2)
    plt.plot(losses_large_batch, label="Batch GD (Large Dataset)")
    plt.plot(losses_large_stoch, label="Stochastic GD (Large Dataset)")
    plt. xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Large Dataset")    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()