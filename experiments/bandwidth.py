import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

# Points for visualization
x = np.linspace(-5, 5, 400)
y = 0

# Compute RBF kernel values for different bandwidths
sigma_values = [0.5, 1.0, 2.0]
rbf_values = {sigma: [rbf_kernel(np.array([xi]), np.array([y]), sigma) for xi in x] for sigma in sigma_values}

# Plot the RBF kernel for different bandwidths
plt.figure(figsize=(10, 6))
for sigma, values in rbf_values.items():
    plt.plot(x, values, label=f'$\sigma$ = {sigma}')
plt.title('RBF Kernel for Different Bandwidths')
plt.xlabel('x')
plt.ylabel('RBF Kernel Value')
plt.legend()
plt.show()