import numpy as np
import matplotlib.pyplot as plt

def generate_OU():
  # Use a seed for creating random data so that it is repeatable
  np.random.seed(0)

  theta = 1
  mu = 0.2
  sigma = 0.3

  t = np.linspace(0, 364, 365)
  dt = t[1] - t[0]

  X = np.zeros(t.shape)
  X[0] = 1.1

  # Create an Ornstein-Uhlenbeck process
  for i in range( t.size-1 ):
      X[i+1] = X[i] + mu * (theta - X[i]) * dt + sigma * np.sqrt(dt) * np.random.normal()
      
  plt.plot(t, X)
  plt.grid(True)
  plt.xlabel('t')
  plt.ylabel('X')
  plt.show()

  print(f'True theta: {theta}')
  print(f'True mu: {mu}')
  print(f'True sigma: {sigma}')

  return X

def max_Likelihood_Estimation_OU(X):
  ''' Function to find the optimal parameters for an Ornstein-Uhlenbeck process by carrying out Maximum Likelihood Estimation (MLE) when passed an array of historical data
  Args:
  - X: 1-D NumPy array of historical data to calibrate with
  Returns:
  - mu, sigma, theta
  ''' 
  N = X.size
  Xx  = np.sum(X[:-1])
  Xy  = np.sum(X[1:])
  Xxx = np.sum(X[:-1]**2)
  Xyy = np.sum(X[1:]**2)
  Xxy = np.sum(X[:-1] * X[1:])
  dt = 1

  # Find optimal theta
  theta0 = (Xy * Xxx - Xx * Xxy) / (N * (Xxx - Xxy) - (Xx**2 - Xx * Xy))

  mu0 = (Xxy - (theta0 * Xx) - (theta0 * Xy) + (N * theta0**2)) / (Xxx - (2 * theta0 * Xx) + (N * theta0**2))
  mu0 = -(1 / dt) * np.log(mu0)

  exponent = np.exp(-mu0 * dt)

  first_factor = 2 * mu0 / (N * (1 - np.exp(-2 * mu0 * dt)))
  second_factor = Xyy - (2 * exponent * Xxy) + (np.exp(-2 * mu0 * dt) * Xxx) - (2 * theta0 *(1 - exponent)) * (Xy - exponent * Xx) + (N * theta0**2 * (1 - exponent)**2)

  sigma0 = np.sqrt(first_factor * second_factor)

  print(f'MLE theta*: {theta0}')
  print(f'MLE mu*: {mu0}')
  print(f'MLE sigma*: {sigma0}')

def main():
  # First create a random OU process
  OU_data = generate_OU()

  # Then test the Maximum Likelihood Estimation function by passing the randomly generated OU NumPy array to see how far off the estimated parameters are from the true values
  max_Likelihood_Estimation_OU(OU_data)

if __name__ == '__main__':
  main()
