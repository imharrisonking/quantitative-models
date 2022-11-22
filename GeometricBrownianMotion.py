'''
This module contains classes which each define a stochastic process.
These stochastic processes are implemented for simulating commodity prices.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import yfinance as yf

# LOGGING CONFIGURATION
format = '%(asctime)s - %(levelname)s - %(message)s - %(name)s'
logging.basicConfig(level = logging.INFO, format=format)

# def get_Stock(stock):


class GeometricBrownianMotion:
  ''' Apply Geometric Brownian Motion stochastic process to a foward price curve
  Args:
    mu: The drift coeffienct 0 >= mu >= 2 [default = 0.1]
    n: The number of time steps in a year [default = 365.2425 for day time resolution]
    T: The total time period to be modelled in years [default = 1]
    M: The number of simulations [default = 5000]
    P0: The initial commodity price [default = 100]
    sigma: Measure of volatility [default = 0.3]
  
  Returns:
    The above arguements when their methods are called
  '''
  def __init__(self, P0=6.78, mu=0.1, n=365, T=1, M=5000, sigma=0.3):
    if mu > 2 or mu < 0:
      raise ValueError('Mu drift coefficient must be between 0 and 2')
    self._P0 = P0
    self._mu = mu
    self._n = n
    self._T = T
    self._M = M
    self._sigma = sigma
  
  @property
  def mu(self):
    return self._mu
  
  @property
  def n(self):
    return self._n

  @property
  def T(self):
    return self._T

  @property
  def M(self):
    return self._M

  @property
  def P0(self):
    return self._P0

  @property
  def sigma(self):
    return self._sigma
  
  def run_Simulation(self):
    ''' Simulate the commodity price directly and multiple the exponential terms together at each time step
    Args:
      
    Returns:
      A tuple numpy array with the time of simulation and the commodity forward price curves with Geometric Brownian Motion applied: (tt, Pt)
    '''
    # Log initialisation
    logging.info('Running Geometric Brownian Motion Monte-Carlo simulation...')

    # Calculate each time step
    T = self.T
    n = self.n

    dt = T/n

    # Simulation using numpy arrays where Pt is the commodity price at time t
    # Randomly sample from the browian motion normal distribution with a size of time step (n) * number of simulations (M)
    # Take the transpose of this to get the simulation for each time step
    mu = self.mu
    sigma = self.sigma
    M = self.M

    Pt = np.exp(
      (mu - (sigma**2) / 2) * dt
      + sigma * np.random.normal(0, np.sqrt(dt), size=(n,M))
    )

    # Inlcude an initial array of 1's
    Pt = np.vstack([np.ones(M), Pt])

    # Multipe through by P0 and return the cumulative product of elements along a given simulation path (axis=0) to get the commodity price with time
    P0 = self.P0
    Pt = P0 * Pt.cumprod(axis=0)

    # Define the time interval by producing an array of time from 0 to T, with a timestep of n+1
    time = np.linspace(0, T, n+1)

    # Create numpy array same shape as Pt
    tt = np.full(shape=(M, n+1), fill_value=time).T

    return tt, Pt

# The below code would be run in the Coal, Gas or Carbon module when Stochastic is imported
def main():
    # Geometric Brownian Motion simulation and visualisation
    gbm = GeometricBrownianMotion()
    sim_Results = gbm.run_Simulation()
    tt = sim_Results[0]
    Pt = sim_Results[1]

    # Plot the results of the GBM Monte-Carlo simulation 
    logging.info(f'Plotting GBM Monte-Carlo simulation for {gbm.M} simulations...')
    plt.figure(figsize=(16, 9)) 
    plt.plot(tt, Pt)
    plt.xlabel('Years (t)')
    plt.ylabel('Commodity price P(t)')
    plt.title(
      "Geometric Brownian Motion of a commodity price with {0} simulations\n $dP_t = \mu P_t dt + \sigma P_t dW_t$\n $P_0 = {1}, \mu = {2}, \sigma = {3}$".format(gbm.M, gbm.P0, gbm.mu, gbm.sigma)
    )
    
    plt.show()
  
if __name__ == '__main__':
    main()



