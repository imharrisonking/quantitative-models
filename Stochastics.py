'''
This module contains a Geometric Brownian Motion class and a class to retrieve stock price data using Yahoo Finance
when passed a stock ticker as a string and a historical start and end date
'''

import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import logging
import yfinance as yf
from datetime import datetime

# LOGGING CONFIGURATION
format = '%(asctime)s - %(levelname)s - %(message)s - %(name)s'
logging.basicConfig(level = logging.INFO, format=format)

class GeometricBrownianMotion:
  ''' Class for a Geometric Brownian Motion process
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
  def __init__(self, P0, n, T, mu=0.08, sigma=0.25, M=1000):
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
  
  def run_Path(self):
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
      + sigma * np.random.normal(0, np.sqrt(dt), size=(n-1,M))
    )

    # Inlcude an initial array of 1's
    Pt = np.vstack([np.ones(M), Pt])

    # Multipe through by P0 and return the cumulative product of elements along a given simulation path (axis=0) to get the commodity price with time
    P0 = self.P0
    Pt = P0 * Pt.cumprod(axis=0)

    # Define the time interval by producing an array of time from 0 to T, with a timestep of n+1
    time = np.linspace(0, T, n)

    # Create numpy array same shape as Pt
    tt = np.full(shape=(M, n), fill_value=time).T

    return tt, Pt
  
class OrnsteinUhlenbeck:
    """
    Class for the an Ornstein-Uhlenbeck process:
    Args:
        theta: long term mean
        sigma: diffusion coefficient
        kappa: mean reversion coefficient
        P0: starting price
        n: number of time steps (e.g 365 days when T = 1 years gives a time step of 1 day)
        T: Time in years
        paths: number of paths
    
    Returns:
        A tuple numpy array with the time of simulation and the commodity forward price curves with Geometric Brownian Motion applied: (tt, Pt)
    """
    def __init__(self, sigma=2, theta=0, kappa=10, n=365, T=1, X0=0, paths=1000):
        if (sigma < 0 or kappa < 0):
            raise ValueError("sigma and kappa must be positive")
        else:
            self._sigma = sigma
            self._kappa = kappa            
        
        self._theta = theta
        self._n = n
        self._T = T
        self._X0 = X0
        self._paths = paths

    @property
    def theta(self):
        return self._theta
    
    @property
    def sigma(self):
        return self._sigma

    @property
    def kappa(self):
        return self._kappa
    
    @property
    def n(self):
        return self._n
    
    @property
    def T(self):
        return self._T

    @property
    def X0(self):
        return self._X0

    @property
    def paths(self):
        return self._paths

    def run_Path(self, X0=0, T=1, n=365, paths=100):
        """ Simulate the price of a commodity using a loop and interating for each time step. Produces a matrix of an Ornstein-Uhlenbeck process: X[n, paths]
        Args:

        Returns:
          A tuple of a matrix of the generated time array, tt and the generated Ornstein-Uhlenbeck process: Pt[n, paths]
        """
        
        tt, dt = np.linspace(0, T, n, retstep=True) 
        X = np.zeros((n, paths))
        X[:,0] = X0
        W = ss.norm.rvs( loc=0, scale=1, size=(n-1, paths))

        std_dt = np.sqrt(self.sigma**2 /(2*self.kappa) * (1-np.exp(-2*self.kappa*dt)))
        for t in range(0,n-1):
            X[t+1, :] = self.theta + np.exp(-self.kappa*dt)*(X[t, :]-self.theta) + std_dt * W[t, :]        
                
        return tt, X

def main():
  ornstein = OrnsteinUhlenbeck()
  sim_Results = ornstein.run_Path()
  tt = sim_Results[0]
  Pt = sim_Results[1]
  logging.info(f'Plotting GBM Monte-Carlo simulation for {ornstein.paths} simulations...')
  plt.figure(figsize=(16, 9))
  plt.plot(tt, Pt)
  plt.xlabel('Years (t)')
  plt.ylabel('Commodity price (Pt)')
  plt.title("Ornstein-Uhlenbeck process of a commodity price with {0} simulations\n $dX_t = \kappa(\u03B8 - X_t)dt + \sigma dW_t$\n $P_0 = {1}, \u03B8 = {2}, \sigma = {3}, \kappa = {4}$".format(ornstein.paths, ornstein.X0, ornstein.theta, ornstein.sigma, ornstein.kappa))
  plt.show()

if __name__ == '__main__':
    main()



