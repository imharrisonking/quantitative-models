import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import logging

# LOGGING CONFIGURATION
format = '%(asctime)s - %(levelname)s - %(message)s - %(name)s'
logging.basicConfig(level = logging.INFO, format=format)

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
