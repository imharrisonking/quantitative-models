'''
This module contains a Geometric Brownian Motion class and a class to retrieve stock price data using Yahoo Finance
when passed a stock ticker as a string and a historical start and end date
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import yfinance as yf
from datetime import datetime

# LOGGING CONFIGURATION
format = '%(asctime)s - %(levelname)s - %(message)s - %(name)s'
logging.basicConfig(level = logging.INFO, format=format)

class StockData:
  ''' Class for allowing the user to retrieve historic stock data across a given time period
  Args:
    name = The name of the stock ticker given as string
  '''
  def __init__(self, start_date, end_date):
    has_valid_input = False
    # Keep prompting the user until we get a valid input
    while not has_valid_input:
      # Get the user's input
      string_input = input("Please enter a stock ticker: ")
      
      # Check whether the input is a valid stock ticker string
      if self.is_Valid_Stock_Ticker(string_input):
        # Input is valid, so we can set the flag to True
        has_valid_input = True

      else:
        # Input is not valid, so we need to prompt the user again
        print("Sorry, that is not a valid stock ticker. Please try again: ")
    self._name = string_input

    self._start_Date = start_date
    self._end_Date = end_date
    self._price_Data = self.get_Data()

  @property
  def name(self):
    return self._name
  
  @property
  def start_Date(self):
    return self._start_Date

  @property
  def end_Date(self):
    return self._end_Date
  
  @property
  def long_Name(self):
    return self._price_Data[0]

  @property
  def close_Prices(self):
    return self._price_Data[1]
  
  @property
  def close_Prices_NumPy(self):
    return self.close_Prices.to_numpy()
  
  @property
  def P0(self):
    return self.close_Prices.iloc[0]

  @property
  def days(self):
    return self._price_Data[2]

  @property
  def currency(self):
    return self._price_Data[3] 

  @property
  def years(self):
    date_format = '%Y-%m-%d'
    DAYS_IN_YEAR = 365.2425

    start = datetime.strptime(self.start_Date, date_format)
    end = datetime.strptime(self.end_Date, date_format)
    return (end - start).days / DAYS_IN_YEAR

  @staticmethod
  def is_Valid_Stock_Ticker(string_input):
    # Check whether the string is the correct length
    if len(string_input) < 1 or len(string_input) > 5:
      return False

    # Check whether the string contains only letters and numbers
    if not string_input.isalnum():
      return False
    
    return True
  
  def get_Data(self):
    stockInstance = yf.Ticker(self.name)
    long_Name = stockInstance.info['longName']
    currency = stockInstance.info['currency']
    priceHistory = stockInstance.history(start=self.start_Date, end=self.end_Date, interval='1d')
    days = len(priceHistory)
    logging.info('Getting {0} close price data for {1} business days between {2} and {3}'.format(long_Name, days, self.start_Date, self.end_Date))

    return long_Name, priceHistory['Close'], days, currency

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

# The below code would be run in the Coal, Gas or Carbon module when Stochastic is imported
def main():
    # Get closing stock price data - prompts the user to enter a stock ticker string 
    stock = StockData(start_date='2021-01-01', end_date='2022-01-01')
    pd.DataFrame(stock.close_Prices_NumPy).to_csv('closePrices.csv')

    # Geometric Brownian Motion simulation and visualisation
    gbm = GeometricBrownianMotion(P0=stock.P0, n=stock.days, T=stock.years)
    sim_Results = gbm.run_Path()
    tt = sim_Results[0]
    Pt = sim_Results[1]
    print(tt.shape)
    print(Pt.shape)

    # Plot a random selection of the paths from the GBM Monte-Carlo simulation 
    logging.info(f'Plotting GBM Monte-Carlo simulation for {gbm.M} simulations...')

    plt.figure(figsize=(16, 9)) 
    for i in np.random.choice(np.array(range(gbm.M)), size=50):
      plt.plot(tt[:, i], Pt[:, i], 'b', lw=0.5)
    plt.plot(tt, stock.close_Prices_NumPy, 'r', lw=1)
    plt.xlabel('Years (t)')
    plt.ylabel('Stock price P(t) {0}'.format(stock.currency))
    plt.title("Geometric Brownian Motion of {0} stock price with {1} simulations\n $dP_t = \mu P_t dt + \sigma P_t dW_t$\n $P_0 = {2}, \mu = {3}, \sigma = {4}$".format(stock.long_Name, gbm.M, gbm.P0, gbm.mu, gbm.sigma))
    plt.show()
  
if __name__ == '__main__':
    main()



