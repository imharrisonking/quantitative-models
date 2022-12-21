'''
This program carries out a Monte-Carlo simulation of a stock price when passed a stock ticker at the command line. Geometric Brownian Motion is used to model
the stock price stochastically.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

import GetData
import Stochastics 

# LOGGING CONFIGURATION
format = '%(asctime)s - %(levelname)s - %(message)s - %(name)s'
logging.basicConfig(level = logging.INFO, format=format)

def main():
  # Get closing stock price data - prompts the user to enter a stock ticker string 
  stock = GetData.StockData(start_date='2021-01-01', end_date='2022-01-01')
  pd.DataFrame(stock.close_Prices_NumPy).to_csv('closePrices.csv')

  # Geometric Brownian Motion simulation and visualisation
  gbm = Stochastics.GeometricBrownianMotion(P0=stock.P0, n=stock.days, T=stock.years)
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