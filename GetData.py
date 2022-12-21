'''
This module defines classes which retrieve data and methods which can return or graph this data
'''

import logging
import yfinance as yf
from datetime import datetime

# LOGGING CONFIGURATION
format = '%(asctime)s - %(levelname)s - %(message)s - %(name)s'
logging.basicConfig(level = logging.INFO, format=format)

class StockData:
  ''' Class for allowing the user to retrieve historic stock data across a given time period
  Args:
    start_date = The start date to retrieve stock data from in the format YYYY-MM-DD
    end_date = The end date to retrieve stock data from in the format YYYY-MM-DD
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