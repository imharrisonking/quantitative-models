import numpy as np
import matplotlib.pyplot as plt

'''
 PARAMETERS
'''
# Drift coefficent (10% over a year long horizon)
mu = 0.1
# Number of steps
n = 100
# Time in years
T = 1
# Number of simulations
M = 100
# Initial commodity price
P0 = 100
# Volatility (30% volatility)
sigma = 0.3

'''
SIMULATION:
Here we simulate the commodity price directly throughout the simulation and multiply the exponential terms together
at each time step
'''
# Calculate each time step
dt = T/n

# Simulation using numpy arrays where St is the commodity price at time t
# Randomly sample from the browian motion normal distribution with a size of time step (n) * number of simulations (M)
# Take the transpose of this to get the simulation for each time step
Pt = np.exp(
  (mu - (sigma**2) / 2) * dt
  + sigma * np.random.normal(0, np.sqrt(dt), size=(M,n))
)

# Inlcude an initial array of 1's
Pt = np.vstack([np.ones(M), Pt])

# Multipe through by S0 and return the cumulative product of elements along a given simulation path (axis=0)
Pt = P0 * Pt.cumprod(axis=0)

# CONSIDER THE TIME INTERVAL IN YEARS
# Define time interval correctly, from 0 to time T, with a timestep of n+1. This will produce an array of time
time = np.linspace(0, T, n+1)

# Require numpy array that is the same shape as St
tt = np.full(shape=(M,n+1), fill_value=time).T

'''
VISUALISATION
'''
plt.plot(tt, Pt)
plt.xlabel('Years (t)')
plt.ylabel('Commodity price P(t)')
plt.title(
    "Geometric Brownian Motion of a commodity price\n $dP_t = \mu P_t dt + \sigma P_t dW_t$\n $P_0 = {0}, \mu = {1}, \sigma = {2}$".format(P0, mu, sigma)
)
plt.show()
