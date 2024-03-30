import numpy as np
import matplotlib.pyplot as plt
import torch


# the following function generates a path of n assets with drift mu and volatility matrix sigma
def generate_asset_path(mu, sigma, T, dt):
    n = len(mu)
    N_steps =  int(T/dt)

    # generate multivariate normal random variables

    dW = np.random.multivariate_normal(mean=np.zeros(n), cov=np.eye(n), size=N_steps)
    random = np.matmul(sigma, dW.T).T
    random = random * np.sqrt(dt)
    drift = mu * dt
    path = np.zeros((n, N_steps))
    path[:, 0] = np.zeros(n)
    for i in range(1, N_steps):
        path[:, i] = path[:, i-1] + drift + random[i-1]
    return path

############################################################################################################



class Functional:
    def __init__(self, value_network):
        """
        Initialize the class with a given value network.
        
        Parameters:
        - value_network: a PyTorch neural network model used for calculating the functional's value and its derivatives.
        """
        self.value_network = value_network

    def value(self, x):
        """
        Calculate the value of the functional.

        Parameters:
        - x: torch.Tensor or np.array, the path of wealth
        """
        x = self._ensure_tensor(x)
        return self.value_network(x).item()
    
    def partial_t(self, x, dt):
        """
        Calculate the partial derivative of the functional with respect to time.

        Parameters:
        - x: torch.Tensor or np.array, the path of wealth
        - dt: float, the small change in time
        """
        x = self._ensure_tensor(x)
        new_x = torch.cat((x, x[-1].unsqueeze(0)))
        new_output = self.value_network(new_x)
        output = self.value_network(x)
        partial_t = (new_output - output) / dt
        return partial_t.detach().numpy()

    def partial_x(self, x, h):
        """
        Calculate the first partial derivative of the functional with respect to x.

        Parameters:
        - x: torch.Tensor or np.array, the path of wealth
        - h: float, the small change in x
        """
        x = self._ensure_tensor(x)
        original_x_last = x[-1].item()  # Store original value to reset later
        output = self.value_network(x)
        x[-1] += h
        new_output = self.value_network(x)
        partial_x = (new_output - output) / h
        x[-1] = original_x_last  # Reset to original last value
        return partial_x.detach().numpy()

    def partial_xx(self, x, h):
        """
        Calculate the second partial derivative of the functional with respect to x.

        Parameters:
        - x: torch.Tensor or np.array, the path of wealth
        - h: float, the small change in x
        """
        x = self._ensure_tensor(x)
        original_x_last = x[-1].item()  # Store original value to reset later
        output = self.value_network(x)
        x[-1] += h
        output_plus_h = self.value_network(x)
        x[-1] = original_x_last - h
        output_minus_h = self.value_network(x)
        partial_xx = (output_plus_h - 2 * output + output_minus_h) / (h ** 2)
        x[-1] = original_x_last  # Reset to original last value
        return partial_xx.detach().numpy()

    def _ensure_tensor(self, x):
        """
        Helper method to ensure the input x is a PyTorch tensor.

        Parameters:
        - x: Input which can be a numpy array or a PyTorch tensor.
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        return x
    


    ############################################################################################################

    # the following is the implementation of the trading policy 
    # define a class of trading policy 
# the trading policy object

class Trading_Policy:
    def __init__(self, f, states):
        # f: is a functional object
        # states: should be M * n matrix, where M is the number of states, n is the number of assets
        self.states = states
        self.f = f

    def distribution_numerator(self, x, mu, sigma, holdings, dt, h, gamma):
        # x is the sequence (path of the asset prices)
        # mu is the expected return of the assets
        # sigma is the volatilty matrix
        # holdings is the current holdings of the assets
        # dt is the time step for partial_t
        # h is the increment for the partial_xx
        # gamma is the risk aversion coefficient

        x = self._ensure_tensor(x)
        par_t = self.f.partial_t(x, dt)
        par_x = self.f.partial_x(x, h)
        par_xx = self.f.partial_xx(x, h)

        # now, par_t, par_x, par_xx are all numpy arrays
        # mu, sigma are also numpy arrays

        first_term = np.matmul(mu, holdings.T) * par_t
        cov = np.matmul(sigma, sigma.T)
        second_term = 0.5 * np.matmul(np.matmul(holdings, cov), holdings.T) * par_xx

        inverse_gamma = 1 / gamma
        return inverse_gamma * (first_term + second_term)

    def distribution(self, x, mu, sigma, dt, h, gamma):
        
        # dist has length same as the number of states[0]
        dist = np.zeros(len(self.states))
        for i in range(len(self.states)):
            holdings = self.states[i]
            dist[i] = self.distribution_numerator(x, mu, sigma, holdings, dt, h, gamma)

        # we have to softmax the dist, and normalize it
        dist = np.exp(dist) / np.sum(np.exp(dist))
        return dist
    
    def action(self, x, mu, sigma, dt, h, gamma):
        # output the holdings
        dist = self.distribution(x, mu, sigma, dt, h, gamma)
        # randomly choose an index from the distribution
        index = np.random.choice(len(self.states), 1, p=dist)
        # return the index of the state
        holding = self.states[index]

        # let holding be an 1-dimensional array
        holding = holding[0]
        return holding
    

    def _ensure_tensor(self, x):
        """
        Helper method to ensure the input x is a PyTorch tensor.

        Parameters:
        - x: Input which can be a numpy array or a PyTorch tensor.
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        return x
    

    ############################################################################################################

    # the following function to simulate the trading process
def simulated_trading(policy, path, intial_wealth, mu, sigma, dt, h, gamma):
    # policy is a trading policy object
    # path is a simulated asset path of dimension (T/dt, n) 
    # n is the number of assets

    # initialize the wealth to be 1
    # meaning the initial cash is 1
    
    # this function is to simulate the trading process
    # return the holding history and wealth history

    wealth_history = np.zeros(len(path[0]))
    holding_history = np.zeros((path.shape[1], len(path)))

    wealth_history[0] = intial_wealth

    for i in range(1, len(path[0])):
        x = wealth_history[:i]
        holding = policy.action(x, mu, sigma, dt, h, gamma)
        holding_history[i] = holding
        
        # to compute the wealth history 
        increment = path[:, i] - path[:, i - 1]
        wealth_change = np.matmul(holding, increment)
        wealth_history[i] = wealth_history[i - 1] + wealth_change

    return holding_history, wealth_history