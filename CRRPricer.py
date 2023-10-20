import numpy as np


def CRROption(currStockPrice, strikePrice, intRate, vol, totSteps, yearsToExp, american):
    """ BinomialOptionPricer implements a simple binomial model for option pricing
    
        currStockPrice = Current price of the underlying stock
        strikePrice = Strike price of the option
        intRate = Annualized continuous compounding interest rate
        vol = Expected forward annualized stock price volatility
        totSteps = Depth of the tree model, which is N
        yearsToExp = Time to expiry in years, which is T
        american = true or false, i.e. is early excercise allowed (true = American option, false = European)
        option type will be put by given
    
        returns the calculated price of the option
    """

    # calculate the number of time steps that we require, delta t = T/N
    timeStep = yearsToExp / totSteps

    # one step random walk (price increases) as given
    u = np.exp(intRate * timeStep + vol * np.sqrt(timeStep))
    # one step random walk (price decreases) as given
    d = np.exp(intRate * timeStep - vol * np.sqrt(timeStep))

    # risk neutral probability of an up move
    pu = 1 / 2 * (1 - 1 / 2 * vol * np.sqrt(timeStep))
    # risk neutral probability of a down move
    pd = 1 - pu
    # Tree is evaluated in two passes
    # We first calculate the stock price tree using the stochastic dynmaics given
    # Then we calculate the option value tree using the Q-measure with bank account as the numeriare

    priceTree = np.full((totSteps, totSteps), np.nan)  # matrix filled with NaN
    # Only use the top diagonal

    # Initialize with the current stock price
    priceTree[0, 0] = currStockPrice

    for ii in range(1, totSteps):
        # vector calculation of all the up steps
        priceTree[0:ii, ii] = priceTree[0:ii, (ii - 1)] * u

        # vector calculation of all the down steps
        priceTree[ii, ii] = priceTree[(ii - 1), (ii - 1)] * d

        # print("\n", priceTree)

    # Calculate the option value tree
    optionValueTree = np.full_like(priceTree, np.nan)

    # Calculate the terminal value
    optionValueTree[:, -1] = np.maximum(0, strikePrice - priceTree[:, -1])

    # print("\n", optionValueTree)

    backSteps = priceTree.shape[1] - 1  # start with the last column
    early_exercise_index = []
    early_exercise_price = []

    for ii in range(backSteps, 0, -1):
        # We divide the numeraire asset's value at time ii, and then multiply with its value at ii-1 to get the option value at time ii-1
        optionValueTree[0:ii, ii - 1] = np.exp(intRate * (ii / totSteps)) * (
                    pu * (optionValueTree[0:ii, ii] / np.exp(intRate * (ii + 1) / totSteps)) \
                    + pd * (optionValueTree[1:(ii + 1), ii] / np.exp(intRate * (ii + 1) / totSteps)))

        if american:
            # Check whether or not we have early exercise points
            for time_index in range(ii):
                if strikePrice - priceTree[time_index, ii - 1] > optionValueTree[time_index, ii - 1]:
                    early_exercise_index.append(time_index)
                    early_exercise_price.append(priceTree[time_index, ii])
                    break
            # updating the new calculated column with comparison of intrinsic value and hold value
            optionValueTree[0:ii, ii - 1] = np.maximum(strikePrice - priceTree[0:ii, ii - 1],
                                                       optionValueTree[0:ii, ii - 1])
        # print("\n", optionValueTree)

    # get the option price
    optionPrice = optionValueTree[0, 0]

    # reverse the early_exercise_price since we record them backwards
    return priceTree

def get_hedge_strat(priceTree, optionValueTree, strikePrice, intRate, time_list):
    '''
    priceTree: Tree of the asset price
    optionValueTree: Tree of the option value at each time point
    intRate: risk-free interest rate
    time_list: [0, 1/4, 1/2, 3/4, 1]
    Hedging should be that for P + S = C + PV(K), P = C - S + PV(K), 
    so we are replicating the put option with short on stock and long bank account
    '''
    totSteps = priceTree.shape[1]
    alpha = []
    beta = []
    alpha.append((optionValueTree[0, 1] - optionValueTree[1, 1]) / (priceTree[0, 1] - priceTree[1, 1]))
    beta.append((optionValueTree[0, 1] - alpha[0] * priceTree[0, 1]) / np.exp(intRate))
    for index in range(1, len(time_list) - 1):
        time_index = int(time_list[index] * totSteps)
        # no need to add 1 to time_index since python starts index with 0
        alpha.append((optionValueTree[0:time_index, time_index] - optionValueTree[1:(time_index + 1), time_index]) \
                     / (priceTree[0:time_index, time_index] - priceTree[1:(time_index + 1), time_index]))
        beta.append(
            (optionValueTree[0:time_index, time_index] - alpha[index] * priceTree[0:time_index, time_index]) / np.exp(
                intRate * time_list[index]))
    # At expiry date, it would either hold the stock or not
    expiry_index = int(time_list[-1] * totSteps)
    alpha.append([])
    beta.append([])
    for ii in range(expiry_index):
        # Case when we would exercise at the expiry date
        if strikePrice - priceTree[ii, -1] > 0:
            alpha[-1].append(-1)
            beta[-1].append((optionValueTree[ii, -1] - (-1) * priceTree[ii, -1]) / np.exp(intRate * time_list[-1]))
        else:
            alpha[-1].append(0)
            beta[-1].append((optionValueTree[ii, -1]))

    return alpha, beta
