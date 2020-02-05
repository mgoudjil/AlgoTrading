
from random import randint,random,seed
from operator import add
from functools import reduce

def individual(length, min, max):

    'Create a portoflio weights'

    return [ randint(min,max) for x in range(length) ] #=> Create portfolio

def population(count, length, min, max):
    """
    Create a number of portfolios (i.e. a population).

    """
    return [ individual(length, min, max) for x in range(count) ]

def fitness(individual, target):
    """
    Determine the profit of a portfolio. Higher is better.
    """

    sum = portfolio_opt(individual)

    return abs(target-sum)

def grade(pop, target):
    'Find average fitness for a population.'
    summed = reduce(add, (fitness(x, target) for x in pop))
    return summed / (len(pop) * 1.0)

def evolve(pop, target, retain=0.2, random_select=0.05, mutate=0.01):

    graded = [ (fitness(x, target), x) for x in pop]
    graded = [ x[1] for x in sorted(graded)]
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]

    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)

    # mutate some individuals
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)
            individual[pos_to_mutate] = randint(
                min(individual), max(individual))
        print("WTF %s"%(min(individual)))
        print(max(individual))

    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) / 2
            child = male[:int(half)] + female[int(half):]
            children.append(child)        
    parents.extend(children)


    return parents

def portfolio_opt(indiv):

    stoploss_coeff = indiv[0]
    weight_coeff = indiv[1]

    stop_loss = 0.005 * stoploss_coeff
    weight = 5 * weight_coeff
    port = launch_backtesting(df_price,df_output,Portfolio(100000),stop_loss, weight)
    profit = port.historical_positions['netto_profit'].sum()
    
    return profit

from crypto_trading import *
from genetic import *
from pandas import read_csv

save_global = []
target = 10000
p_count = 5
i_length = 2
i_min = 0
i_max = 100

coin_name = "btc"
df_price = read_csv('./backtesting/%s_price.csv'%(coin_name), header=0, index_col=0)
df_output = read_csv('./backtesting/%s_output.csv'%(coin_name), header=0, index_col=0) 


p = population(p_count, i_length, i_min, i_max)
fitness_history = [grade(p, target),]

for i in range(20):
    p = evolve(p, target)
    fitness_history.append(grade(p, target))


