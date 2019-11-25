import gym
import numpy as np
import time
from maps import map4,map8,map16,map32,map64,map128,map256 
import pdb
import seaborn as sns
import pandas as pd

def run_episode(env, policy, gamma = 1.0, render = False):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    
    return total_reward



def evaluate_policy(env, policy, gamma = 1.0, n = 1000):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return scores

def compute_value_function(env,policy, gamma=1.0):

    # initialize value table with zeros
    value_table = np.zeros(env.nS)

    # set the threshold
    threshold = 1e-10

    while True:

        # copy the value table to the updated_value_table
        updated_value_table = np.copy(value_table)

        # for each state in the environment, select the action according to the policy and compute the value table
        for state in range(env.nS):
            action = policy[state]

            # build the value table with the selected action
            value_table[state] = sum([trans_prob * (reward_prob + gamma * updated_value_table[next_state])
                        for trans_prob, next_state, reward_prob, _ in env.P[state][action]])

        if (np.sum((np.fabs(updated_value_table - value_table))) <= threshold):
            break

    return value_table



def extract_policy(env,value_table, gamma = 1.0):

    # Initialize the policy with zeros
    policy = np.zeros(env.observation_space.n)

    for state in range(env.observation_space.n):

        # initialize the Q table for a state
        Q_table = np.zeros(env.action_space.n)
        # compute Q value for all ations in the state
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))
        
        
        
        # Select the action which has maximum Q value as an optimal action of the state
        policy[state] = np.argmax(Q_table)
        
    return policy
def policy_iteration(env,gamma = 1.0):

    # Initialize policy with zeros
    old_policy = np.zeros(env.observation_space.n)
    no_of_iterations = 200000

    for i in range(no_of_iterations):

        # compute the value function
        new_value_function = compute_value_function(env,old_policy, gamma)
        # Extract new policy from the computed value function
        new_policy = extract_policy(env,new_value_function, gamma)
       
         
        # Then we check whether we have reached convergence i.e whether we found the optimal
        # policy by comparing old_policy and new policy if it same we will break the iteration
        # else we update old_policy with new_policy

        if (np.all(old_policy == new_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            
            return new_policy,new_value_function,i+1,True 
            break
        old_policy = new_policy

    return new_policy,new_value_function,i,False


maps = [map4,map8,map16,map32,map64,map128]

map_size = [4,8,16,32,64,128]
time_taken = []
conv_it = []
score_1000 = []
policy_final=[]
v_map = []
conv = []
    
     
env = gym.make('FrozenLake-v0',desc=map16)
env.reset()
start = time.time()
policy,value,iters,converged = policy_iteration(env,gamma=.6) 
end = time.time()



bob = np.array(value)

bob3 =bob.reshape(16,16)

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(bob3,  cmap="YlGnBu", annot=False, cbar=False)
plt.savefig('graphs/pi_VMAP16.png')

