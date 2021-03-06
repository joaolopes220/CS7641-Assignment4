import gym
import numpy as np
import time
from maps import map4,map8,map16,map32,map64,map128,map256
import pandas as pd
#partial soource: https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa

def run_episode(env, policy, gamma = 1.0, render = False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.
    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
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


def evaluate_policy(env, policy, gamma = 1.0,  n = 100):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
            run_episode(env, policy, gamma = gamma, render = False)
            for _ in range(n)]
    return np.mean(scores)


def value_iteration(env, gamma = 1.0):

    # initialize value table with zeros
    value_table = np.zeros(env.observation_space.n)

    # set number of iterations and threshold
    no_of_iterations = 100000
    threshold = 1e-20

    for i in range(no_of_iterations):

        # On each iteration, copy the value table to the updated_value_table
        updated_value_table = np.copy(value_table)

        # Now we calculate Q Value for each actions in the state
        # and update the value of a state with maximum Q value

        for state in range(env.observation_space.n):
            Q_value = []
            for action in range(env.action_space.n):
                next_states_rewards = []
                for next_sr in env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    next_states_rewards.append((trans_prob * (reward_prob + gamma * updated_value_table[next_state])))

                Q_value.append(np.sum(next_states_rewards))

            value_table[state] = max(Q_value)

        # we will check whether we have reached the convergence i.e whether the difference
        # between our value table and updated value table is very small. But how do we know it is very
        # small? We set some threshold and then we will see if the difference is less
        # than our threshold, if it is less, we break the loop and return the value function as optimal
        # value function

        if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
             print ('Value-iteration converged at iteration# %d.' %(i+1))
             return value_table, i+1,True
             break

    return value_table,i+1,False 


def extract_policy(env,value_table, gamma = 1.0):

    # initialize the policy with zeros
    policy = np.zeros(env.observation_space.n)


    for state in range(env.observation_space.n):

        # initialize the Q table for a state
        Q_table = np.zeros(env.action_space.n)

        # compute Q value for all ations in the state
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))

        # select the action which has maximum Q value as an optimal action of the state
        policy[state] = np.argmax(Q_table)

    return policy

maps = [map4,map8,map16,map32,map64,map128]

map_size = [4,8,16,32,64,128]
time_taken = []
conv_it = []
score_1000 = []
policy_final=[]
v_map = []
conv = []
for m in maps:


    env = gym.make('FrozenLake-v0',desc=m,is_slippery=False)
    env.reset()
        
    start = time.time()
    optimal_value_function,iters, converged = value_iteration(env,gamma=0.6)
    policy = extract_policy(env,optimal_value_function, gamma=0.6)
    end = time.time()
    
    score  = evaluate_policy(env, policy, gamma=1.0, n=1000)

    s = np.mean(score)
    time_taken.append(end-start)
    conv_it.append(iters)
    score_1000.append(s)
    policy_final.append(policy)
    v_map.append(optimal_value_function)
    conv.append(converged)

df = pd.DataFrame(data=list(zip(
        map_size,time_taken,conv_it,score_1000,policy_final,conv,v_map)),
            columns = ['map_size','time','iters','avg_score','policy','converge','v_map'])
df.to_csv('data/frozenlake_value_iteration_gamma6_slip.csv')


def test_gammas(map_env,save_file):
    gammas = np.arange(0.05,0.95,.05)

    time_taken = []
    conv_it = []
    #score_1000 = []
    policy_final=[]
    v_map = []
    conv = []
    print(str(map_env))
    for g in gammas:

        env = gym.make('FrozenLake-v0',desc=map_env,is_slippery=False)
        env.reset()

        start = time.time()
        
        optimal_value_function,iters, converged = value_iteration(env,gamma=g)
        policy = extract_policy(env,optimal_value_function, gamma=g)
        print('testing')
        end = time.time()
        #score = evaluate_policy(env,policy,g)
        #s = np.mean(score)
        time_taken.append(end-start)
        conv_it.append(iters)
        #score_1000.append(s)
        policy_final.append(policy)
        v_map.append(optimal_value_function)
        conv.append(converged)

        print(f'policy: {policy}')
    df = pd.DataFrame(data=list(zip(
            gammas,time_taken,conv_it,policy_final,conv,v_map)),
                columns = ['gammas','time','iters','policy','converge','v_map'])
    df.to_csv(save_file)


test_gammas(map4,'data/frozenlake_value_iteration_gamma_map4_slip.csv')
test_gammas(map8,'data/frozenlake_value_iteration_gamma_map8_slip.csv')
test_gammas(map16,'data/frozenlake_value_iteration_gamma_map16_slip.csv')
test_gammas(map32,'data/frozenlake_value_iteration_gamma_map32_slip.csv')
#test_gammas(map64,'data/frozenlake_value_iteration_gamma_map64.csv')
                                                                
