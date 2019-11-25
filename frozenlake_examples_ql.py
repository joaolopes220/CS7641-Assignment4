from time import sleep
import numpy as np
import gym
import time
from maps import map4,map8,map16,map32,map64,map128,map256
import pandas as pd



# Environment
def compare_states_size(env,eps=0.2,epsilonDecay=0.999999,lr=0.75,lrDecay=0.9999,gamma=0.99,iters=100000):

    inputCount = env.observation_space.n
    actionsCount = env.action_space.n

    # Init Q-Table
    Q = {}
    for i in range(inputCount):
        Q[i] = np.random.rand(actionsCount)

    # Hyperparameters
    lr = lr 
    lrMin = 0.1
    lrDecay = lrDecay 
    gamma = gamma 
    epsilon = eps 
    epsilonMin = 0.001
    epsilonDecay = epsilonDecay 
    episodes = iters 
    # Training
    for i in range(episodes):
        print("Episode {}/{}".format(i + 1, episodes))
        s = env.reset()
        done = False

        while not done:
            if np.random.random() < epsilon:
                a = np.random.randint(0, actionsCount)
            else:
                a = np.argmax(Q[s])

            newS, r, done, _ = env.step(a)
            
            if done and r!=1:
                r = -10 
            elif r ==0:
                r = -0.1
            
            Q[s][a] = Q[s][a] + lr * (r + gamma * np.max(Q[newS]) - Q[s][a])
            s = newS

            if lr > lrMin:
                lr *= lrDecay

            if not r<=0 and epsilon > epsilonMin:
                epsilon *= epsilonDecay


    print("")
    print("Learning Rate :", lr)
    print("Epsilon :", epsilon)

    return env,Q


def get_score(env,Q):
    # Testing
    print("\nPlay Game on 100 episodes...")

    avg_r = 0
    tr = 0
    for i in range(100):
        s = env.reset()
        done = False
        policy = []
        while not done:
            a = np.argmax(Q[s])
            newS, r, done, _ = env.step(a)
            s = newS
            policy.append(a)
        avg_r += r/100.0
    print("Average reward on 100 episodes :", avg_r)
    return avg_r,policy

maps = [map4,map8,map16,map32,map64,map128]

map_size = [4,8,16,32,64,128]
time_taken = []
policy_final=[]
Q_map = []
Avg_score = []
total_score = []

env = gym.make('FrozenLake-v0',desc=map8,is_slippery=False)
env.reset()
env,Q = compare_states_size(env)
avg_r,policy = get_score(env,Q)


bob = []
for i in Q:
    bob.extend(Q[i])

bob2 = np.array(bob)
bob3 =bob2.reshape(64,4)

print(bob3)
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(bob3,  cmap="YlGnBu", annot=False, cbar=True)
plt.savefig('graphs/qlQmap88.png')


