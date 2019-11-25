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

for m in maps:
    env = gym.make('FrozenLake-v0',desc=m,is_slippery=False)
    env.reset()
    start = time.time()
    env,Q = compare_states_size(env)
    end = time.time()

    avg_r,policy = get_score(env,Q)

    time_taken.append(end-start)
    policy_final.append(policy)
    Q_map.append(Q)
    Avg_score.append(avg_r)



df = pd.DataFrame(data=list(zip(
        map_size,time_taken,Avg_score,policy_final,Q_map)),
            columns = ['map_size','time','average_score','policy','Q_map'])
df.to_csv('data/frozenlake_qlearner.csv')

def test_gamma(map_env,filename):
    map_size = [4,8,16,32,64,128]
    time_taken = []
    policy_final=[]
    Q_map = []
    Avg_score = []
    total_score = []
    gammas = np.arange(0.05,1.0,0.05)
    for g in gammas:
        env = gym.make('FrozenLake-v0',desc=map_env,is_slippery=False)
        env.reset()
        start = time.time()
        env,Q = compare_states_size(env,eps=0.2,epsilonDecay=0.999999,lr=0.75,lrDecay=0.9999,gamma=g,iters=100000)
        end = time.time()

        avg_r,policy = get_score(env,Q)

        time_taken.append(end-start)
        policy_final.append(policy)
        Q_map.append(Q)
        Avg_score.append(avg_r)

    df = pd.DataFrame(data=list(zip(
            gammas,time_taken,Avg_score,policy_final,Q_map)),
                columns = ['gammas','time','average_score','policy','q_map'])
    f = f'data/{filename}'
    df.to_csv(f)

def test_epsilon(map_env,filename,decay=False):
    map_size = [4,8,16,32,64,128]
    time_taken = []
    policy_final=[]
    Q_map = []
    Avg_score = []
    total_score = []
    if decay:
        epsilonDecay=0.95
    else:
        epsilonDecay=1.0

    epsilons = np.arange(0.05,1.0,0.05)
    for e in epsilons:
        env = gym.make('FrozenLake-v0',desc=map_env,is_slippery=False)
        env.reset()
        start = time.time()
        env,Q = compare_states_size(env,eps=e,epsilonDecay=epsilonDecay,iters=100000)
        end = time.time()

        avg_r,policy = get_score(env,Q)

        time_taken.append(end-start)
        policy_final.append(policy)
        Q_map.append(Q)
        Avg_score.append(avg_r)

    df = pd.DataFrame(data=list(zip(
            epsilons,time_taken,Avg_score,policy_final,Q_map)),
                columns = ['eps','time','average_score','policy','q_map'])
    f = f'data/{filename}'
    df.to_csv(f)


def test_lr(map_env,filename,decay=False):
    map_size = [4,8,16,32,64,128]
    time_taken = []
    policy_final=[]
    Q_map = []
    Avg_score = []
    total_score = []
    if decay:
        lrDecay=0.95
    else:
        lrDecay=1.0

    lrs= np.arange(0.05,1.0,0.05)
    for lr in lrs:
        env = gym.make('FrozenLake-v0',desc=map_env,is_slippery=False)
        env.reset()
        start = time.time()
        env,Q = compare_states_size(env,lr=lr,lrDecay=lrDecay,iters=100000)
        end = time.time()

        avg_r,policy = get_score(env,Q)

        time_taken.append(end-start)
        policy_final.append(policy)
        Q_map.append(Q)
        Avg_score.append(avg_r)

    df = pd.DataFrame(data=list(zip(
            lrs,time_taken,Avg_score,policy_final,Q_map)),
                columns = ['lrs','time','average_score','policy','q_map'])
    f = f'data/{filename}'
    df.to_csv(f)
test_gamma(map8,'frozenlake_gammas8.csv')
test_gamma(map32,'frozenlake_gammas32.csv')
test_gamma(map32,'frozenlake_gammas64.csv')


test_epsilon(map8,'frozenlake_eps8nd.csv',decay=False)
test_epsilon(map32,'frozenlake_eps32nd.csv',decay=False)
test_epsilon(map64,'frozenlake_eps64nd.csv',decay=False)

test_epsilon(map8,'frozenlake_eps8d.csv',decay=True)
test_epsilon(map32,'frozenlake_eps32d.csv',decay=True)
test_epsilon(map64,'frozenlake_eps64d.csv',decay=True)
test_lr(map8,'frozenlake_lr8nd.csv',decay=False)
test_lr(map32,'frozenlake_lr32nd.csv',decay=False)
test_lr(map64,'frozenlake_lr64nd.csv',decay=False)

test_lr(map8,'frozenlake_lr8d.csv',decay=True)
test_lr(map32,'frozenlake_lr32d.csv',decay=True)
test_lr(map64,'frozenlake_lr64d.csv',decay=True)
