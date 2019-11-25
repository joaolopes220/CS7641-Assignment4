from hiive.mdptoolbox import mdp
from hiive.mdptoolbox import example
import numpy as np
import pandas as pd
import time 


def vi_solveMDP(P,R,max_iter = 10000,gamma = 0.96):
    """Solve the problem as a finite horizon Markov decision process.

    The optimal policy at each stage is found using backwards induction.
    Possingham and Tuck report strategies for a 50 year time horizon, so the
    number of stages for the finite horizon algorithm is set to 50. There is no
    discount factor reported, so we set it to 0.96 rather arbitrarily.

    Returns
    -------
    mdp : mdptoolbox.mdp.FiniteHorizon
        The PyMDPtoolbox object that represents a finite horizon MDP. The
        optimal policy for each stage is accessed with mdp.policy, which is a
        numpy array with 50 columns (one for each stage).

    """

    vi = mdp.PolicyIteration(P,R,gamma = gamma,max_iter=100)

    #vi.setVerbose()
    vi.max_iter = 2000000 
    vi.run()
    return vi

def solve_all():
    MAX_POP = 36
    MIN_POP = 3

    POP_CLASSES = np.arange(MIN_POP,MAX_POP,1)

    states = []
    conv_iters = []
    max_value = []
    times_api = []
    times_real = []
    policies = []
    policy_updates = []
    V = []
    Stats = []
    mean_reward= []
    for pop in POP_CLASSES:

        P,R = example.forest(S=pop)


        start = time.time()
        vi = vi_solveMDP(P,R)
        end = time.time()

        states.append(pop*pop)
        conv_iters.append(vi.iter)
        max_value.append(vi.run_stats[-1]['Max V'])

        bob = pd.DataFrame(vi.run_stats)
        times_api.append(np.sum(bob['Time'].values))
        times_real.append(end-start)

        policies.append(vi.policy)
        stats = vi.run_stats
        policy_updates.append(len(vi.run_stats))
        V.append(vi.V)
        Stats.append(stats)


    df = pd.DataFrame(data=list(zip(
        states,conv_iters,max_value,times_api,times_real,policy_updates,policies,V,Stats)),
            columns = ['states','conv_iters','max_value','times_api','times_real','policy_updates','policies','V','all_stats'])


    df.to_csv('data/pi_fm.csv')

def solve_gammas(map_size, filename):



    gammas = np.arange(0.05,1.00,0.05)
    states = []
    conv_iters = []
    max_value = []
    times_api = []
    times_real = []
    policies = []
    policy_updates = []
    V = []
    Stats = []
    mean_reward= []
    for g in gammas:

        P,R = example.forest(S=map_size)


        start = time.time()
        vi = vi_solveMDP(P,R,gamma=g)
        end = time.time()
        print('testing')
        conv_iters.append(vi.iter)
        max_value.append(vi.run_stats[-1]['Max V'])

        bob = pd.DataFrame(vi.run_stats)
        times_api.append(np.sum(bob['Time'].values))
        times_real.append(end-start)

        policies.append(vi.policy)
        stats = vi.run_stats
        policy_updates.append(len(vi.run_stats))
        V.append(vi.V)
        Stats.append(stats)


    df = pd.DataFrame(data=list(zip(
        gammas,conv_iters,max_value,times_api,times_real,policy_updates,policies,V,Stats)),
            columns = ['gammas','conv_iters','max_value','times_api','times_real','policy_updates','policies','V','all_stats'])


    df.to_csv(f'data/{filename}')

if __name__ == "__main__":
    
    solve_all()
    solve_gammas(4,'fm_pi_gammas4.csv')
    solve_gammas(8,'fm_pi_gammas8.csv')
    solve_gammas(12,'fm_pi_gammas12.csv')
    solve_gammas(16,'fm_pi_gammas16.csv')
    solve_gammas(20,'fm_pi_gammas20.csv')
    solve_gammas(24,'fm_pi_gammas24.csv')
