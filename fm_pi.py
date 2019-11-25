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


    vi = mdp.PolicyIteration(P,R,gamma = 0.96,max_iter=100)
    #vi.setVerbose()
    vi.max_iter = max_iter 
    vi.run()
    return vi






if __name__ == "__main__":
    
    
    MAX_POP = 36 
    MIN_POP = 3

    POP_CLASSES = np.arange(MIN_POP,MAX_POP,1)

    states = []
    conv_iters = []
    threshold = []
    max_value = []
    times_api = []
    times_real = []
    policies = []
    policy_updates = []
    stats = [] 
    V = []
    for pop in POP_CLASSES:

        P,R = example.forest(S=pop)

        print(type(P[1]))
        print(R.shape)
        start = time.time()
        vi = vi_solveMDP(P,R)
        end = time.time()
        
        states.append(pop*pop)
        conv_iters.append(vi.iter)
        threshold.append(0)
        max_value.append(vi.run_stats[-1]['Max V'])

        bob = pd.DataFrame(vi.run_stats)
        times_api.append(np.sum(bob['Time'].values))
        times_real.append(end-start)

        policies.append(vi.policy)

        policy_updates.append(len(vi.run_stats)) 
        V.append(vi.V)
        stats.append(vi.run_stats) 
    
    df = pd.DataFrame(data=list(zip(
        states,conv_iters,threshold,max_value,times_api,times_real,policy_updates,policies)),
            columns = ['states','conv_iters','threshold','max_value','times_api','times_real','policy_updates','policies'])


    df.to_csv('pi_fm.csv')
