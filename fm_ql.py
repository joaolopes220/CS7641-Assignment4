from hiive.mdptoolbox import mdp
from hiive.mdptoolbox import example
import numpy as np
import pandas as pd
import time 


def vi_solveMDP(P,R,max_iter = 100000,gamma = 0.96,alpha=0.1, alpha_decay=0.99,
                 epsilon=1.0, epsilon_decay=0.99):
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


    vi = mdp.QLearning(P,R,gamma = gamma,alpha=alpha,alpha_decay=alpha_decay,epsilon=epsilon,epsilon_decay=0.99)
    #vi.setVerbose()
    vi.max_iter = max_iter 
    vi.run()
    return vi



def basic_qlearn():
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
    for pop in POP_CLASSES:

        P,R = example.forest(S=pop)


        start = time.time()
        vi = vi_solveMDP(P,R)
        end = time.time()

        states.append(pop*pop)
        conv_iters.append(0)
        threshold.append(0)
        max_value.append(vi.run_stats[-1]['Max V'])

        bob = pd.DataFrame(vi.run_stats)
        times_api.append(np.mean(bob['Time'].values))
        times_real.append(end-start)

        policies.append(vi.policy)

        policy_updates.append(len(vi.run_stats))



    df = pd.DataFrame(data=list(zip(
        states,conv_iters,threshold,max_value,times_api,times_real,policy_updates,policies)),
            columns = ['states','conv_iters','threshold','max_value','times_api','times_real','policy_updates','policies'])

    df.to_csv('data/ql_fm.csv')


def test_gammas(popsize,filename):


    states = []
    conv_iters = []
    threshold = []
    max_value = []
    times_api = []
    times_real = []
    policies = []
    policy_updates = []
    
    gammas = np.arange(0,.95,.05)
    for g in gammas:

        P,R = example.forest(S=popsize)


        start = time.time()
        vi = vi_solveMDP(P,R,gamma=g)
        end = time.time()

        conv_iters.append(0)
        threshold.append(0)
        max_value.append(vi.run_stats[-1]['Max V'])

        bob = pd.DataFrame(vi.run_stats)
        times_api.append(np.mean(bob['Time'].values))
        times_real.append(end-start)

        policies.append(vi.policy)

        policy_updates.append(len(vi.run_stats))



    df = pd.DataFrame(data=list(zip(
        gammas,conv_iters,threshold,max_value,times_api,times_real,policy_updates,policies)),
            columns = ['gamma','conv_iters','threshold','max_value','times_api','times_real','policy_updates','policies'])

    f = f'data/{filename}'
    df.to_csv(f)


def test_eps(popsize,filename,decay=False):


    states = []
    conv_iters = []
    threshold = []
    max_value = []
    times_api = []
    times_real = []
    policies = []
    policy_updates = []

    epsilons = np.arange(0,.95,.05)

    if decay:
        ed=0.96
    else:
        ed=0.999999

    for e in epsilons:

        P,R = example.forest(S=popsize)


        start = time.time()
        vi = vi_solveMDP(P,R,epsilon=e,epsilon_decay=ed)
        end = time.time()

        conv_iters.append(0)
        threshold.append(0)
        max_value.append(vi.run_stats[-1]['Max V'])

        bob = pd.DataFrame(vi.run_stats)
        times_api.append(np.mean(bob['Time'].values))
        times_real.append(end-start)

        policies.append(vi.policy)

        policy_updates.append(len(vi.run_stats))



    df = pd.DataFrame(data=list(zip(
        epsilons,conv_iters,threshold,max_value,times_api,times_real,policy_updates,policies)),
            columns = ['eps','conv_iters','threshold','max_value','times_api','times_real','policy_updates','policies'])

    f = f'data/{filename}'
    df.to_csv(f)

def test_lr(popsize,filename,decay=False):


    states = []
    conv_iters = []
    threshold = []
    max_value = []
    times_api = []
    times_real = []
    policies = []
    policy_updates = []

    lrs = np.arange(0,.95,.05)

    if decay:
        lr_decay=0.96
    else:
        lr_decay=0.999999

    for lr in lrs:

        P,R = example.forest(S=popsize)


        start = time.time()
        vi = vi_solveMDP(P,R,alpha=lr,alpha_decay=lr_decay)
        end = time.time()

        conv_iters.append(0)
        threshold.append(0)
        max_value.append(vi.run_stats[-1]['Max V'])

        bob = pd.DataFrame(vi.run_stats)
        times_api.append(np.mean(bob['Time'].values))
        times_real.append(end-start)

        policies.append(vi.policy)

        policy_updates.append(len(vi.run_stats))



    df = pd.DataFrame(data=list(zip(
        lrs,conv_iters,threshold,max_value,times_api,times_real,policy_updates,policies)),
            columns = ['learning_rate','conv_iters','threshold','max_value','times_api','times_real','policy_updates','policies'])

    f = f'data/{filename}'
    df.to_csv(f)




if __name__ == "__main__":
    basic_qlearn()
    
    
    
    test_gammas(4,'fm_ql_gamma_m4.csv')
    test_gammas(8,'fm_ql_gamma_m8.csv')
    test_gammas(12,'fm_ql_gamma_m12.csv')
    test_gammas(16,'fm_ql_gamma_m16.csv')

    test_lr(4,'fm_ql_lr_m4nd.csv',decay=False)
    test_lr(8,'fm_ql_lr_m8nd.csv',decay=False)
    test_lr(12,'fm_ql_lr_m12nd.csv',decay=False)
    test_lr(16,'fm_ql_lr_m16nd.csv',decay=False)
    test_lr(4,'fm_ql_lr_m4d.csv',decay=True)
    test_lr(8,'fm_ql_lr_m8d.csv',decay=True)
    test_lr(12,'fm_ql_lr_m12d.csv',decay=True)
    test_lr(16,'fm_ql_lr_m16d.csv',decay=True)
    

    
    test_eps(4,'fm_ql_eps_m4nd.csv',decay=False)
    test_eps(8,'fm_ql_eps_m8nd.csv',decay=False)
    test_eps(12,'fm_ql_eps_m12nd.csv',decay=False)
    test_eps(16,'fm_ql_eps_m16nd.csv',decay=False)
    
    #test_eps(4,'fm_ql_eps_m4d.csv',decay=True)
    #test_eps(8,'fm_ql_eps_m8d.csv',decay=True)
    #test_eps(12,'fm_ql_eps_m12d.csv',decay=True)
    #test_eps(16,'fm_ql_eps_m16d.csv',decay=True)
