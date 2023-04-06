import numpy as np
import pulp as p1
import argparse

def fileOutput(file_name):
    mdp = {}
    with open(file_name) as f:
        data = f.readlines()
    for line in data:
        line = line.strip("\n")
        str_list = line.split()
        x = str_list[0]
        if x == 'numStates':
            mdp["num_states"] = int(str_list[1])
        elif x == 'numActions':
            mdp["num_actions"] = int(str_list[1])
            reward = np.zeros((mdp["num_states"],mdp["num_actions"],mdp["num_states"]))
            transition = np.zeros((mdp["num_states"],mdp["num_actions"],mdp["num_states"]))
            end  = list()
        elif x == 'mdptype':
            mdp["type"] = str_list[1]
        elif x == 'discount':
            mdp["gamma"] = np.double(str_list[1])
        elif x == 'transition':
            reward[int(str_list[1])][int(str_list[2])][int(str_list[3])] = np.double(str_list[4])
            transition[int(str_list[1])][int(str_list[2])][int(str_list[3])] = np.double(str_list[5])
        elif x == 'end':
            end = [int(x) for x in str_list[1:]]
    return mdp, reward,transition

def vi_algo(mdp,reward,transition):
    seed = 42
    np.random.seed(seed)
    v = np.random.rand(mdp["num_states"])
    while(1):
        v_old = v
        v = (np.max(np.sum(transition*(reward+mdp["gamma"]*v_old),axis=-1),axis=-1))
        if np.allclose(v, v_old, rtol=0, atol=1e-10):
                break
    p = np.argmax(np.sum(transition*(reward+mdp["gamma"]*v_old),axis=-1),axis=-1)
    return v,p


def lp_algo(mdp,reward,transition):
    v = [int(x) for x in range(mdp["num_states"])]
    model = p1.LpProblem("MDP_LP",p1.LpMaximize)
    v = p1.LpVariable.dicts("v",v).values()
    v = np.array(list(v))
    model += p1.lpSum(-v)
    for s in range(mdp["num_states"]):
            for a in range(mdp["num_actions"]):
                model += v[s] >= p1.lpSum(transition[s, a,:]*(reward[s, a,:] + mdp["gamma"]*v))
    model.solve(p1.PULP_CBC_CMD(msg=0))
    v = np.array(list(map(p1.value, v)))
    policy = np.argmax(np.sum(transition*(reward + mdp["gamma"] * v), axis=-1), axis=-1)
    return v, policy

def hpi_algo(mdp,reward,transition):
    pi = np.zeros(mdp["num_states"],dtype=int)
    while(1):
        policy_old = pi
        T = transition[np.arange(mdp["num_states"]), policy_old]
        R = reward[np.arange(mdp["num_states"]), policy_old]
        a = np.eye(mdp["num_states"]) - mdp["gamma"]*T
        b = np.sum(T*R,axis=-1)
        v = np.linalg.solve(a,b)
        pi = np.argmax(np.sum(transition*(reward+mdp["gamma"]*v),axis=-1),axis=-1)
        if (policy_old==pi).all():
                break
    return v,pi

def out(v,pi):
    for idx in range(len(v)):
        print(f"{v[idx]:.6f} {pi[idx]}\n")

def policy_eval(policy_file,mdp,reward,transition):
    p = []
    with open(policy_file) as f:
        data = f.readlines()
    for line in data:
        line = line.strip("\n")
        p.append(line)
    p = [int(x) for x in p]
    T = transition[np.arange(mdp["num_states"]), p]
    R = reward[np.arange(mdp["num_states"]), p]
    a = np.eye(mdp["num_states"]) - mdp["gamma"]*T
    b = np.sum(T*R,axis=-1)
    v = np.linalg.solve(a,b)
    return v,p

parser = argparse.ArgumentParser()
parser.add_argument("--mdp", type=str)
parser.add_argument("--algorithm", type=str, required=False, default="vi")
parser.add_argument('--policy', type=str, required=False)
args = parser.parse_args()

file_name = args.mdp
mdp, reward,transition = fileOutput(file_name)
if args.algorithm == "vi" and args.policy == None:
    v,p = vi_algo(mdp,reward,transition)
    out(v,p)
elif args.algorithm == "lp":
    v,p = lp_algo(mdp,reward,transition)
    out(v,p)
elif args.algorithm == "hpi":
    v,p = hpi_algo(mdp,reward,transition)
    out(v,p)

if args.policy:
    policy_file = args.policy
    mdp, reward,transition = fileOutput(file_name)
    v_pi,p_pi = policy_eval(policy_file,mdp, reward,transition)
    out(v_pi,p_pi)
