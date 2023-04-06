import numpy as np
import argparse
state_file = 'data/states.txt'
param_file = 'data/cricket/sample-p1.txt'
states = []
params = {}
with open(state_file) as f:
    data = f.readlines()
    for line in data:
        line = line.strip("\n")
        states.append(line)
with open(param_file) as f:
    data = f.readlines()
    data = data[1:]
    for line in data:
        line = line.strip("\n")
        str_list = line.split()
        params[str_list[0]] = [float(x) for x in str_list[1:]]
print(states)
print(params)
for state in states:
    for action in [0,1,2,4,6]:
        pass
        



parser = argparse.ArgumentParser()
parser.add_argument("--states", type=str)
parser.add_argument("--parameters", type=str)
parser.add_argument('--q', type=float)
args = parser.parse_args()
