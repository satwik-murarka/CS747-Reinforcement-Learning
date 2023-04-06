"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
# You can use this space to define any helper functions that you need
def KL(x,y):
    if x == 1:
        return x*math.log((x/y)+1e-9)
    elif x == 0:
        return (1-x)*math.log((1-x)/(1-y)+1e-9)
    else:
        return x*math.log((x/y)+1e-9) + (1-x)*math.log((1-x)/(1-y)+1e-9)
# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.time = 0
        self.num_arms = int(num_arms)
        self.counts = np.zeros(num_arms)
        self.emp_mean = np.zeros(num_arms)
        self.ucb = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        if np.sum(self.counts) < self.num_arms:
            return int(np.sum(self.counts))
        else:
            return np.argmax(self.ucb)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        arm_index = int(arm_index)
        prev_mean = self.emp_mean[arm_index]
        n = self.counts[arm_index]
        self.time += 1
        self.counts[arm_index] += 1
        self.emp_mean[arm_index] = (prev_mean*(n) + reward)/(n+1)
        for arm in range(self.num_arms):
            if(self.counts[arm]!=0):
                self.ucb[arm] = self.emp_mean[arm]  + np.sqrt(2*math.log(self.time)/self.counts[arm])
            else:
                continue
        # END EDITING HERE
        pass

class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.time = 0
        self.c = 3
        self.constant = 0
        self.num_arms = int(num_arms)
        self.counts = np.zeros(num_arms)
        self.emp_mean = np.zeros(num_arms)
        self.ucb = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        if np.sum(self.counts) < self.num_arms:
            return int(np.sum(self.counts))
        else:
            return np.argmax(self.ucb)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        prev_mean = self.emp_mean[arm_index]
        n = self.counts[arm_index]
        self.time += 1
        self.counts[arm_index] += 1
        self.emp_mean[arm_index] = (prev_mean*(n) + reward)/(n+1)
        if(self.time>self.num_arms):
            self.constant = math.log(self.time) + (self.c)*math.log(math.log(self.time))
        for arm in range(self.num_arms):
            if((self.counts[arm]!=0) and (self.time>self.num_arms)):
                emp_mean = self.emp_mean[arm]
                self.ucb[arm] = 1
                q = 0.99
                i =  emp_mean
                while(i<=q):
                    mid = (i+q)/2
                    if(self.counts[arm]*KL(emp_mean,mid) == self.constant):
                        q = mid
                        break
                    elif self.counts[arm]*KL(emp_mean,mid)< self.constant:
                        i = mid + 0.01
                    else:
                        q = mid - 0.01
                self.ucb[arm] = q
            else:
                continue
        # END EDITING HERE


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.num_arms = num_arms
        self.true = np.zeros(num_arms)
        self.false = np.zeros(num_arms)
        self.samples = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        return np.argmax(self.samples)
        # START EDITING HERE
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if reward == 1:
            self.true[arm_index] +=1
        else:
            self.false[arm_index] +=1
        for arm in range(self.num_arms):
            self.samples[arm] = np.random.beta(self.true[arm]+1,self.false[arm]+1)
        # END EDITING HERE
