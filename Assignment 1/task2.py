"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

You need to complete the following methods:
    - give_pull(self): This method is called when the algorithm needs to
        select the arms to pull for the next round. The method should return
        two arrays: the first array should contain the indices of the arms
        that need to be pulled, and the second array should contain how many
        times each arm needs to be pulled. For example, if the method returns
        ([0, 1], [2, 3]), then the first arm should be pulled 2 times, and the
        second arm should be pulled 3 times. Note that the sum of values in
        the second array should be equal to the batch size of the bandit.
    
    - get_reward(self, arm_rewards): This method is called just after the
        give_pull method. The method should update the algorithm's internal
        state based on the rewards that were received. arm_rewards is a dictionary
        from arm_indices to a list of rewards received. For example, if the
        give_pull method returned ([0, 1], [2, 3]), then arm_rewards will be
        {0: [r1, r2], 1: [r3, r4, r5]}. (r1 to r5 are each either 0 or 1.)
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need.
# END EDITING HERE

class AlgorithmBatched:
    def __init__(self, num_arms, horizon, batch_size):
        self.num_arms = num_arms
        self.horizon = horizon
        self.batch_size = batch_size
        assert self.horizon % self.batch_size == 0, "Horizon must be a multiple of batch size"
        # START EDITING HERE
        # Add any other variables you need here
        self.time = 0
        self.i = 0
        self.emp_mean = np.zeros(num_arms)
        self.counts = np.zeros(num_arms)
        self.true = np.zeros(num_arms)
        self.false = np.zeros(num_arms)
        self.samples = np.zeros(num_arms)
        self.eps = 1/(self.time*self.batch_size+1)
        self.ucb = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        if self.batch_size*self.time > np.power(self.horizon,0.5):
            return [np.argmax(self.emp_mean)],[self.batch_size]
        else:
            return [idx for idx in np.random.randint(0,self.num_arms,self.batch_size)],np.ones(self.batch_size,dtype=int)
        # if self.batch_size*self.time>self.num_arms:
        #     return [np.argmax(self.ucb)],[self.batch_size]
        # else:
        #     idx = []
        #     for id in range(self.time*self.batch_size,(self.time+1)*self.batch_size):
        #         if(id<self.num_arms):
        #             idx.append(id)
        #         else:
        #             idx.append(np.argmax(self.ucb))
        #     return idx,np.ones(self.batch_size,dtype=int)
                

        # END EDITING HERE
    
    def get_reward(self, arm_rewards):
        # START EDITING HERE
        self.time += 1
        for arm_index in arm_rewards:
            prev_mean = self.emp_mean[arm_index]
            n = self.counts[arm_index]
            self.time += 1
            self.counts[arm_index] += len(arm_rewards[arm_index])
            if(self.counts[arm_index]!=0):
                self.emp_mean[arm_index] = (prev_mean*(n) + np.sum(arm_rewards[arm_index]))/self.counts[arm_index]
            # for arm in range(self.num_arms):
            #     if(self.counts[arm]!=0):
            #         self.ucb[arm] = self.emp_mean[arm]  + np.sqrt(2*np.log(self.time*self.batch_size)/self.counts[arm])
            #     else:
            #         continue
        # for arm in arm_rewards:
        #     for i in arm_rewards[arm]:
        #         if i == 1:
        #             self.true[arm] +=1
        #         else:
        #             self.false[int(arm)] +=1
        # for arm in range(self.num_arms):
        #     self.samples[arm] = np.random.beta(self.true[arm]+1,self.false[arm]+1)
        # END EDITING HERE