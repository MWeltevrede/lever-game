from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


DEFAULT_PAYOFFS = [0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1]
DEFAULT_N_AGENTS = 2


class LeverGame():
    """
    Description:
        Two agents need to coordinate by choosing the same lever 
        from a set of 10 available levers.
    
    Source:
        The environment is defined in Hu, Hengyuan, et al. ""Other-Play" for Zero-Shot Coordination." 
        arXiv preprint arXiv:2003.02979 (2020)
        
    Actions:
        Type: tuple(2)
        Idx    Observation                 Type      Min        Max
        0      First Player's Choice       int       0          9
        1      Second Player's Choice      int       0          9
    
    Observation:
        Returns the action taken as an observation.
        
    Reward:
        If both players choose the same lever, the reward equals the payoff associated with that lever.
        If the players choose different levers, the reward is 0
    """
    
    def __init__(self, lever_payoffs=DEFAULT_PAYOFFS):
        self.n_levers = len(lever_payoffs)
        self.payoffs = lever_payoffs
        
        self.n_agents = DEFAULT_N_AGENTS
        
        self.action_space = set(product(list(range(self.n_levers)), repeat=self.n_agents))
        
        self.last_action = None
        
    def step(self, action):
        assert action in self.action_space, "%r (%s) is an invalid action" % (action, type(action))
        
        if action[0] == action[1]:
            reward = self.payoffs[action[0]]
        else:
            reward = 0
            
        self.last_action = action
        return action, reward, True, dict()
    
    def render(self):
        player_choices = [[0 for l in range(self.n_levers)] for p in range(self.n_agents)]
        
        if self.last_action:
            for c,v in enumerate(self.last_action):
                player_choices[c][v] = c+1

        fig, ax = plt.subplots(figsize=(4,4./5), dpi=100, subplot_kw={'yticks': range(self.n_agents), 'yticklabels': ['Player 1', 'Player 2'], 'xticks': range(self.n_levers), 'xticklabels': self.payoffs})
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xlabel('Lever payoffs')

        x = np.arange(-.5, 10, 1)
        y = np.arange(-.5, 2, 1)
        cmap = ListedColormap([[1.,1.,1.], [255/255.,131/255.,120/255.], [124/255.,202/255.,226/255.]])
        ax.pcolormesh(x, y, player_choices, cmap=cmap, vmin=0, vmax=self.n_agents, edgecolors='black', linewidths=1)
    
        
    def reset(self):
        self.last_action = None