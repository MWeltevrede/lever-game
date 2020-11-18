from itertools import product

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
        If the players choose differenrt levers, the reward is 0
    """
    
    def __init__(self, lever_payoffs=[0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1]):
        self.n_levers = len(lever_payoffs)
        self.payoffs = lever_payoffs
        
        self.n_agents = 2
        
        self.action_space = set(product(list(range(self.n_levers)), repeat=self.n_agents))
        
    def step(self, action):
        assert action in self.action_space, "%r (%s) is an invalid action" % (action, type(action))
        
        if action[0] == action[1]:
            reward = self.payoffs[action[0]]
        else:
            reward = 0
            
        return action, reward, False, dict()
    
        
        