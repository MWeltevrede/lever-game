import numpy as np

class LinearDecay():
    """
    Description:
        Basic implementation of a linearly decaying epsilon
    """
    def __init__(self, epsilon_start, epsilon_end, decayed_by):
        self.start = epsilon_start
        self.end = epsilon_end
        self.decayed = decayed_by
    
    def get_epsilon(self, step):
        p = np.max([self.start - (self.start - self.end) * (step / self.decayed), self.end])
        return p