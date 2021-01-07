import numpy as np


DEFAULT_DISCOUNT = 0.9
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_EPSILON = 0.05


class QLearning():
    """
    Description:
        Basic implementation of the Q-learning algorithm with epsilon-greedy action selection.
    """
    def __init__(self, num_states, num_actions, discount=DEFAULT_DISCOUNT, learning_rate=DEFAULT_LEARNING_RATE):
        # initialize Q-values to 0
        self.Q_values = np.zeros((num_states, num_actions))
        self.gamma = discount
        self.alpha = learning_rate
        self.num_actions = num_actions
        
    def update_q_values(self, state, action, next_state, reward, done):
        """
            Update the Q-values based on state, action, next state and reward
        """
        # Following the Q-Learning update rule
        if done:
            self.Q_values[state,action] = (1 - self.alpha) * self.Q_values[state,action] + self.alpha * (reward)
        else:
            self.Q_values[state,action] = (1 - self.alpha) * self.Q_values[state,action] + self.alpha * (reward + self.gamma * np.amax(self.Q_values[next_state]))
        
        
    def select_action(self, state):
        """
            Select action using epsilon-greedy action selection based on the current Q-values
        """
        if np.random.random() > DEFAULT_EPSILON:
            # greedy action selection
            
            # check if there are multiple equivalent greedy actions
            if sum(self.Q_values[state] == np.amax(self.Q_values[state])) > 1:
                # select one of the greedy actions randomly
                idxs = np.where(self.Q_values[state] == np.amax(self.Q_values[state]))[0]
                return idxs[np.random.randint(0, idxs.size)]
            else:
                # return the unique greedy action
                return np.argmax(self.Q_values[state])
        else:
            # random action selection
            return np.random.randint(0, self.num_actions)