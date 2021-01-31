import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.ndimage.filters import uniform_filter1d

from agents.qlearning import QLearning

class Logger():
    """
    Description:
        A logging class that stores and plots the training and test scores for zero-shot coordination games.
    """
    def __init__(self, population_size, total_episodes):
        self.total_episodes = total_episodes
        self.population_size = population_size
        self.rewards_train = np.zeros((population_size, total_episodes))
        self.rewards_test = np.zeros((total_episodes, population_size, population_size-1))
        
        self.include_test_graph = False
        
    def update_training_log(self, iteration, episode, reward):
        """
            Store the training score
        """
        self.rewards_train[iteration, episode] = reward
        
    def update_testing_log(self, episode, rewards_matrix):
        """
            Store the test score in the form of a cross-play reward matrix
        """
        if not self.include_test_graph:
            self.include_test_graph = True
        
        # remove the diagonal (the self-play pay-offs)
        diag = np.eye(self.population_size, dtype=bool)
        rewards_matrix = rewards_matrix[~diag].reshape(self.population_size, self.population_size-1)
        
        self.rewards_test[episode] = rewards_matrix
    
    
    def show_results(self, smoothness=None):
        """
            Plot training and/or testing scores with provided smoothness
        """
        cmap = matplotlib.cm.get_cmap('Set2')
        
        if not self.include_test_graph:
            fig, axs = plt.subplots(dpi=100, figsize=[6,6])

            mean = self.rewards_train.mean(axis=0)
            sem = self.rewards_train.std(axis=0) / (self.population_size**.5)

            # Smooth the graph with moving average over the mean of iterations
            if smoothness is not None:
                mean = uniform_filter1d(mean, size=smoothness, origin=int((smoothness-1)/2), mode='nearest')
                sem = uniform_filter1d(sem, size=smoothness, origin=int((smoothness-1)/2), mode='nearest')

            axs.plot(list(range(self.total_episodes)), mean, color=cmap(2))
            axs.fill_between(list(range(self.total_episodes)), mean +  sem, mean - sem, alpha = 0.3, color=cmap(2))
            axs.set_xlabel('# of episodes')
            axs.set_ylabel('Reward')
            axs.set_title('Training')
            axs.set_ylim([0, 1])
            axs.set_xlim([0, self.total_episodes])
    
        else:
            fig, axs = plt.subplots(1, 2, dpi=100, figsize=[12,6])

            mean = self.rewards_train.mean(axis=0)
            sem = self.rewards_train.std(axis=0) / (self.population_size**.5)

            # Smooth the graph with moving average over the mean of iterations
            if smoothness is not None:
                mean = uniform_filter1d(mean, size=smoothness, origin=int((smoothness-1)/2), mode='nearest')
                sem = uniform_filter1d(sem, size=smoothness, origin=int((smoothness-1)/2), mode='nearest')

            axs[0].plot(list(range(self.total_episodes)), mean, color=cmap(2))
            axs[0].fill_between(list(range(self.total_episodes)), mean +  sem, mean - sem, alpha = 0.3, color=cmap(2))
            axs[0].set_xlabel('# of episodes')
            axs[0].set_ylabel('Reward')
            axs[0].set_title('Training')
            axs[0].set_ylim([0, 1])
            axs[0].set_xlim([0, self.total_episodes])

            mean = self.rewards_test.mean(axis=(1,2))
            sem = self.rewards_test.std(axis=(1,2)) / (self.population_size**.5)

            if smoothness is not None:
                mean = uniform_filter1d(mean, size=smoothness, origin=int((smoothness-1)/2), mode='nearest')
                sem = uniform_filter1d(sem, size=smoothness, origin=int((smoothness-1)/2), mode='nearest')

            axs[1].plot(list(range(self.total_episodes)), mean, color=cmap(2))
            axs[1].fill_between(list(range(self.total_episodes)), mean +  sem, mean - sem, alpha = 0.3, color=cmap(2))
            axs[1].set_xlabel('# of episodes')
            axs[1].set_ylabel('Reward')
            axs[1].set_title('Testing (Zero-Shot)')
            axs[1].set_ylim([0, 1])
            axs[1].set_xlim([0, self.total_episodes])
        
    
    
        