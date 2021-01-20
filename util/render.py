import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import imageio

from agents.qlearning import QLearning


class Renderer():
    """
    Description:
        A rendering class that renders the Q values for the lever coordination game.
    """
    def __init__(self, total_episodes, players):
        self.total_episodes = total_episodes
        
        if isinstance(players, QLearning):
            self.q_values_train = np.zeros((1, self.total_episodes, *players.Q_values.shape))
        elif isinstance(players, list):
            self.q_values_train = np.zeros((len(players), self.total_episodes, *players[0].Q_values.shape))
    

    def store_q_values(self, episode, players):
        """
            Store the Q values of the given player(s)
        """
        if isinstance(players, QLearning):
            self.q_values_train[0][episode] = players.Q_values.copy()
        elif isinstance(players, list):
            for idx, player in enumerate(players):
                assert isinstance(player, QLearning), "Players need to be of type QLearning"
                self.q_values_train[idx][episode] = player.Q_values.copy()
                
    def render_q_values(self, filename, interval=0):
        """
            Render the evolution of the Q values and store the resulting animation in a .gif
        """
        backend = matplotlib.get_backend()
        matplotlib.rcParams['text.antialiased'] = False
        plt.switch_backend('Agg')    # makes sure we can convert to rgb array
        
        frames = []
        frame_shape = (*self.q_values_train[0][0].shape, 4)
        
        num_players = self.q_values_train.shape[0]
        
        if num_players == 1:
            for episode in range(self.total_episodes):
                if interval == 0 or episode == 0 or episode == self.total_episodes - 1 or (episode+1) % interval == 0:
                    fig, axs = plt.subplots(3, 1, dpi=80, figsize=[8,2])
                    axs[0].axis('off')
                    axs[2].axis('off')

                    image = axs[1].imshow(self.q_values_train[0][episode], vmin=0, vmax=1, aspect='auto')

                    fig.colorbar(image, ax=axs)
                    axs[1].set_yticks([1])
                    axs[1].set_xticks(range(self.q_values_train[0][episode].size))
                    axs[1].set_title('Q values during training', pad=20)
                    axs[1].set_xlabel(f'Episode {episode+1}', rotation='horizontal', labelpad=20)

                    # draw to canvas and convert to rgb array
                    fig.canvas.draw()
                    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    frames.append(data)
                    plt.close(fig)
                    
            
        elif num_players > 1:
            for episode in range(self.total_episodes):
                if interval == 0 or episode == 0 or episode == self.total_episodes - 1 or (episode+1) % interval == 0:
                    fig, axs = plt.subplots(num_players, 1, dpi=80, figsize=[8,2 * num_players/3])

                    for i in range(num_players):
                        image = axs[i].imshow(self.q_values_train[i][episode], vmin=0, vmax=1, aspect='auto')

                        axs[i].set_yticks([1])
                        axs[i].set_xticks(range(self.q_values_train[i][episode].size))
                        axs[i].set_ylabel(f'Player {i+1}', rotation='horizontal', labelpad=35)

                    axs[0].set_title('Q values during training', pad=5)
                    axs[-1].set_xlabel(f'Episode {episode+1}', rotation='horizontal', labelpad=None)
                    
                    # to make sure the x label doesn't fall of the bottom
                    plt.subplots_adjust(bottom=0.15)
                    
                    fig.colorbar(image, ax=axs)

                    # draw to canvas and convert to rgb array
                    fig.canvas.draw()
                    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    frames.append(data)
                    plt.close(fig)
            
                
                
                
                
        durations = [1/100 for i in range(len(frames))]
        durations[-1] = 5
        imageio.mimwrite(f'gif/{filename}.gif', frames, duration=durations)
        plt.switch_backend(backend)    # so we can draw to screen again