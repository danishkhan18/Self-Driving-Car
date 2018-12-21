# AI for Self Driving Car


# Importing the libraries

# -*- coding: utf-8 -*-

import random #We will import this library to use the random function
import os     #We will import this library to save and load the last saved brain of the AI
import torch  #We will import this library to implement our neural network- Developed by facebook- open source - better than google tensorflow because it can handle dynamic graphs
import torch.nn as nn  #We will import this module of the torch library to specifically include the neural network module as nn to make the calling of this module easy later on
import torch.nn.functional as F #We will import this library to implement the functions for the use of neural network- this module will be used for the huber loss function because that improves convergence - we will import it as simply F lettr to make it easier for later calling 
import torch.optim as optim #We will import this library to use some optimisers- we will use c-grade descent - we will import it as optim to make it easier for later calling
from torch.autograd import Variable


# Creating the architecture of the Neural Network


class Network(nn.Module): #We make this class to make the architecture neural network of our AI
    
    def __init__(self, input_size, nb_action): #Initialisation of the class - mandatory - python syntax - This function defines the variable of the object that is the neural network - In this function we will define the architecture of the neural network which will contain 5 input neurons - Because we have 5 dimensions in the encoded vector of the input states - Then we will have the output layer which will contain all the possible actions we can play at each time
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
    
    def forward(self, state): #It will activate the neurons in the neural network - This will activate the signals - we will use the rectified activation function because we are dealing with a purely non linear problem and this rectified function breaks the linearity - we are making this forward functions to return the q values which are the outputs of the network - later we will be returning the final values by either taking the max of each values or the soft max method.
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values


# Implementing Experience Replay


class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


# Implementing Deep Q Learning


class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action




    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
