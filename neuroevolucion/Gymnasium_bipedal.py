import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import concurrent.futures
import gymnasium as gym
import numpy as np
import torch

class BipedalWalker(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(24, 16),
            nn.ReLU(),        
            nn.Linear(16, 8), 
            nn.ReLU(),
            nn.Linear(8, 4)
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            x = self.fc(x)
            x = torch.tanh(x)
            
        return x.numpy()
    
    def from_chromosome(self, chromosome):
        vector_to_parameters(torch.tensor(chromosome, dtype=torch.float32), self.parameters())
    
    def to_chromosome(self):
        return parameters_to_vector(self.parameters()).detach().numpy()



def evaluate_chromosome_parallel(chromosome, env_name="BipedalWalker-v3", N=6):
    local_env = gym.make(env_name)
    local_model = BipedalWalker()
    local_model.from_chromosome(chromosome)
    
    fitness_total = 0
    
    for _ in range(N):
        observation, info = local_env.reset()
        terminated = False
        truncated = False
        max_steps = 1000
        steps = 0

        while not (terminated or truncated) and steps < max_steps:
            action = local_model.forward(observation)
            observation, reward, terminated, truncated, info = local_env.step(action)
            
            fitness_total += reward
            steps += 1

        if steps >= max_steps:
            fitness_total -= 100 
            
    local_env.close()
    return fitness_total / N