import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        # self.linear1 = nn.Linear(input_size, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.linear3 = nn.Linear(hidden_size, hidden_size)
        # self.linear4 = nn.Linear(hidden_size, hidden_size)
        # self.linear5 = nn.Linear(hidden_size, hidden_size)
        # self.linear6 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        # x = F.relu(self.linear4(x))
        # x = F.relu(self.linear5(x))
        # x = self.linear6(x)
        return self.net(x)
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
class QTrainer:
    def __init__(self, model, target_model, lr, gamma, device):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        #self.criterion = nn.MSELoss()
        self.criterion = nn.SmoothL1Loss()
        
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)  # Float para multiplicação fácil

        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        #batch_size = state.shape[0]

        pred = self.model(state)  # Shape: [batch, n_actions]

        with torch.no_grad():
            next_pred = self.model(next_state)
            next_action = next_pred.argmax(dim=1, keepdim=True)  # [batch, 1]
            next_Q = self.target_model(next_state).gather(1, next_action).squeeze(-1)  # [batch]
            Q_target = reward + self.gamma * next_Q * (1 - done)

        action = action.unsqueeze(1) if action.dim() == 1 else action  # [batch, 1]
        current_Q = pred.gather(1, action).squeeze(-1) 

        loss = self.criterion(current_Q, Q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item() 
        
    def update_target(self, tau=0.005):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)