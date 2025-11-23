import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        #self.criterion = nn.MSELoss()
        self.criterion = nn.SmoothL1Loss()
        
    def train_step(self, state, action, reward, next_state, done):
        state = torch.from_numpy(np.array(state, dtype=np.float32))
        next_state = torch.from_numpy(np.array(next_state, dtype=np.float32))
        
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.bool)
        # (n, x)
        
        if state.ndim == 1:
            # (1, x)
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)
            
        batch_size = state.shape[0]
            
        # 1: predicted Q values with current state
        pred = self.model(state)
        
        with torch.no_grad():
            Q_next = self.model(next_state).max(dim=1).values
        
        target = pred.detach().clone()
        
        # for idx in range(len(done)):
        #     Q_new = reward[idx]
        #     if not done[idx]:
        #         #Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
        #         # avoiding ghosts
        #         with torch.no_grad():
        #             Q_next = torch.max(self.model(next_state[idx]))
        #         Q_new = reward[idx] + self.gamma * Q_next

        #     target[idx][torch.argmax(action).item()] = Q_new
        Q_new = reward + self.gamma * Q_next * (~done)
        target[range(batch_size), action] = Q_new
        
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()