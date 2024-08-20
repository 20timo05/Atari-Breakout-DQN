import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from hyperparameters import RETURN_DISCOUNT, DEVICE


class DQN(nn.Module):
  def __init__(self, input_shape, out_features):
    super().__init__()
    self.out_features = out_features

    self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

    # Compute shape of the tensor after CNN layers
    with torch.no_grad():
        sample = torch.zeros(1, *input_shape)
        conv_out = self._forward_conv(sample)
        n_flatten = int(np.prod(conv_out.shape))

    self.fc1 = nn.Linear(n_flatten, 512)
    self.fc2 = nn.Linear(512, out_features)

    # Initialize weights
    self._initialize_weights()

  def _forward_conv(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    return F.relu(self.conv3(x))

  def forward(self, x):
    x = self._forward_conv(x)
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    return self.fc2(x)

  def _initialize_weights(self):
    for module in self.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

  def act(self, states, epsilon=1):
    states_tensor = torch.tensor(np.array(states, dtype=np.float32), device=DEVICE)

    qvalues = self(states_tensor)
    actions = qvalues.argmax(dim=-1).detach().tolist()

    for idx in range(len(actions)):
      if torch.rand(1) < epsilon:
        # explore instead of exploit
        actions[idx] = torch.randint(self.out_features, (1,)).item()
    
    return actions
  
  def compute_loss(self, batch, target_network):
    states, actions, rewards, next_states, dones = batch

    q_values = self(states)
    # gather q-values corresponding to action taken
    q_values = q_values.gather(1, actions.view(-1, 1)).view(-1)

    with torch.no_grad():
        # Bellmann Equation, Note: (1 - b_done) is making the discounted future returns 0
        # in case the previous state was a terminal state, meaning the new state S' is non existent => no discounted reward
        next_q_values = target_network(next_states)
        max_next_q_values = next_q_values.max(dim=1).values
        targets = rewards + (RETURN_DISCOUNT * max_next_q_values * (1 - dones))
        targets = targets.detach() # gradients should not flow through during backpropagation

    loss = F.smooth_l1_loss(q_values, targets)

    return loss



    
class ReplayMemory():
  def __init__(self, max_size, device):
    self.max_size = max_size
    self.device = device
    self.data = []
 
  def append(self, experience):
    self.data = (self.data + [experience])[-self.max_size:]
 
  def sample(self, batch_size):
    if len(self.data) < batch_size:
      return None
   
    batch = random.sample(self.data, batch_size)
    b_states, b_actions, b_rewards, b_next_states, b_dones = zip(*batch)
    b_states = torch.tensor(np.array(b_states, dtype=np.float32), device=self.device)
    b_actions = torch.tensor(np.array(b_actions, dtype=np.int64), device=self.device)
    b_rewards = torch.tensor(np.array(b_rewards, dtype=np.float32), device=self.device)
    b_next_states = torch.tensor(np.array(b_next_states, dtype=np.float32), device=self.device)
    b_dones = torch.tensor(np.array(b_dones, dtype=np.float32), device=self.device)
    return (b_states, b_actions, b_rewards, b_next_states, b_dones)
 
  def __len__(self):
    return len(self.data)