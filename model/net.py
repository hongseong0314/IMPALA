import torch
import torch.nn as nn

## IMPALA Actor 구현
class LearnerNetwork(nn.Module):
    """
    Learner 네트워크
    """
    def __init__(self, input_dim, action_dim):
        super(LearnerNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.policy_logits = nn.Linear(128, action_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.policy_logits(x), self.value(x)

class ActorNetwork(nn.Module):
    """
    Actor 네트워크 
    """
    def __init__(self, input_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.policy_logits = nn.Linear(128, action_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.policy_logits(x)#, self.value(x)