import time
import torch
import torch.nn as nn

from config import *
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

def learner_process(learner_network, optimizer, global_counter, experience_queue, params_queue, metrics_queue):
    batch = []
    while True:
        start_time = time.time()
        # 경험 수집
        while len(batch) < batch_size:
            experience = experience_queue.get()
            batch.append(experience)
        batch_time = time.time() - start_time

        # 배치 구성
        states = torch.stack([exp['state'] for exp in batch])
        actions = torch.stack([exp['action'] for exp in batch]).squeeze()
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32)
        next_states = torch.stack([exp['next_state'] for exp in batch])
        dones = torch.tensor([exp['done'] for exp in batch], dtype=torch.float32)
        actor_log_probs = torch.stack([exp['log_prob'] for exp in batch]).squeeze()

        # 학습자 네트워크로 logits, values, log_probs, entropies 재계산
        logits, values = learner_network(states)
        values = values.squeeze()
        dist = torch.distributions.Categorical(logits=logits)
        learner_log_probs = dist.log_prob(actions)
        entropies = dist.entropy()

        # 중요도 샘플링 비율 재계산
        with torch.no_grad():
            imp_ratios = torch.exp(learner_log_probs - actor_log_probs).clamp(max=10)

        # 중요도 샘플링 비율 통계
        imp_ratio_min = imp_ratios.min().item()
        imp_ratio_max = imp_ratios.max().item()
        imp_ratio_avg = imp_ratios.mean().item()

        # 학습 단계
        start_forward = time.time()
        _, next_values = learner_network(next_states)
        forward_time = time.time() - start_forward
        next_values = next_values.squeeze()

        returns = rewards + gamma * next_values * (1 - dones)
        advantages = returns - values

        # 손실 계산
        policy_loss = - (imp_ratios * learner_log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        entropy = entropies.mean()
        loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

        # 역전파 및 업데이트
        start_backward = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_time = time.time() - start_backward

        # 최신 파라미터를 배우들에게 전달
        params_queue.put(learner_network.state_dict())

        # 메트릭 수집
        metrics = {
            'critic_loss': value_loss.item(),
            'entropy': entropy.mean().item(),
            'imp_ratio_min': imp_ratio_min,
            'imp_ratio_max': imp_ratio_max,
            'imp_ratio_avg': imp_ratio_avg,
            'batch_time': batch_time,
            'forward_time': forward_time,
            'backward_time': backward_time,
            'global_steps': global_counter.value
        }
        metrics_queue.put(metrics)

        batch = []
