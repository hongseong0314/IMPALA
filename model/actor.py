import torch
import torch.nn as nn
import gym
from config import *
## IMPALA Actor 구현

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
    
def actor_process(actor_id, global_counter, params_queue, experience_queue, metrics_queue):
    env = gym.make("CartPole-v1")

    # 환경 초기화 수정
    state, _ = env.reset()
    state = torch.FloatTensor(state)
    done = False
    episode_reward = 0

    local_network = ActorNetwork(4, 2)
    steps = 0

    # 초기에는 학습자의 파라미터를 로드
    if not params_queue.empty():
        state_dict = params_queue.get()
        local_network.load_state_dict(state_dict)

    while True:
        # 주기적으로 파라미터 업데이트
        if steps % params_update_interval == 0 and not params_queue.empty():
            state_dict = params_queue.get()
            local_network.load_state_dict(state_dict)

        logits = local_network(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # 환경 상호작용 수정
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        next_state = torch.FloatTensor(next_state)

        # 경험 저장
        experience = {
            'state': state.detach(),
            'action': action.detach(),
            'reward': reward,
            'next_state': next_state.detach(),
            'done': done,
            'log_prob': log_prob.detach()
        }
        experience_queue.put(experience)

        state = next_state
        episode_reward += reward
        steps += 1
        global_counter.value += 1

        if done or steps >= max_steps:
            # 에피소드 보상 합계를 메트릭 큐에 전달
            metrics_queue.put({'score': episode_reward, 'global_steps': global_counter.value})
            # 환경 재설정 수정
            state, _ = env.reset()
            state = torch.FloatTensor(state)
            done = False
            steps = 0
            episode_reward = 0