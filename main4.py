import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import gym
import time
import numpy as np
import random
import matplotlib.pyplot as plt

# 신경망 정의
class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(ActorCriticNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),  # CartPole에 적합한 작은 신경망
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.policy_logits = nn.Linear(64, num_actions)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc(x)
        logits = self.policy_logits(x)
        value = self.value(x)
        return logits, value

# 하이퍼파라미터 설정
learning_rate = 1e-3
entropy_coef = 0.01
batch_size = 64
trajectory_length = 20  # 트래젝토리 길이
gamma = 0.99
value_loss_coef = 0.5
max_steps = 200
params_update_interval = 50  # 정책 동기화 간격 증가

def actor_process(actor_id, params_queue, experience_queue, metrics_queue, global_episode_counter):
    torch.manual_seed(123 + actor_id)
    np.random.seed(123 + actor_id)
    random.seed(123 + actor_id)

    env = gym.make("CartPole-v1")
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    episode_reward = 0
    steps = 0

    # 로컬 네트워크
    local_network = ActorCriticNetwork(4, 2)
    local_network.load_state_dict(params_queue.get())  # 초기 파라미터 수신

    while True:
        # 정책 동기화
        if steps % params_update_interval == 0:
            while not params_queue.empty():
                params = params_queue.get()
                local_network.load_state_dict(params)

        trajectory = []

        for _ in range(trajectory_length):
            logits, _ = local_network(state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            next_state = torch.tensor(next_state, dtype=torch.float32)

            trajectory.append({
                'state': state.numpy(),
                'action': action.item(),
                'reward': reward,
                'done': done,
                'log_prob': log_prob.item(),
                'next_state': next_state.numpy()
            })

            state = next_state
            episode_reward += reward
            steps += 1

            if done or steps >= max_steps:
                metrics_queue.put({'score': episode_reward})
                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float32)
                episode_reward = 0
                steps = 0
                with global_episode_counter.get_lock():
                    global_episode_counter.value += 1
                break

        # 트래젝토리 전송
        experience_queue.put(trajectory)

def compute_vtrace(behavior_log_probs, target_log_probs, rewards, values, bootstrap_value, dones, gamma, rho_bar=1.0, c_bar=1.0):
    with torch.no_grad():
        rho = torch.exp(target_log_probs - behavior_log_probs)
        rho = torch.clamp(rho, max=rho_bar)
        c = torch.clamp(rho, max=c_bar)

        values_t_plus_1 = torch.cat([values[1:], bootstrap_value.unsqueeze(0)], dim=0)
        deltas = rho * (rewards + gamma * values_t_plus_1 * (1 - dones) - values)

        vs = []
        vs_plus_1 = bootstrap_value
        for i in reversed(range(len(deltas))):
            vs_t = values[i] + deltas[i] + gamma * c[i] * (vs_plus_1 - values_t_plus_1[i]) * (1 - dones[i])
            vs.insert(0, vs_t)
            vs_plus_1 = vs_t

        vs = torch.stack(vs)
    return vs

def learner_process(global_network, optimizer, params_queue, experience_queue, metrics_queue):
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    params_queue.put(global_network.state_dict())  # 초기 파라미터 전송

    while True:
        trajectories = []
        total_steps = 0

        # 경험 수집
        while total_steps < batch_size:
            trajectory = experience_queue.get()
            trajectories.append(trajectory)
            total_steps += len(trajectory)

        # 배치 데이터 준비
        batch = []
        for trajectory in trajectories:
            batch.extend(trajectory)

        states = torch.tensor([item['state'] for item in batch], dtype=torch.float32)
        actions = torch.tensor([item['action'] for item in batch], dtype=torch.int64)
        rewards = torch.tensor([item['reward'] for item in batch], dtype=torch.float32)
        dones = torch.tensor([item['done'] for item in batch], dtype=torch.float32)
        behavior_log_probs = torch.tensor([item['log_prob'] for item in batch], dtype=torch.float32)

        logits, values = global_network(states)
        dist = torch.distributions.Categorical(logits=logits)
        target_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # V-trace 타깃 계산
        with torch.no_grad():
            next_states = torch.tensor([item['next_state'] for item in batch], dtype=torch.float32)
            _, bootstrap_values = global_network(next_states)
            bootstrap_values = bootstrap_values.squeeze(-1)

        values = values.squeeze(-1)

        vtrace_returns = compute_vtrace(
            behavior_log_probs=behavior_log_probs,
            target_log_probs=target_log_probs,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_values[-1],
            dones=dones,
            gamma=gamma
        )

        advantages = vtrace_returns - values
        policy_loss = -torch.mean(target_log_probs * advantages.detach())
        value_loss = torch.mean((vtrace_returns - values) ** 2)
        total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(global_network.parameters(), max_norm=0.5)
        optimizer.step()

        # 파라미터 업데이트
        params_queue.put(global_network.state_dict())

if __name__ == "__main__":
    mp.set_start_method('spawn')
    num_actors = 4

    global_network = ActorCriticNetwork(4, 2)
    global_network.share_memory()

    optimizer = optim.Adam(global_network.parameters(), lr=learning_rate)

    params_queue = mp.Queue()
    params_queue.put(global_network.state_dict())
    experience_queue = mp.Queue()
    metrics_queue = mp.Queue()
    global_episode_counter = mp.Value('i', 0)

    processes = []

    # 학습자 프로세스 시작
    learner = mp.Process(target=learner_process, args=(global_network, optimizer, params_queue, experience_queue, metrics_queue))
    learner.start()
    processes.append(learner)

    # 배우 프로세스 시작
    for actor_id in range(num_actors):
        p = mp.Process(target=actor_process, args=(actor_id, params_queue, experience_queue, metrics_queue, global_episode_counter))
        p.start()
        processes.append(p)

    # 학습 모니터링 및 로그 출력
    scores = []
    try:
        while True:
            if not metrics_queue.empty():
                metrics = metrics_queue.get()
                if 'score' in metrics:
                    scores.append(metrics['score'])
                    if len(scores) % 10 == 0:
                        avg_score = np.mean(scores[-10:])
                        print(f"Episode {len(scores)}, Average Score: {avg_score}")
                # 추가적인 메트릭 처리 가능
            if global_episode_counter.value >= 12000:
                break
    except KeyboardInterrupt:
        pass
    finally:
        for p in processes:
            p.terminate()
            p.join()

    # 결과 시각화
    if scores:
        plt.figure()
        plt.plot(scores)
        plt.title("Score (Sum of Rewards per Episode)")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.show()
