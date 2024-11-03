import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import gym
import time
import numpy as np
from collections import deque
from model.net import *
# 하이퍼파라미터 설정
learning_rate = 1e-3
gamma = 0.99
entropy_coef = 0.01
value_loss_coef = 0.5
max_steps = 200  # 각 에피소드 최대 스텝 수
batch_size = 256
update_interval = 10  # 학습자가 업데이트를 수행하는 간격
params_update_interval = 10  # 배우가 학습자의 파라미터를 수신하는 간격

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

if __name__ == "__main__":
    # global learner_network
    mp.set_start_method('spawn')
    num_actors = 4

    learner_network = LearnerNetwork(4, 2)
    learner_network.share_memory()
    optimizer = optim.Adam(learner_network.parameters(), lr=learning_rate)
    global_counter = mp.Value('i', 0)
    params_queue = mp.Queue()
    experience_queue = mp.Queue()
    metrics_queue = mp.Queue()

    # Learner init 파라미터
    params_queue.put(learner_network.state_dict())

    # 프로세스 생성
    processes = []
    for actor_id in range(num_actors):
        p = mp.Process(target=actor_process, args=(actor_id, global_counter, params_queue, experience_queue, metrics_queue))
        p.start()
        processes.append(p)

    learner = mp.Process(target=learner_process, args=(learner_network, optimizer, global_counter, experience_queue, params_queue, metrics_queue))
    learner.start()
    processes.append(learner)

    # 학습
    metrics_list = []
    scores = []
    start_time = time.time()
    try:
        while True:
            if not metrics_queue.empty():
                metrics = metrics_queue.get()
                if 'score' in metrics:
                    scores.append(metrics['score'])
                else:
                    metrics_list.append(metrics)
                if len(metrics_list) % 10 == 0 and len(metrics_list) > 0:
                    print(f"Global Steps: {metrics['global_steps']}, Critic Loss: {metrics.get('critic_loss', 0):.4f}, Entropy: {metrics.get('entropy', 0):.4f}, Imp_Ratio Avg: {metrics.get('imp_ratio_avg', 0):.4f}")
            if global_counter.value >= 200000: 
                break
    except KeyboardInterrupt:
        pass
    finally:
        for p in processes:
            p.terminate()
            p.join()

        # 메트릭 시각화
    import matplotlib.pyplot as plt

    # Critic Loss, Entropy 등 수집된 메트릭이 있는지 확인
    if metrics_list:
        critic_losses = [m['critic_loss'] for m in metrics_list]
        entropies = [m['entropy'] for m in metrics_list]
        imp_ratio_mins = [m['imp_ratio_min'] for m in metrics_list]
        imp_ratio_maxs = [m['imp_ratio_max'] for m in metrics_list]
        imp_ratio_avgs = [m['imp_ratio_avg'] for m in metrics_list]
        batch_times = [m['batch_time'] for m in metrics_list]
        forward_times = [m['forward_time'] for m in metrics_list]
        backward_times = [m['backward_time'] for m in metrics_list]

        fig, axs = plt.subplots(4, 2, figsize=(12, 10))

        axs[0, 0].plot(critic_losses)
        axs[0, 0].set_title("Critic Loss")

        axs[0, 1].plot(entropies)
        axs[0, 1].set_title("Entropy")

        axs[1, 0].plot(imp_ratio_mins)
        axs[1, 0].set_title("Importance Ratio Min")

        axs[1, 1].plot(imp_ratio_maxs)
        axs[1, 1].set_title("Importance Ratio Max")

        axs[2, 0].plot(imp_ratio_avgs)
        axs[2, 0].set_title("Importance Ratio Avg")

        axs[2, 1].plot(batch_times)
        axs[2, 1].set_title("Learner Batching Time")

        axs[3, 0].plot(forward_times)
        axs[3, 0].set_title("Learner Forward Time")

        axs[3, 1].plot(backward_times)
        axs[3, 1].set_title("Learner Backward Time")

        plt.tight_layout()
        plt.show()

    # Score를 시각화
    if scores:
        plt.figure()
        plt.plot(scores)
        plt.title("Score (Sum of Rewards per Episode)")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.show()
