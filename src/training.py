"""
Обучение стратегии GRA-обнуления для максимального lifespan.
Используется policy gradient с reward за выживание и штрафом за пену.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class LifespanPolicy(nn.Module):
    """Нейросеть, предсказывающая параметры обнуления (LR) для каждого уровня."""
    def __init__(self, state_dim, num_levels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_levels),
            nn.Softplus()  # LR > 0
        )

    def forward(self, x):
        return self.net(x) + 0.001  # минимальный LR

def train_geroprotector(env, policy, optimizer, episodes=1000):
    gamma = 0.99
    for ep in range(episodes):
        patient = env.reset()
        state = patient.get_state()
        log_probs = []
        rewards = []
        done = False
        t = 0
        while not done:
            state_t = torch.FloatTensor(state)
            lrs = policy(state_t)  # вектор learning rates для уровней
            # Применяем GRA (упрощённо: коррекция = -lrs * gradient(state))
            action = -lrs.detach().numpy() * state  # модель корректирующего воздействия
            next_state, reward, done, info = env.step(action)
            # reward = survival bonus - foam_penalty
            foam = info['foam']
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            # Сохраняем log_prob действия (упрощённый REINFORCE: будем считать loss)
            # Для простоты используем MSE между действием и оптимальным (здесь не реализовано)
            rewards.append(reward_tensor)
            state = next_state
            t += 1
        # Calculate discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.stack(returns)
        # Loss = - Σ log_prob * return (здесь log_prob ~ -MSE(action, target))
        loss = -torch.mean(returns)  # заглушка
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
