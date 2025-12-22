import argparse
import json
from pathlib import Path
import torch
from torch import nn, optim

def parse_state_file(path):
    states = {}
    rewards = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                parts = line.split()
                if line.startswith('# step_cum_rate'):
                    continue
                continue
            parts = line.split()
            if len(parts) < 15:
                continue
            time = int(float(parts[0]))
            typ = parts[1]
            if typ != 'S':
                continue
            host = parts[2]
            dest = parts[4]
            contacts = float(parts[5])
            pred = float(parts[6])
            bufocc = float(parts[7])
            cap = float(parts[8])
            self_util = float(parts[9])
            reward = float(parts[13])
            pressure = self_util - bufocc
            features = [contacts, pred, bufocc, cap, self_util, pressure]
            states[(time, host, dest)] = features
            rewards[(time, host, dest)] = reward
    return states, rewards

def parse_action_file(path):
    actions = {}
    with open(path, 'r') as f:
        for line in f:
            if ' A ' not in line:
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            time = int(float(parts[0]))
            host = parts[2]
            dest = parts[4]
            delta = float(parts[5])
            actions[(time, host, dest)] = delta
    return actions

class ActorNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

class CriticNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)


def train(model, x, y, epochs=100, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', required=True)
    parser.add_argument('--bridge', required=True)
    parser.add_argument('--output_dir', default='models')
    args = parser.parse_args()

    state_map, reward_map = parse_state_file(args.state)
    action_map = parse_action_file(args.bridge)

    samples = []
    critic_samples = []
    for key, delta in action_map.items():
        if key not in state_map:
            continue
        feats = state_map[key]
        # append rate delta later
        samples.append((feats, delta))
        critic_samples.append((feats, reward_map.get(key, 0.0)))

    if not samples:
        raise RuntimeError('No matching state/action pairs found')

    x = torch.tensor([s for s, _ in samples], dtype=torch.float32)
    y = torch.tensor([[d] for _, d in samples], dtype=torch.float32)
    actor = ActorNet(x.shape[1])
    train(actor, x, y)

    xc = torch.tensor([s for s, _ in critic_samples], dtype=torch.float32)
    yc = torch.tensor([[v] for _, v in critic_samples], dtype=torch.float32)
    critic = CriticNet(xc.shape[1])
    train(critic, xc, yc)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), out_dir / 'buf10_bc_actor.pt')
    torch.save(critic.state_dict(), out_dir / 'buf10_bc_critic.pt')

if __name__ == '__main__':
    main()
