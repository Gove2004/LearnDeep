import gymnasium as gym
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

def run_learn(*args, **kwargs):
    """DQN and RL"""

    EPOCH = 1000
    BATCH_SIZE = 256
    LR = 0.001
    HIDDEN_LAYER = 128
    LOSS_SCORE = -10
    GAMMA = 0.9
    EPSILON = 1
    EPSILON_DECAY = 0.995
    TARGET_REPLACE_ITER = 10
    MEMORY_CAPACITY = 100_000
    env = gym.make('CartPole-v1')
    N_STATES = env.observation_space.shape[0]
    N_ACTIONS = env.action_space.n
    DEVICES = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[learn9] Using device: {DEVICES}")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(N_STATES, HIDDEN_LAYER)
            self.fc1.weight.data.normal_(0, 0.1)
            self.fc2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
            self.fc2.weight.data.normal_(0, 0.1)
            self.out = nn.Linear(HIDDEN_LAYER, N_ACTIONS)
            self.out.weight.data.normal_(0, 0.1)

        def forward(self, x):
            return self.out(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

    class DQN(object):
        def __init__(self):
            self.eval_net, self.target_net = Net(), Net()
            self.learn_step_counter = 0
            self.memory_counter = 0
            self.memory = torch.zeros((MEMORY_CAPACITY, N_STATES * 2 + 3))
            self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
            self.loss_func = nn.SmoothL1Loss()
            self.epsilon = EPSILON

            self.eval_net.to(DEVICES)
            self.target_net.to(DEVICES)
            self.memory = self.memory.to(DEVICES)

        def choose_action(self, x):
            x = torch.unsqueeze(torch.tensor(x, dtype=torch.float32, device=DEVICES), 0)
            if torch.rand(1).item() < self.epsilon:
                return env.action_space.sample()
            else:
                with torch.no_grad():
                    return torch.max(self.eval_net(x), 1)[1].item()

        def store_transition(self, s, a, r, s_, done):
            transition = torch.cat((
                torch.tensor(s, dtype=torch.float32, device=DEVICES),
                torch.tensor([a, r], dtype=torch.float32, device=DEVICES),
                torch.tensor([1.0 if done else 0.0], dtype=torch.float32, device=DEVICES),
                torch.tensor(s_, dtype=torch.float32, device=DEVICES),
            )).to(DEVICES)
            idx = self.memory_counter % MEMORY_CAPACITY
            self.memory[idx, :] = transition
            self.memory_counter += 1

        def learn(self):
            if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
            self.learn_step_counter += 1

            idx = torch.randint(0, min(MEMORY_CAPACITY, self.memory_counter), (BATCH_SIZE,))
            b_m = self.memory[idx, :]
            b_s = b_m[:, :N_STATES]
            b_a = b_m[:, N_STATES:N_STATES + 1].long()
            b_r = b_m[:, N_STATES + 1:N_STATES + 2]
            b_done = b_m[:, N_STATES + 2:N_STATES + 3]
            b_s_ = b_m[:, N_STATES + 3:]

            q_eval = self.eval_net(b_s).gather(1, b_a)
            q_eval_next = self.eval_net(b_s_)
            best_actions = q_eval_next.max(1)[1].view(BATCH_SIZE, 1)
            q_next = self.target_net(b_s_).gather(1, best_actions)
            q_target = b_r + GAMMA * q_next * (1 - b_done)

            loss = self.loss_func(q_eval, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    dqn = DQN()
    reward_history = []

    for epoch in range(EPOCH):
        s, _ = env.reset()
        done = False
        ep_reward = 0
        dqn.epsilon = max(0.01, dqn.epsilon * EPSILON_DECAY)

        while not done:
            a = dqn.choose_action(s)
            s_, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            r = LOSS_SCORE if done else r

            dqn.store_transition(s, a, r, s_, done)
            ep_reward += r

            if dqn.memory_counter >= BATCH_SIZE:
                dqn.learn()

            s = s_

        reward_history.append(ep_reward)

        if epoch % 10 == 0:
            avg = sum(reward_history[-10:]) / min(10, len(reward_history))
            print(f"[learn9] Epoch {epoch}: Avg Reward = {avg:.1f}")

            view_env_render = gym.make('CartPole-v1', render_mode='human')
            view_s, _ = view_env_render.reset()
            view_done = False
            while not view_done:
                view_a = dqn.choose_action(view_s)
                view_s, view_r, view_terminated, view_truncated, _ = view_env_render.step(view_a)
                view_done = view_terminated or view_truncated
            view_env_render.close()

    plt.plot(reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Training Progress')
    plt.show()

    env.close()
