import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
#import rl_utils

class replaybuffer:
    ''' 经验回放池 '''    
    def __init__(self, capacity):        
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出
    def add(self, picture, action, reward, next_picture, done):  # 将数据加入buffer        
        self.buffer.append((picture, action, reward, next_picture, done))
    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size        
        transitions = random.sample(self.buffer, batch_size)        
        picture, action, reward, next_picture, done = zip(*transitions)        
        return picture, action, reward, next_picture, done
    def size(self):  # 目前buffer中数据的数量        
        return len(self.buffer)




class Qnetwork(nn.Module):
    def __init__(self, actions_dim):
        super(Qnetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)  # 输入为4帧，输出为32个特征图
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # 根据图像尺寸调整输入
        self.fc2 = nn.Linear(512, actions_dim)   # 输出层，num_actions 是动作的数量

        self.relu = nn.ReLU()
        
    def forward(self, x):
        #print("Input shape:", x.shape) # B C H W
        x = self.relu(self.conv1(x))
        # print("After conv1:", x.shape)
        x = self.relu(self.conv2(x))
        # print("After conv2:", x.shape)
        x = self.relu(self.conv3(x))
        #print("After conv3:", x.shape)
        B, C, H, W = x.shape
        x = x.view(B, -1)
        
        # print("After flatten:", x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        return x




class dqn:
    def __init__(self, action_dim, learning_rate, gamma,epsilon, target_update, device):        
        self.action_dim = action_dim        
        self.q_net = Qnetwork(self.action_dim).to(device)  # Q网络        # 目标网络        
        self.target_q_net = Qnetwork(self.action_dim).to(device)        
        # 使用Adam优化器        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),lr=learning_rate)        
        self.gamma = gamma  # 折扣因子        
        self.epsilon = epsilon  # epsilon-贪婪策略        
        self.target_update = target_update  # 目标网络更新频率        
        self.count = 0  # 计数器,记录更新次数        
        self.device = device
    
    def take_action(self, picture):  # epsilon-贪婪策略采取动作        
        if np.random.random() < self.epsilon:            
            action = np.random.randint(self.action_dim)        
        else:                        
            action = self.q_net(picture).argmax().item()        
        return action
    
    def update(self, transition_dict):
        #print(f"fuck: {torch.stack(transition_dict['picture']).shape}")  # 打印堆叠后的形状
        picture=torch.stack(transition_dict['picture']).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)        
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)        
        next_picture = torch.stack(transition_dict['next_picture']).to(self.device)      
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)
        q_values = self.q_net(picture).gather(1, actions)  # Q值        
        
        
        # 下个状态的最大Q值        
        max_next_q_values = self.target_q_net(next_picture).max(1)[0].view(-1, 1)        
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标        
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数        
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0        
        dqn_loss.backward()  # 反向传播更新参数        
        self.optimizer.step()
        if self.count % self.target_update == 0:            
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络        
        self.count += 1




lr = 2e-3
num_episodes = 100
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 256
device = torch.device("cpu") 
env = gym.make('Breakout-v4')
random.seed(0)
np.random.seed(0)
env.reset(seed=0)
torch.manual_seed(0)
replay_buffer = replaybuffer(buffer_size)
action_dim = env.action_space.n
agent = dqn(action_dim, lr, gamma, epsilon,target_update, device)
return_list = []
# 初始化帧存储
#frames = []
#print(gym.__version__)
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def preprocess_frame(frame):
    # 将图像转换为灰度并缩放
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized_frame = cv2.resize(gray_frame, (84, 84))  # 缩放为 84x84
    return resized_frame

def stack_frames(frames):
    # 假设frames是一个包含4个处理后帧的列表
    stacked = np.stack(frames)  # 堆叠帧
    #print(f"Stacked shape before permute: {stacked.shape}")  # 打印堆叠后的形状

    # 将堆叠的数组转换为Tensor
    stacked = torch.from_numpy(stacked).float()  # 转换为Tensor
    

    #print(f"Stacked shape after permute: {stacked.shape}")  # 打印变换后的形状
    
    return stacked

for i in range(20):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            frames = []
            episode_return = 0
            env.reset(seed=0)
            state = env.reset(seed=0)
            state = state[0] 
            done = False
            # 预处理帧
            processed_frame = preprocess_frame(state)

            # 将处理后的帧加入到帧列表中
            frames.append(processed_frame)
            frames.append(processed_frame)
            frames.append(processed_frame)
            frames.append(processed_frame)
            picture=stack_frames(frames)
            #print(f"Picture type: {type(picture)}")
            #print('picture shape: ', picture.shape)
            # picture = picture.unsqueeze(0)
            while not done:
                picture0 = picture.unsqueeze(0)
                action = agent.take_action(picture0)
                next_state, reward, done, truncated, _ = env.step(action)
                #print(next_state.shape)
                processed_frame = preprocess_frame(next_state)
                frames.append(processed_frame)
                if len(frames) > 4:
                    frames.pop(0)
                next_picture=stack_frames(frames)

                done=done or truncated
                replay_buffer.add(picture, action, reward, next_picture, done)
                picture = next_picture
                episode_return += reward

                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    
                    transition_dict = {                        
                        'picture': b_s,                        
                        'actions': b_a,                        
                        'next_picture': b_ns,                        
                        'rewards': b_r,                        
                        'dones': b_d  
                    }                    
                    agent.update(transition_dict)            
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({                    
                    'episode':                    
                    '%d' % (num_episodes / 10 * i + i_episode + 1),                    
                    'return':                    
                    '%.3f' % np.mean(return_list[-10:])                
                })            
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format('Breakout-v4'))
plt.show()
#mv_return = rl_utils.moving_average(return_list, 9)
mv_return = moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format('Breakout-v4'))


import os
import torch

# 定义两个文件夹名称
models_dir = 'models_full'  # 保存整个模型的文件夹
models_params_dir = 'models_params'  # 仅保存模型参数的文件夹

# 检查两个文件夹是否存在，如果不存在则创建
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(models_params_dir):
    os.makedirs(models_params_dir)

# 保存整个模型到 models_full 文件夹
full_model_path = os.path.join(models_dir, 'dqn_full_model.pth')
torch.save(agent.q_net, full_model_path)
print(f"Full model saved to {full_model_path}")

# 保存模型参数到 models_params 文件夹
params_model_path = os.path.join(models_params_dir, 'dqn_model_params.pth')
torch.save(agent.q_net.state_dict(), params_model_path)
print(f"Model parameters saved to {params_model_path}")
