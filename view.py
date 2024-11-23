import gym
import torch
import cv2
import numpy as np

# 定义预处理函数
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 转换为灰度图
    resized_frame = cv2.resize(gray_frame, (84, 84))  # 缩放到84x84
    return resized_frame

# 定义堆叠帧的函数
def stack_frames(frames):
    stacked = np.stack(frames)  # 堆叠帧
    stacked = torch.from_numpy(stacked).float()  # 转换为Tensor
    return stacked

# 加载模型
class Qnetwork(torch.nn.Module):
    def __init__(self, actions_dim):
        super(Qnetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 512)
        self.fc2 = torch.nn.Linear(512, actions_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        B, C, H, W = x.shape
        x = x.view(B, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 实例化模型并加载参数
model_path = 'models_params/dqn_model_params.pth'  # 模型参数路径
actions_dim = 4  # 假设Breakout-v4有4个动作
q_net = Qnetwork(actions_dim)
q_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
q_net.eval()

# 创建环境
env = gym.make('Breakout-v4')
env.reset()

# 尝试获取渲染模式
try:
    processed_frame = preprocess_frame(env.render(mode='rgb_array'))
except TypeError:
    # 如果不支持 mode 参数，则尝试直接渲染
    try:
        processed_frame = preprocess_frame(env.render())
    except gym.error.Error as e:
        print(f"Render error: {e}")
        # 如果渲染模式无效，则尝试 'human' 模式
        env.render(mode='human')
        processed_frame = preprocess_frame(env.render(mode='rgb_array'))

# 初始化帧存储
frames = []
frames.append(processed_frame)
frames.append(processed_frame)
frames.append(processed_frame)
frames.append(processed_frame)

# 游戏主循环
done = False
while not done:
    # 将帧堆叠成状态
    picture = stack_frames(frames)
    picture = picture.unsqueeze(0)  # 添加批次维度

    # 使用模型选择动作
    action = q_net(picture).argmax().item()

    # 执行动作
    _, _, done, _, _ = env.step(action)

    # 预处理新的帧并添加到帧列表
    try:
        new_frame = preprocess_frame(env.render(mode='rgb_array'))
    except TypeError:
        try:
            new_frame = preprocess_frame(env.render())
        except gym.error.Error as e:
            print(f"Render error: {e}")
            env.render(mode='human')
            new_frame = preprocess_frame(env.render(mode='rgb_array'))
    frames.append(new_frame)
    if len(frames) > 4:
        frames.pop(0)

    # 渲染环境
    env.render(mode='human')

env.close()