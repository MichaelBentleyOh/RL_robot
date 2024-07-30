import time
import gymnasium as gym
from xArm6_PickandPlace_Env import xArm6GraspEnv


# 환경을 초기화합니다.
env = xArm6GraspEnv()

# 환경을 리셋하여 초기 상태를 얻습니다.
observation = env.reset()

# 몇 번의 랜덤 스텝을 수행하며 환경을 렌더링합니다.
for _ in range(10000):
    # 0.1초 대기합니다.
    time.sleep(0.1)

    # 랜덤 액션을 샘플링합니다.
    action = env.action_space.sample()
    
    # 스텝을 수행합니다.
    observation, reward, done = env.step(action)
    
    # 환경을 렌더링합니다.
    env.render()
    
    # 0.1초 대기합니다.
    time.sleep(0.1)
    
    # 에피소드가 종료되면 환경을 리셋합니다.
    if done:
        observation = env.reset()

# 환경을 종료합니다.
env.close()
