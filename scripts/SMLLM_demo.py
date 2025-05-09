# demo.py

from avp_env.envs.avp_env import AutonomousParkingEnv
from avp_env.agents.LLM_agent import multiImgMultimodalLLMAgent
from PIL import Image
import numpy as np
from avp_env.agents.image_process import split_multi_view_image


def main():
    # 初始化环境和 agent
    env = AutonomousParkingEnv()
    agent = multiImgMultimodalLLMAgent(api_key="your-openai-api-key")

    state = env.reset()
    instruction = env.getTargetInstruction().instruction
    done = False
    total_reward = 0

    while not done:
        position = env.getPosition()
        img_array = env.render()[0]  # shape: (H, W, 12)
        front_img, left_img, right_img, back_img = split_multi_view_image(img_array)

        views = [front_img, left_img, right_img, back_img]

        # 使用 agent 推理动作
        action = agent.get_action(views, instruction, position)

        # 执行动作
        state, reward, done, info = env.step(action)
        total_reward += reward

        print(f"[Position {position}] Action: {action}, Reward: {reward}")

    print(f"\n🎯 Final total reward: {total_reward}")

if __name__ == "__main__":
    main()
