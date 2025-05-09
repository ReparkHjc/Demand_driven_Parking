from avp_env.envs.avp_env import MetricsVLLMEnv
from avp_env.agents.LLM_agent import combineMultimodalLLMAgent
from avp_env.agents.image_process import combine_views, split_multi_view_image

from PIL import Image
import numpy as np

env = MetricsVLLMEnv()
agent = combineMultimodalLLMAgent()

state = env.reset()
done = False
total_reward = 0
instruction = env.getTargetInstruction().instruction

while not done:
    position = env.getPosition()
    img_np = env.render()[0]  # HWC image as numpy
    front_img, left_img, right_img, back_img = split_multi_view_image(img_np)
    img = combine_views(front_img, left_img, right_img, back_img)

    action = agent.get_action(img, instruction, position)
    state, reward, done, info = env.step(action)
    total_reward += reward
    print(f"[pos={position}] action={action}, reward={reward}")

print("总奖励:", total_reward)
