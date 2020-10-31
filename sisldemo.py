import time
import numpy as np
from PIL import Image
from pettingzoo.butterfly import pistonball_v0

env = pistonball_v0.parallel_env(local_ratio=.02, continuous=False, random_drop=True,
starting_angular_momentum=True, ball_mass = .75, ball_friction=.3,
ball_elasticity=1.5, max_frames=2000)


observation = env.reset()
# env.render(mode='human')
while True:
	actions = {}
	for agent in env.agents:
		actions[agent] = env.action_spaces[agent].sample()
	# print(actions)
	observations, rewards, dones, infos = env.step(actions)
	# joint_obs = Image.new('RGB', (4000, 120))
	# x_offset = 0
	for agent in env.agents:
		img = Image.fromarray(observations[agent], 'RGB')
		img.save('frames/train/' + agent + '.png')
		# img.show()
		# joint_obs.paste(img, (x_offset,0))
		# x_offset += img.size[0]
	# joint_obs.save("joint.png")
	break
	# env.render()
	if True in dones.values():
		print("here")
		break



