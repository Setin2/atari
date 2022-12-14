import numpy as np
from matplotlib import pyplot as plt

rewards = np.load('logs/clipped_log.npy')

window = 100
average_data = []
std = []
for ind in range(len(rewards) - window + 1):
    average_data.append(np.mean(rewards[ind:ind + window]))
    std.append(np.std(rewards[ind:ind + window]))

ln = list(range(len(average_data)))
plt.plot(average_data)
#print(len(ln), len(average_data), len(std))
plt.fill_between(ln, np.subtract(average_data, std), np.add(average_data, std), alpha=0.5)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig('rewards.png')
plt.clf()
