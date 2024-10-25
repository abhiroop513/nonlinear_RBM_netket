# Created by Abhiroop Lahiri at 15:18 05/09/2022 using PyCharm
import numpy as np
import matplotlib.pyplot as plt
import json

# import the data from log file
data_RBM = json.load(open("RBM.log"))

iters_RBM = data_RBM["Energy"]["iters"]
energy_RBM = data_RBM["Energy"]["Mean"]["real"]

fig, ax1 = plt.subplots()
ax1.plot(iters_RBM, energy_RBM1, color='r', label='aplha=1.2')
ax1.set_ylabel('Energy')
ax1.set_xlabel('Epochs')
ax1.legend()

plt.savefig('energy_vs_epoch.png', dpi=300)
plt.show()
