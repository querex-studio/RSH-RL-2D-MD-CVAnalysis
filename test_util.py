#this is test python code for 2D util files
from util import *
import matplotlib.pyplot as plt
import numpy as np
import config

#we get a 2D MM.

fes = np.zeros(shape=(config.num_bins, config.num_bins))
gaussian_params = np.loadtxt('./params/gaussian_fes_param.txt')
for i in range(config.num_bins):
    for j in range(config.num_bins):
        fes[i,j] = gaussian_2D(i,j, *gaussian_params)

plt.imshow(fes, cmap = 'coolwarm', origin='lower')
plt.savefig("./test.png")
plt.close()

print("all done")