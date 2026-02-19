#this is a python script fit some gaussians to the dialanine fes.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from PIL import Image
plt.rcParams.update({'font.size': 16})

# Define a single 2D Gaussian
def gaussian_2D(params, x, y):
    A, x0, y0, sigma_x, sigma_y = params
    return A * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2)))

# Define the sum of Gaussians
def sum_of_gaussians(params, x, y, n_gaussians, N=100):
    total = np.zeros((N,N))
    
    A = params[0::5]
    x0 = params[1::5]
    y0 = params[2::5]
    sigma_x = params[3::5]
    sigma_y = params[4::5]

    for i in range(n_gaussians):
        total += gaussian_2D([A[i], x0[i], y0[i], sigma_x[i], sigma_y[i]], x, y)
    return total

# Define the error function to be minimized
def error_function(params, x, y, fes, n_gaussians):
    fes_dim = fes.shape[0]
    diff = fes - sum_of_gaussians(params, x, y, n_gaussians, N = fes_dim)
    return np.sum(diff**2)

# Assume x_data, y_data, and fes_data are the 2D grid and FES values
# Let's fit using 5 Gaussians for illustration
n_gaussians = 15
test = True
img = Image.open("./fes_digitize.png")
img = np.array(img)

#we add the red, minus the blue, and add the average of greyscale.
img_greyscale = 0.6 * img[:,:,0] + 0.4 * img[:,:,1] + 0.11 * img[:,:,2]  # 0.8 * img[:,:,0] - 0.15 * img[:,:,1] - 0.2 * img[:,:,2]

img = img_greyscale
img = img/np.max(img)
img = img - np.min(img)

#get img square.
img = img[0:img.shape[0], 0:img.shape[0]]

#the image is on -pi to pi, we shift it to 0 to 2pi
img = np.roll(img, int(img.shape[0]/2), axis=0)
img = np.roll(img, int(img.shape[1]/2), axis=1)

plt.imshow(img, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=1.2)
plt.savefig("./test.png")

#note fes is normalized.
#now we fit it with gaussians
#we cast this into x,y coordinates [-pi, pi] x [-pi, pi]
#x, y = np.meshgrid(np.linspace(-np.pi, np.pi, img.shape[0]), np.linspace(-np.pi, np.pi, img.shape[0]))
x, y = np.meshgrid(np.linspace(0, 2*np.pi, img.shape[0]), np.linspace(0, 2*np.pi, img.shape[0]))

if test:
    gaussian_param = np.loadtxt("./fes_digitize_gauss_params_15_2.txt")

    #we flip the yaxis, so we change the y0. y0' = 2pi - y0
    gaussian_param[2::5] = 2*np.pi - gaussian_param[2::5]

    reconstructed = sum_of_gaussians(gaussian_param, x, y, n_gaussians, N=x.shape[0])
    plt.figure()
    plt.imshow(reconstructed, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], origin="lower")
    plt.colorbar()
    plt.title("normalized fes")
    plt.savefig("./fes_digitize_gauss15_2.png")
    plt.close()

#we distribute initial guess randomly over the grid.
initial_guess = np.zeros(n_gaussians*5)
initial_guess[0::5] = np.random.uniform(low=0.8, high=1, size=n_gaussians) #amplitude
initial_guess[1::5] = np.random.uniform(low=0, high=2*np.pi, size=n_gaussians) #x0
initial_guess[2::5] = np.random.uniform(low=0, high=2*np.pi, size=n_gaussians) #y0
initial_guess[3::5] = np.random.uniform(low=0.5, high=5, size=n_gaussians) #sigma_x
initial_guess[4::5] = np.random.uniform(low=0.5, high=5, size=n_gaussians) #sigma_y

result = minimize(error_function, initial_guess, args=(x, y, img, n_gaussians),tol=1e-1)
np.savetxt("./fes_digitize_gauss_params_15_2.txt", result.x)

#visualize the result
reconstructed = sum_of_gaussians(result.x, x, y, n_gaussians, N=x.shape[0])
plt.figure()
plt.imshow(reconstructed, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi])
plt.savefig("./fes_digitize_gauss15_2.png")
plt.close()

print("All done")