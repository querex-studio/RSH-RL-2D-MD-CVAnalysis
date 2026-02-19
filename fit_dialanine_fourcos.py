#this is a python script fit fourier cosine set to the dialanine fes.
# because we wanted to represenet the dialanine phi/psi fes as a sum of fourier cosine set.
# on -pi and pi, the periodicity of the fes should be 2pi. L = pi.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from PIL import Image

# Define a single 2D cosine fourier set
def fourier_sincos_full(params, x, y, m, n):
    """
    we take the sum of the polynomial of x and y:
    f(x,y) = (sum_{i=0}^m(a_i cos(i x) + b_i sin(i x)) )* (sum_{j=0}^n(c_j cos(j y) + d_j sin(j y)))
    if we unfold the brackets, we get:
    f(x,y) = sum_{i=0}^m sum_{j=0}^n (a_i c_j cos(i x) cos(j y) + a_i d_j cos(i x) sin(j y) + b_i c_j sin(i x) cos(j y) + b_i d_j sin(i x) sin(j y))
    and if we clean this up combinning the coeefeicients, we get:
    f(x,y) = sum_{i=0}^m sum_{j=0}^n (a_i c_j cos(i x) cos(j y) + (a_i d_j + b_i c_j) cos(i x) sin(j y) + b_i d_j sin(i x) sin(j y))
    """
    #unpack
    a = params[0:m*n]
    b = params[m*n:2*m*n]
    c = params[2*m*n:3*m*n]
    d = params[3*m*n:4*m*n]

    total = np.zeros(x.shape)
    for i in range(m):
        for j in range(n):
            total += a[i*n+j] * np.cos(i*x) * np.cos(j*y) + (a[i*n+j]*d[i*n+j] + b[i*n+j]*c[i*n+j]) * np.cos(i*x) * np.sin(j*y) + b[i*n+j]*d[i*n+j] * np.sin(i*x) * np.sin(j*y)
    return total

def wrapped_norm(params, x, y):
    """
    we use the 2D wrapped norm to calculate the error.
    1D probability density function:
    f(x, mu, sigma) = 1/(sigma sqrt(2pi)) sum_n (exp(-(x-mu + 2*pi*n)^2/(2 sigma^2)))

    2D equation:
    f(x, y, mu_x, mu_y, sigma_x, sigma_y) = 1/(sigma_x sigma_y 2pi) sum_n sum_m (exp(-(x-mu_x + 2*pi*n)^2/(2 sigma_x^2) - (y-mu_y + 2*pi*m)^2/(2 sigma_y^2)))
    
    #although this is periodic, we can fit the function in range of 0 to 2pi.
    #we can use the following form:
    f(x, y, mu_x, mu_y, sigma_x, sigma_y) = 1/(sigma_x sigma_y 2pi) sum_n sum_m (exp(-(x-mu_x + 2*pi*n)^2/(2 sigma_x^2) - (y-mu_y + 2*pi*m)^2/(2 sigma_y^2)))
    """
    num_gaussians = int(len(params)/5)

    #unpack
    a = params[0:num_gaussians]
    mu_x = params[num_gaussians:2*num_gaussians]
    mu_y = params[2*num_gaussians:3*num_gaussians]
    sigma_x = params[3*num_gaussians:4*num_gaussians]
    sigma_y = params[4*num_gaussians:5*num_gaussians]


    total = np.zeros(x.shape)
    for i in range(num_gaussians):
        total += a[i] * np.exp(-(x-mu_x[i])**2/(2 * sigma_x[i]**2) - (y-mu_y[i])**2/(2 * sigma_y[i]**2))

    #now we take into account the periodicity, k,j = -1 and k,j = 1, and add the residual to the total.
    # surrounding 8 images.
    for i in range(num_gaussians):
        total += a[i] * np.exp(-(x-mu_x[i])**2/(2 * sigma_x[i]**2) - (y-mu_y[i] + 2*np.pi)**2/(2 * sigma_y[i]**2))
        total += a[i] * np.exp(-(x-mu_x[i] + 2*np.pi)**2/(2 * sigma_x[i]**2) - (y-mu_y[i])**2/(2 * sigma_y[i]**2))
        total += a[i] * np.exp(-(x-mu_x[i] - 2*np.pi)**2/(2 * sigma_x[i]**2) - (y-mu_y[i])**2/(2 * sigma_y[i]**2))
        total += a[i] * np.exp(-(x-mu_x[i])**2/(2 * sigma_x[i]**2) - (y-mu_y[i] - 2*np.pi)**2/(2 * sigma_y[i]**2))
        total += a[i] * np.exp(-(x-mu_x[i] + 2*np.pi)**2/(2 * sigma_x[i]**2) - (y-mu_y[i] + 2*np.pi)**2/(2 * sigma_y[i]**2))
        total += a[i] * np.exp(-(x-mu_x[i] - 2*np.pi)**2/(2 * sigma_x[i]**2) - (y-mu_y[i] - 2*np.pi)**2/(2 * sigma_y[i]**2))
        total += a[i] * np.exp(-(x-mu_x[i] + 2*np.pi)**2/(2 * sigma_x[i]**2) - (y-mu_y[i] - 2*np.pi)**2/(2 * sigma_y[i]**2))
        total += a[i] * np.exp(-(x-mu_x[i] - 2*np.pi)**2/(2 * sigma_x[i]**2) - (y-mu_y[i] + 2*np.pi)**2/(2 * sigma_y[i]**2))
    
    return total

# Define the error function to be minimized
def error_function(params, x, y, fes, m=None, n=None):
    #diff = fes - fourier_sincos_full(params, x, y, m, n) #fourier_sincos_full(params, x, y, m, n)
    diff = fes - wrapped_norm(params, x, y)
    return np.sum(diff**2)

def residuals(params, x, y, fes, m=None, n=None):
    return (fes - fourier_sincos_full(params, x, y, m, n)).ravel()

# Assume x_data, y_data, and fes_data are the 2D grid and FES values
m = 5 #order of the fourier set
n = 5
test = False
img = Image.open("./fes_digitize.png")
img = np.array(img)
target_wrap_norm = True
target_minimize = False
target_leastsq = False
img_greyscale =  0.6 * img[:,:,0] + 0.4 * img[:,:,1] + 0.11 * img[:,:,2] #0.8 * img[:,:,0] - 0.15 * img[:,:,1] - 0.2 * img[:,:,2]
img = img_greyscale
img = img - np.min(img)
img = img/np.max(img)

#get img square.
img = img[0:img.shape[0], 0:img.shape[0]]

#the image is on -pi to pi, we shift it to 0 to 2pi
img = np.roll(img, int(img.shape[0]/2), axis=0)
img = np.roll(img, int(img.shape[1]/2), axis=1)

plt.figure()
plt.imshow(img, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=1.2)
plt.savefig("./test.png")
plt.close()


#note fes is normalized.
#we cast this into x,y coordinates [-pi, pi] x [-pi, pi]
#x, y = np.meshgrid(np.linspace(-np.pi, np.pi, img.shape[0]), np.linspace(-np.pi, np.pi, img.shape[0]))
if target_wrap_norm:
    num_gaussians = 20
    x, y = np.meshgrid(np.linspace(0, 2*np.pi, img.shape[0]), np.linspace(0, 2*np.pi, img.shape[0]))
    initial_guess = np.zeros(5*num_gaussians)
    initial_guess[0:num_gaussians] = np.random.uniform(low=0.1, high=1, size=num_gaussians) #a
    initial_guess[num_gaussians:2*num_gaussians] = np.random.uniform(low=0, high=2*np.pi, size=num_gaussians) #mu_x
    initial_guess[2*num_gaussians:3*num_gaussians] = np.random.uniform(low=0, high=2*np.pi, size=num_gaussians) #mu_y
    initial_guess[3*num_gaussians:4*num_gaussians] = np.random.uniform(low=0.5, high=5, size=num_gaussians) #sigma_x
    initial_guess[4*num_gaussians:5*num_gaussians] = np.random.uniform(low=0.5, high=5, size=num_gaussians) #sigma_y

    #visualize initial guess.
    reconstructed = wrapped_norm(initial_guess, x, y)
    plt.figure()
    plt.imshow(reconstructed, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi])
    plt.savefig("./basis_to_opt_wrapnorm.png")
    plt.close()


    result = minimize(error_function, initial_guess, args=(x, y, img), method='Nelder-Mead', tol=1e-6)
    np.savetxt("./fes_digitize_wrapnorm_params.txt", result.x)

    #visualize the result
    reconstructed = wrapped_norm(result.x, x, y)
    plt.figure()
    plt.imshow(reconstructed, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi])
    plt.savefig("./fes_digitize_wrapnorm.png")
    plt.close()
    print("All done")


if target_minimize:
    x, y = np.meshgrid(np.linspace(0, 2*np.pi, img.shape[0]), np.linspace(0, 2* np.pi, img.shape[0]))

    #we distribute initial guess randomly over the grid.
    """
    initial_guess = np.zeros(4*m*n+1)
    initial_guess[0] = 0.5
    initial_guess[1:m*n+1] = np.random.uniform(low=-1, high=1, size=m*n) #a
    initial_guess[m*n+1:2*m*n+1] = np.random.uniform(low=-1, high=1, size=m*n) #b
    initial_guess[2*m*n+1:3*m*n+1] = np.random.uniform(low=-1, high=1, size=m*n) #c
    initial_guess[3*m*n+1:4*m*n+1] = np.random.uniform(low=-1, high=1, size=m*n) #d
    """
    initial_guess = np.zeros(4*m*n)
    initial_guess[0:m*n] = np.random.uniform(low=-1, high=1, size=m*n) #a
    initial_guess[m*n:2*m*n] = np.random.uniform(low=-1, high=1, size=m*n) #b
    initial_guess[2*m*n:3*m*n] = np.random.uniform(low=-1, high=1, size=m*n) #c
    initial_guess[3*m*n:4*m*n] = np.random.uniform(low=-1, high=1, size=m*n) #d

    #show the initial guess
    reconstructed = fourier_sincos_full(initial_guess, x, y, m, n)
    plt.figure()
    plt.imshow(reconstructed, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi])
    plt.savefig("./basis_to_opt.png")
    plt.close()

    result = minimize(error_function, initial_guess, args=(x, y, img, m, n), method='Nelder-Mead', tol=1e-10)
    np.savetxt("./fes_digitize_fourier_params.txt", result.x)

    #visualize the result
    reconstructed = fourier_sincos_full(result.x, x, y, m, n)
    plt.figure()
    plt.imshow(reconstructed, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi])
    plt.savefig("./fes_digitize_fourier.png")
    plt.close()

    print("All done")
if target_leastsq:
    from scipy.linalg import leastsq
    params_out, conv_x, infodict, mesg, ier = leastsq(residuals, initial_guess, args=(x, y, img, m, n), full_output=True)
    np.savetxt("./fes_digitize_fourier_params_ls.txt", params_out)

    #visualize
    reconstructed = fourier_sincos_full(params_out, x, y, m, n)
    plt.figure()
    plt.imshow(reconstructed, cmap="coolwarm", extent=[-np.pi, np.pi,-np.pi, np.pi])
    plt.savefig("./fes_digitize_fourier_ls.png")
    