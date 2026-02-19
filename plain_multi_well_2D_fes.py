#this returns the free energy surface for a 2D system with multiple wells
#given the qspace in meshgrid form.

import numpy as np
import matplotlib.pyplot as plt

def multi_well_fes(X,Y, amp = 6):
    num_wells = 9
    num_barrier = 1

    #here's the well params
    A_i = np.array([0.9, 0.3, 0.5, 1, 0.2, 0.4, 0.9, 0.9, 0.9]) * amp #this is in kcal/mol.
    x0_i = [1.12, 1, 3, 4.15, 4, 5.27, 5.5, 6, 1] # this is in nm.
    y0_i = [1.34, 2.25, 2.31, 3.62, 5, 4.14, 4.5, 1.52, 5]
    sigma_x_i = [0.5, 0.3, 0.4, 2, 0.9, 1, 0.3, 0.5, 0.5]
    sigma_y_i = [0.5, 0.3, 1, 0.8, 0.2, 0.3, 1, 0.6, 0.7]

    #here's the barrier params
    A_j = np.array([0.3]) * amp
    x0_j = [np.pi]
    y0_j = [np.pi]
    sigma_x_j = [3]
    sigma_y_j = [0.3]
    
    #initialize fes in XY space
    fes = np.zeros_like(X)

    for i in range(num_wells):
        fes += A_i[i] * np.exp(-((X-x0_i[i])**2/(2*sigma_x_i[i]**2) + (Y-y0_i[i])**2/(2*sigma_y_i[i]**2)))

    for j in range(num_barrier):
        fes += A_j[j] * np.exp(-((X-x0_j[j])**2/(2*sigma_x_j[j]**2) + (Y-y0_j[j])**2/(2*sigma_y_j[j]**2)))


    #normalize fes
    fes -= np.min(fes)    
    return fes

if __name__ == "__main__":
    #this is a test.
    X, Y = np.meshgrid(np.linspace(0,2*np.pi,100), np.linspace(0,2*np.pi,100))
    fes = multi_well_fes(X,Y)
    plt.contourf(X,Y,fes)
    #plt.show()
    plt.savefig("multi_well_fes.png")
    plt.close()