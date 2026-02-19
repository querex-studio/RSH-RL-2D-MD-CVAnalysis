#here we took the 2D surface, create K, define the start/end and generate one-step optimized bias.

import numpy as np
import matplotlib.pyplot as plt
import scipy
from util import *
from config import *

#generate the 2D surface.
N = 20 #n_state
X,Y = np.meshgrid(np.linspace(0,2*np.pi,N),np.linspace(0,2*np.pi,N))

amp = 6
num_wells = 9
A_i = np.array([0.9, 0.3, 0.5, 1, 0.2, 0.4, 0.9, 0.9, 0.9]) * amp #this is in kcal/mol.
x0_i = [1.12, 1, 3, 4.15, 4, 5.27, 5.5, 6, 1] # this is in nm.
y0_i = [1.34, 2.25, 2.31, 3.62, 5, 4.14, 4.5, 1.52, 5]
sigma_x_i = [0.5, 0.3, 0.4, 2, 0.9, 1, 0.3, 0.5, 0.5]
sigma_y_i = [0.5, 0.3, 1, 0.8, 0.2, 0.3, 1, 0.6, 0.7]
num_barrier=1
A_j = np.array([0.3]) * amp
x0_j = [np.pi]
y0_j = [np.pi]
sigma_x_j = [3]
sigma_y_j = [0.3]

Z = np.zeros_like(X)
Z += amp * 4.184 #flat surface
for i in range(num_wells):
    Z -= A_i[i] * np.exp(-(X-x0_i[i])**2/(2*sigma_x_i[i]**2) - (Y-y0_i[i])**2/(2*sigma_y_i[i]**2))
for i in range(num_barrier):
    Z += A_j[i] * np.exp(-(X-x0_j[i])**2/(2*sigma_x_j[i]**2) - (Y-y0_j[i])**2/(2*sigma_y_j[i]**2))

#barrier around the corner.
k = 5  # Steepness of the sigmoid curve
max_barrier = "1e2"  # Scaling factor for the potential maximum
offset = 0.7 #the offset of the boundary energy barrier.


total_energy_barrier = np.zeros_like(X)
total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (X - (-offset))))) #left
total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (X - (2 * pi + offset))))) #right
total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (Y - (-offset)))))
total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (Y - (2 * pi + offset)))))
Z += total_energy_barrier
Z -= np.min(Z)


#plot the surface.
fig = plt.figure()
plt.tight_layout(pad=2.0)
plt.subplots_adjust(bottom=0.2)
plt.imshow(Z, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], origin="lower")
plt.xlabel("x (nm)")
plt.xlim([0, 2*np.pi])
plt.ylim([0, 2*np.pi])
plt.ylabel("y (nm)")
#plt.title("FES mode = multiwell, pbc=False")
cbar=plt.colorbar()
cbar.set_label("U (kcal/mol)")
plt.savefig('./figs/test_2D_fes.png')
plt.close()

#we define start/end coor
start_coor = np.array([5.0, 4.0])
end_coor = np.array([1.0, 1.5])

#get index
start_x = np.argmin(np.abs(X[0,:] - start_coor[0]))
start_y = np.argmin(np.abs(Y[:,0] - start_coor[1]))
end_x = np.argmin(np.abs(X[0,:] - end_coor[0]))
end_y = np.argmin(np.abs(Y[:,0] - end_coor[1]))
start_coor_digitized = [start_x, start_y]
end_coor_digitized = [end_x, end_y]

#ravel it to 1D.
start_index = np.ravel_multi_index(start_coor_digitized, (N,N), order='C')
end_index = np.ravel_multi_index(end_coor_digitized, (N,N), order='C')

from MSM import *
msm = MSM()
msm.num_dimensions = 2
msm.num_states = N # we ravel 2D into 1D.
msm.qspace = [X,Y]
msm.K = np.zeros((msm.num_states**msm.num_dimensions, msm.num_states**msm.num_dimensions), dtype=np.float64)

#build K from scratch.
Z = Z.ravel(order='C')
for i in range(msm.num_states**msm.num_dimensions):
    for j in range(i, msm.num_states**msm.num_dimensions):
        coor_i, coor_j, is_adjacent = msm._is_adjacent(i,j)
        if is_adjacent:
            u_ij = Z[j] - Z[i]
            msm.K[i,j] = np.exp((u_ij / (2 * msm.kBT)))
            msm.K[j,i] = np.exp((-u_ij / (2 * msm.kBT)))
    msm.K[i,i] = -np.sum(msm.K[:,i])

#check K
msm._compute_peq_fes_K()
msm._plot_fes(filename = './figs/free_energy_K_2D.png')

fes_K = np.reshape((msm.free_energy - msm.free_energy.min()), (20,20), order='C')
plt.figure()
plt.tight_layout(pad=2.0)
plt.subplots_adjust(bottom=0.2)
plt.imshow(fes_K, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], origin="lower")
plt.xlabel("x (nm)")
plt.xlim([0, 2*np.pi])
plt.ylim([0, 2*np.pi])
plt.ylabel("y (nm)")
#plt.title("FES mode = multiwell, pbc=False")
cbar=plt.colorbar()
cbar.set_label("U (kcal/mol)")
plt.savefig('./figs/test_2D_fes_K.png')
plt.close()

#now that we created MSM class, we do the random try and opt.

best_mfpt = np.inf
print("start opt, starting state is ", start_index, "end state is ", end_index)
for try_num in range(1000):
    rng = np.random.default_rng()
    a = np.ones(num_gaussian)
    bx = rng.uniform(0, 2*np.pi, num_gaussian)
    by = rng.uniform(0, 2*np.pi, num_gaussian)
    cx = rng.uniform(0.3, 1.5, num_gaussian)
    cy = rng.uniform(0.3, 1.5, num_gaussian)
    gaussian_params = np.concatenate((a, bx, by, cx, cy))
    total_bias = get_total_bias_2d(X, Y, gaussian_params)

    #we duplicate a temp msm object to do the random bias.
    temp_msm = MSM()
    temp_msm.num_dimensions = 2
    temp_msm.num_states = N # we ravel 2D into 1D.
    temp_msm.qspace = [X,Y]
    temp_msm.K = msm.K

    #here we bias the temp_msm
    temp_msm._bias_K(total_bias)
    temp_msm._compute_peq_fes_K()
    temp_msm._build_mfpt_matrix_K()
    mfpts_biased = temp_msm.mfpts

    mfpt_biased = mfpts_biased[start_index, end_index]
    print("random try:", try_num, "mfpt:", mfpt_biased)
    if try_num % 100 == 0:
        print("random try:", try_num, "mfpt:", mfpt_biased)
        temp_msm._kemeny_constant_check()
    if best_mfpt > mfpt_biased:
        best_mfpt = mfpt_biased
        best_params = gaussian_params

#we plot the best bias and biased fes.
total_bias = get_total_bias_2d(X, Y, best_params)
temp_msm = MSM()
temp_msm.num_dimensions = 2
temp_msm.num_states = N # we ravel 2D into 1D.
temp_msm.qspace = [X,Y]
temp_msm.K = msm.K

#here we bias the temp_msm
temp_msm._bias_K(total_bias)
temp_msm._compute_peq_fes_K()

#plot the biased fes.
fes_K = np.reshape((temp_msm.free_energy - temp_msm.free_energy.min()), (20,20), order='C')
fig = plt.figure()
plt.tight_layout(pad=2.0)
plt.subplots_adjust(bottom=0.2)
plt.xlabel("x (nm)")
plt.xlim([0, 2*np.pi])
plt.ylim([0, 2*np.pi])
plt.ylabel("y (nm)")
#plt.title("FES mode = multiwell, pbc=False")
cbar=plt.colorbar()
cbar.set_label("U (kcal/mol)")
plt.savefig('./figs/test_2D_fes_K_besttry.png')
plt.close()

    