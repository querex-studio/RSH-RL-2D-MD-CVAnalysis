#utility file for langevin_sim_mfpt_opt.py
#bt TW 9th Oct 2023.
# Path: langevin_approach/util.py

import numpy as np
from scipy.linalg import eig
from scipy.linalg import expm
from scipy.linalg import inv
from scipy.optimize import minimize
from math import pi
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

import openmm
import config

plt.rcParams.update({'font.size': 18})


"""

def gaussian_2d(x, y, ax, bx, by, cx, cy): #deprecated
    return ax * np.exp(-((x-bx)**2/(2*cx**2) + (y-by)**2/(2*cy**2)))
"""
def gaussian_2D(params, x, y):
    A, x0, y0, sigma_x, sigma_y = params
    return A * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2)))

def random_initial_bias_2d(initial_position, num_gaussians = 20):
    # initial position is a list e.g. [3,3]
    # note this is in 
    #returns a set of random ax,ay, bx, by, cx, cy for the 2d Gaussian function

    #we convert the initial position from openmm quantity object to array with just the value.
    initial_position = initial_position.value_in_unit_system(openmm.unit.md_unit_system)[0] #this is in nm.
    rng = np.random.default_rng()
    a = np.ones(num_gaussians) * 0.01#* 4 #
    #ay = np.ones(num_gaussians) * 0.1 #there's only one amplitude!
    bx = rng.uniform(initial_position[0]-1, initial_position[0]+1, num_gaussians)
    by = rng.uniform(initial_position[1]-1, initial_position[1]+1, num_gaussians)
    cx = rng.uniform(1.0, 5.0, num_gaussians)
    cy = rng.uniform(1.0, 5.0, num_gaussians)
    #"gaussian_param should be in A, x0, y0, sigma_x, sigma_y format."
    return np.concatenate((a, bx, by, cx, cy))

def get_total_bias_2d(x,y, gaussian_params):
    """
    here we get the total bias at x,y.
    note: we used the transposed K matrix, we need to apply transposed total gaussian bias.
    """
    N = x.shape[0] #N is the number of grid points.
    total_bias = np.zeros((N,N))
    num_gaussians = len(gaussian_params)//5
    a = gaussian_params[:num_gaussians]
    bx = gaussian_params[num_gaussians:2*num_gaussians]
    by = gaussian_params[2*num_gaussians:3*num_gaussians]
    cx = gaussian_params[3*num_gaussians:4*num_gaussians]
    cy = gaussian_params[4*num_gaussians:5*num_gaussians]
    for i in range(num_gaussians):
        total_bias = total_bias + gaussian_2D([a[i], bx[i], by[i], cx[i], cy[i]], x, y,)

    return total_bias

def compute_free_energy(K, kT):
    """
    In 2D senario, we just need to reshape the peq and F.

    K is the transition matrix
    kT is the thermal energy
    peq is the stationary distribution #note this was defined as pi in Simian's code.
    F is the free energy
    eigenvectors are the eigenvectors of K

    first we calculate the eigenvalues and eigenvectors of K
    then we use the eigenvalues to calculate the equilibrium distribution: peq.
    then we use the equilibrium distribution to calculate the free energy: F = -kT * ln(peq)
    """
    N = int(np.sqrt(K.shape[0]))
    evalues, evectors = eig(K)

    #sort the eigenvalues and eigenvectors
    index = np.argsort(evalues) #sort the eigenvalues, the largest eigenvalue is at the end of the list
    evalues_sorted = evalues[index] #sort the eigenvalues based on index

    #calculate the equilibrium distribution
    peq = evectors[:, index[-1]].T #normalize the eigenvector
    peq = peq / np.sum(peq)
    peq = peq.real
    #take the real part of the eigenvector i.e. the probability distribution at equilibrium.
    #calculate the free energy
    F = -kT * np.log(peq) #add a small number to avoid log(0))
    #F = F.reshape(N, N)
    return [peq, F, evectors, evalues, evalues_sorted, index]

def compute_free_energy_power_method(K, kT=0.5981):
    """
    this use the power method to calculate the equilibrium distribution.
    num_iter is the number of iterations.
    """
    num_iter = 1000
    N = K.shape[0]
    peq = np.ones(N) / N #initialise the peq
    for i in range(num_iter):
        peq = np.dot(peq, K)
        peq = peq / np.sum(peq)
    F = -kT * np.log(peq)
    return [peq, F]

def kemeny_constant_check(mfpt, peq):
    N2 = mfpt.shape[0]
    kemeny = np.zeros((N2, 1))
    for i in range(N2):
        for j in range(N2):
            kemeny[i] = kemeny[i] + mfpt[i, j] * peq[j]
    #print("Performing Kemeny constant check...")
    print("the min/max of the Kemeny constant is:", np.min(kemeny), np.max(kemeny))
    """
    if np.max(kemeny) - np.min(kemeny) > 1e-6:
        print("Kemeny constant check failed!")
        raise ValueError("Kemeny constant check failed!")"""
    return kemeny

def mfpt_calc(peq, K):
    """
    peq is the probability distribution at equilibrium.
    K is the transition matrix.
    N is the number of states.
    """
    N = K.shape[0] #K is a square matrix.
    onevec = np.ones((N, 1)) #, dtype=np.float64
    Qinv = np.linalg.inv(peq.T * onevec - K.T)

    mfpt = np.zeros((N, N)) #, dtype=np.float64
    for j in range(N):
        for i in range(N):
            #to avoid devided by zero error:
            if peq[j] == 0:
                mfpt[i, j] = 0
            else:
                mfpt[i, j] = 1 / peq[j] * (Qinv[j, j] - Qinv[i, j])
    
    #result = kemeny_constant_check(N, mfpt, peq)
    return mfpt

def Markov_mfpt_calc(peq, M):
    N = M.shape[0]
    onevec = np.ones((N, 1))
    Idn = np.diag(onevec[:, 0])
    A = (peq.reshape(-1, 1)) @ onevec.T #was peq.T @ onevec.T
    A = A.T
    Qinv = inv(Idn + A - M)
    mfpt = np.zeros((N, N))
    for j in range(N):
        for i in range(N):
            term1 = Qinv[j, j] - Qinv[i, j] + Idn[i, j]
            if peq[j] * term1 == 0:
                mfpt[i, j] = 1000000000000
            else:
                mfpt[i, j] = 1/peq[j] * term1
    #result = kemeny_constant_check(N, mfpt, peq)
    return mfpt

def try_and_optim_M(M, working_indices, N=20, num_gaussian=10, start_index=0, end_index=0, plot = False):
    """
    here we try different gaussian params 1000 times
    and use the best one (lowest mfpt) to local optimise the gaussian_params
    
    returns the best gaussian params

    input:
    M: the working transition matrix, square matrix.
    working_indices: the indices of the working states.
    num_gaussian: number of gaussian functions to use.
    start_state: the starting state. note this has to be converted into the index space.
    end_state: the ending state. note this has to be converted into the index space.
    index_offset: the offset of the index space. e.g. if the truncated M (with shape [20, 20]) matrix starts from 13 to 33, then the index_offset is 13.
    """
    #here we find the index of working_indices.
    # e.g. the starting index in the working_indices is working_indices[start_state_working_index]
    # and the end is working_indices[end_state_working_index]
    
    start_state_working_index = np.argmin(np.abs(working_indices - start_index))
    end_state_working_index = np.argmin(np.abs(working_indices - end_index))
    
    start_state_working_index_xy = np.unravel_index(working_indices[start_state_working_index], (N, N), order='C')
    end_state_working_index_xy = np.unravel_index(working_indices[end_state_working_index], (N, N), order='C')
    print("Try and Optim from state:", start_state_working_index_xy, "to state:", end_state_working_index_xy)

    #now our M/working_indices could be incontinues. #N = M.shape[0]
    x,y = np.meshgrid(np.linspace(0, 2*np.pi, N), np.linspace(0, 2*np.pi, N)) #hard coded here. we need to change this.
    best_mfpt = 1e20 #initialise the best mfpt np.inf

    #here we find the x,y maximum and minimun in xy coordinate space, with those working index
    #we use this to generate the random gaussian params.
    working_indices_xy = np.unravel_index(working_indices, (N, N), order='C')

    for try_num in range(1000):
        rng = np.random.default_rng()
        a = np.ones(num_gaussian)
        bx = rng.uniform(0, 2*np.pi, num_gaussian)
        by = rng.uniform(0, 2*np.pi, num_gaussian)
        cx = rng.uniform(0.3, 1.5, num_gaussian)
        cy = rng.uniform(0.3, 1.5, num_gaussian)
        gaussian_params = np.concatenate((a, bx, by, cx, cy))

        total_bias = get_total_bias_2d(x,y, gaussian_params)
        M_biased = np.zeros_like(M)

        #we truncate the total_bias to the working index.
        working_bias = total_bias[working_indices_xy] #say M is in shape[51,51], working bias will be in [51] shape.

        #now we have a discontinues M matrix. we need to apply the bias to the working index.
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                u_ij = working_bias[j] - working_bias[i]
                M_biased[i,j] = M[i,j] * np.exp(-u_ij / (2*0.5981))
            M_biased[i,i] = M[i,i]
        #epsilon_offset = 1e-15
        #M_biased = M_biased / (np.sum(M_biased, axis=0)[:, None] + 1e-15)
        for i in range(M_biased.shape[0]):
            if np.sum(M_biased[i, :]) > 0:
                M_biased[i, :] = M_biased[i, :] / np.sum(M_biased[i, :])
            else:
                M_biased[i, :] = 0
        
        #note our M_biased is in working index. M.shape = (num_working_states, num_working_states)
        peq,F,_,_,_,_ = compute_free_energy(M_biased.T.astype(np.float64), kT=0.5981)
        #print(peq)
        #print(sum(peq))

        mfpts_biased = Markov_mfpt_calc(peq, M_biased)
        mfpt_biased = mfpts_biased[start_state_working_index, end_state_working_index]

        if try_num % 100 == 0:
            kemeny_constant_check(mfpts_biased, peq)
            print("random try:", try_num, "mfpt:", mfpt_biased)
        if best_mfpt > mfpt_biased:
            best_mfpt = mfpt_biased
            best_params = gaussian_params

    print("best mfpt:", best_mfpt)
    
    #now we use the best params to local optimise the gaussian params

    def mfpt_helper(gaussian_params, M, start_state_working_index = start_state_working_index, end_state_working_index = end_state_working_index, kT=0.5981, working_indices=working_indices):
        #print("Try and Optim from state:", start_state_working_index_xy, "to state:", end_state_working_index_xy)
        total_bias = get_total_bias_2d(x,y, gaussian_params)
        M_biased = np.zeros_like(M)

        #we truncate the total_bias to the working index.
        working_bias = total_bias[working_indices_xy] #say M is in shape[51,51], working bias will be in [51] shape.

        #now we have a discontinues M matrix. we need to apply the bias to the working index.
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                u_ij = working_bias[j] - working_bias[i]
                M_biased[i,j] = M[i,j] * np.exp(-u_ij / (2*0.5981))
            M_biased[i,i] = M[i,i]
        #epsilon_offset = 1e-15
        #M_biased = M_biased / (np.sum(M_biased, axis=0)[:, None] + 1e-15)
        for i in range(M_biased.shape[0]):
            if np.sum(M_biased[i, :]) > 0:
                M_biased[i, :] = M_biased[i, :] / np.sum(M_biased[i, :])
            else:
                M_biased[i, :] = 0
        
        #note our M_biased is in working index. M.shape = (num_working_states, num_working_states)
        peq,F,_,_,_,_ = compute_free_energy(M_biased.T.astype(np.float64), kT=0.5981)
        
        mfpts_biased = Markov_mfpt_calc(peq, M_biased)
        mfpt_biased = mfpts_biased[start_state_working_index, end_state_working_index]
        return mfpt_biased

    res = minimize(mfpt_helper, 
                   best_params, 
                   args=(M,
                         start_state_working_index, 
                         end_state_working_index,
                         working_indices), 
                   #method='Nelder-Mead',
                   method="L-BFGS-B", 
                   bounds= [(0.1, 2)]*num_gaussian + [(0, 2*np.pi)]*num_gaussian + [(0, 2*np.pi)]*num_gaussian + [(0.3, 1.5)]*num_gaussian + [(0.3, 1.5)]*num_gaussian,
                   tol=1e-4)
    
    #print("local optimisation result:", res.x)
    return res.x

def save_CV_total(CV_total, time_tag, prop_index):
    np.save(f"./data/{time_tag}_{prop_index}_CV_total.npy", CV_total[-1])

def save_gaussian_params(gaussian_params, time_tag, prop_index):
    np.save(f"./data/{time_tag}_{prop_index}_gaussian_params.npy", gaussian_params)



def apply_fes(system, particle_idx, gaussian_param=None, pbc = False, name = "FES", amp = 7, mode = "gaussian", plot = False, plot_path = "./fes_visualization.png"):
    """
    this function apply the bias given by the gaussian_param to the system.
    """
    pi = np.pi #we need convert this into nm.
        #at last we add huge barrier at the edge of the box. since we are not using pbc.
    #this is to prevent the particle from escaping the box.
    # if x<0, push the atom back to x=0


    k = 5  # Steepness of the sigmoid curve
    max_barrier = "1e2"  # Scaling factor for the potential maximum
    offset = 0.7 #the offset of the boundary energy barrier.
    # Defining the potentials using a sigmoid function
    left_pot = openmm.CustomExternalForce(f"{max_barrier} * (1 / (1 + exp({k} * x - (-{offset}))))")
    right_pot = openmm.CustomExternalForce(f"{max_barrier} * (1 / (1 + exp(-{k} * (x - (2 * {pi} + {offset})))))")
    bottom_pot = openmm.CustomExternalForce(f"{max_barrier} * (1 / (1 + exp({k} * y - (-{offset}))))")
    top_pot = openmm.CustomExternalForce(f"{max_barrier} * (1 / (1 + exp(-{k} * (y - (2 * {pi} + {offset})))))")

    left_pot.addParticle(particle_idx)
    right_pot.addParticle(particle_idx)
    bottom_pot.addParticle(particle_idx)
    top_pot.addParticle(particle_idx)

    system.addForce(left_pot)
    system.addForce(right_pot)
    system.addForce(bottom_pot)
    system.addForce(top_pot)

    
    #unpack gaussian parameters
    if mode == "gaussian":
        num_gaussians = int(len(gaussian_param)/5)
        A = gaussian_param[0::5] * amp #*7
        x0 = gaussian_param[1::5]
        y0 = gaussian_param[2::5]
        sigma_x = gaussian_param[3::5]
        sigma_y = gaussian_param[4::5]

        #now we add the force for all gaussians.
        energy = "0"
        force = openmm.CustomExternalForce(energy)
        for i in range(num_gaussians):
            if pbc:
                energy = f"A{i}*exp(-periodicdistance(x,0,0, x0{i},0,0)^2/(2*sigma_x{i}^2) - periodicdistance(0,y,0, 0,y0{i},0)^2/(2*sigma_y{i}^2))"
                force = openmm.CustomExternalForce(energy)
            else:
                energy = f"A{i}*exp(-(x-x0{i})^2/(2*sigma_x{i}^2) - (y-y0{i})^2/(2*sigma_y{i}^2))"
                force = openmm.CustomExternalForce(energy)

            #examine the current energy term within force.

            print(force.getEnergyFunction())

            force.addGlobalParameter(f"A{i}", A[i])
            force.addGlobalParameter(f"x0{i}", x0[i])
            force.addGlobalParameter(f"y0{i}", y0[i])
            force.addGlobalParameter(f"sigma_x{i}", sigma_x[i])
            force.addGlobalParameter(f"sigma_y{i}", sigma_y[i])
            force.addParticle(particle_idx)
            #we append the force to the system.
            system.addForce(force)
        if plot:
            #plot the fes.
            x = np.linspace(0, 2*np.pi, 100)
            y = np.linspace(0, 2*np.pi, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            for i in range(num_gaussians):
                Z += A[i] * np.exp(-(X-x0[i])**2/(2*sigma_x[i]**2) - (Y-y0[i])**2/(2*sigma_y[i]**2))
            plt.figure()
            plt.imshow(Z, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=amp *12/7, origin="lower")
            plt.xlabel("x")
            plt.xlim([-1, 2*np.pi+1])
            plt.ylim([-1, 2*np.pi+1])
            plt.ylabel("y")
            plt.title("FES mode = gaussian, pbc=False")
            plt.colorbar()
            plt.savefig(plot_path)
            plt.close()
            fes = Z

    if mode == "multiwell":
        """
        here we create a multiple well potential.
         essentially we deduct multiple gaussians from a flat surface, 
         with a positive gaussian acting as an additional barrier.
         note we have to implement this into openmm CustomExternalForce.
            the x,y is [0, 2pi]
         eq:
            U(x,y) = amp * (1                                                                   #flat surface
                            - A_i*exp(-(x-x0i)^2/(2*sigma_xi^2) - (y-y0i)^2/(2*sigma_yi^2))) ...        #deduct gaussians
                            + A_j * exp(-(x-x0j)^2/(2*sigma_xj^2) - (y-y0j)^2/(2*sigma_yj^2))       #add a sharp positive gaussian
        """
        if pbc:
            raise NotImplementedError("pbc not implemented for multi-well potential.")
        else:
            num_wells = 9
            num_barrier = 1

            #here's the well params
            A_i = np.array([0.9, 0.3, 0.5, 1, 0.2, 0.4, 0.9, 0.9, 0.9]) * amp #this is in kcal/mol.
            x0_i = [1.12, 1, 3, 4.15, 4, 5.27, 5.5, 6, 1] # this is in nm.
            y0_i = [1.34, 2.25, 2.31, 3.62, 5, 4.14, 4.5, 1.52, 5]
            sigma_x_i = [0.5, 0.3, 0.4, 2, 0.9, 1, 0.3, 0.5, 0.5]
            sigma_y_i = [0.5, 0.3, 1, 0.8, 0.2, 0.3, 1, 0.6, 0.7]

            #here's the barrier params
            # for example we define a diagonal barrier at x = pi
            A_j = np.array([0.3]) * amp
            x0_j = [np.pi]
            y0_j = [np.pi]
            sigma_x_j = [3]
            sigma_y_j = [0.3]

            #now we add the force for all gaussians.
            #note all energy is in Kj/mol unit.
            energy = str(amp * 4.184) #flat surface
            force = openmm.CustomExternalForce(energy)
            force.addParticle(particle_idx)
            system.addForce(force)
            for i in range(num_wells):
                energy = f"-A{i}*exp(-(x-x0{i})^2/(2*sigma_x{i}^2) - (y-y0{i})^2/(2*sigma_y{i}^2))"
                force = openmm.CustomExternalForce(energy)

                #examine the current energy term within force.

                print(force.getEnergyFunction())

                force.addGlobalParameter(f"A{i}", A_i[i] * 4.184) #convert kcal to kj
                force.addGlobalParameter(f"x0{i}", x0_i[i])
                force.addGlobalParameter(f"y0{i}", y0_i[i])
                force.addGlobalParameter(f"sigma_x{i}", sigma_x_i[i])
                force.addGlobalParameter(f"sigma_y{i}", sigma_y_i[i])
                force.addParticle(particle_idx)
                #we append the force to the system.
                system.addForce(force)
            
            for i in range(num_barrier):
                energy = f"A{i+num_wells}*exp(-(x-x0{i+num_wells})^2/(2*sigma_x{i+num_wells}^2) - (y-y0{i+num_wells})^2/(2*sigma_y{i+num_wells}^2))"
                force = openmm.CustomExternalForce(energy)

                #examine the current energy term within force.

                print(force.getEnergyFunction())

                force.addGlobalParameter(f"A{i+num_wells}", A_j[i])
                force.addGlobalParameter(f"x0{i+num_wells}", x0_j[i])
                force.addGlobalParameter(f"y0{i+num_wells}", y0_j[i])
                force.addGlobalParameter(f"sigma_x{i+num_wells}", sigma_x_j[i])
                force.addGlobalParameter(f"sigma_y{i+num_wells}", sigma_y_j[i])
                force.addParticle(particle_idx)
                #we append the force to the system.
                system.addForce(force)
            
            if plot:
                plot_3d = True
                #plot the fes.
                x = np.linspace(0, 2*np.pi, 100)
                y = np.linspace(0, 2*np.pi, 100)
                X, Y = np.meshgrid(x, y)
                Z = np.zeros_like(X)
                Z += amp * 4.184 #flat surface
                for i in range(num_wells):
                    Z -= A_i[i] * np.exp(-(X-x0_i[i])**2/(2*sigma_x_i[i]**2) - (Y-y0_i[i])**2/(2*sigma_y_i[i]**2))
                for i in range(num_barrier):
                    Z += A_j[i] * np.exp(-(X-x0_j[i])**2/(2*sigma_x_j[i]**2) - (Y-y0_j[i])**2/(2*sigma_y_j[i]**2))
                
                #add the x,y boundary energy barrier.
                total_energy_barrier = np.zeros_like(X)
                total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (X - (-offset))))) #left
                total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (X - (2 * pi + offset))))) #right
                total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (Y - (-offset)))))
                total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (Y - (2 * pi + offset)))))
                Z += total_energy_barrier
                Z = Z - Z.min()

                if plot_3d:
                    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                    #tight layout.
                    plt.tight_layout(pad=2.0)
                    plt.subplots_adjust(bottom=0.2)
                    surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", linewidth=0.2, rstride=5, cstride=5, alpha=0.8)
                    ax.contourf(X, Y, Z, zdir='z', offset=0, cmap="coolwarm")

                    ax.set_xlabel("x (nm)")
                    ax.set_ylabel("y (nm)")
                    #ax.set_zlabel("U (kcal/mol)")
                    #ax.set_zlim([0, amp * 12/7])
                    #ax.set_title("FES mode = multiwell, pbc=False")
                    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
                    cbar.set_label("U (kcal/mol)")
                    plt.savefig(plot_path, dpi=800)
                else:
                    plt.figure()
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
                    plt.savefig(plot_path, dpi=800)
                    plt.close()
                fes = Z
            
    if mode == "funnel":
        """
        this is funnel like potential.
        we start wtih a flat fes, then add/deduct sphrical gaussians
        eq:
            U = 0.7* amp * cos(2 * p * (sqrt((x-pi)^2 + (y-pi)^2))) #cos function. periodicity determines the num of waves.
            - amp exp(-((x-pi)^2+(y-pi)^2))
            + 0.4*amp*((x-pi/8)^2 + (y-pi/8)^2)
        """
        if pbc:
            raise NotImplementedError("pbc not implemented for funnel potential.")
        else:
            plot_3d = False
            periodicity = 8
            energy = f"0.7*{amp} * cos({periodicity} * (sqrt((x-{pi})^2 + (y-{pi})^2))) - 0.6* {amp} * exp(-((x-{pi})^2+(y-{pi})^2)) + 0.4*{amp}*((x-{pi}/8)^2 + (y-{pi}/8)^2)"
            
            force = openmm.CustomExternalForce(energy)
            force.addParticle(particle_idx)
            system.addForce(force)
            if plot:
                if plot_3d:
                    import plotly.graph_objs as go

                    # Define the x, y, and z coordinates
                    x = np.linspace(0, 2*np.pi, 100)
                    y = np.linspace(0, 2*np.pi, 100)
                    X, Y = np.meshgrid(x, y)
                    Z = np.zeros_like(X)
                    Z += 0.9* amp * np.cos(periodicity * (np.sqrt((X-np.pi)**2 + (Y-np.pi)**2))) #cos function. periodicity determines the num of waves.
                    Z -= 0.6* amp * np.exp(-((X-np.pi)**2/0.5+(Y-np.pi)**2)/0.5)
                    Z += 0.4*amp*(((X-np.pi)/8)**2 + ((Y-np.pi)/8)**2)

                    #add the x,y boundary energy barrier.
                    total_energy_barrier = np.zeros_like(X)
                    total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (X - 0)))) #left
                    total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (X - 2 * pi)))) #right
                    total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (Y - 0))))
                    total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (Y - 2 * pi))))
                    Z += total_energy_barrier

                    # Create the 3D contour plot
                    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, cmin = 0, cmax = amp *12/7)])
                    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))
                    fig.update_layout(title='FES mode = funnel, pbc=False', autosize=True,
                                    width=800, height=800,
                                    scene = {
                                        "xaxis": {"nticks": 5},
                                        "yaxis": {"nticks": 5},
                                        "zaxis": {"nticks": 5},
                                        "camera_eye": {"x": 1, "y": 1, "z": 0.4},
                                        "aspectratio": {"x": 1, "y": 1, "z": 0.4}
                                    }
                                    )
                                    #margin=dict(l=65, r=50, b=65, t=90))
                    #save fig.
                    fig.write_image(plot_path)
                    fes = Z
                    
                else:
                    #plot the fes.
                    x = np.linspace(0, 2*np.pi, 100)
                    y = np.linspace(0, 2*np.pi, 100)
                    X, Y = np.meshgrid(x, y)
                    Z = np.zeros_like(X)
                    Z += 0.4* amp * np.cos(periodicity * (np.sqrt((X-np.pi)**2 + (Y-np.pi)**2))) #cos function. periodicity determines the num of waves.
                    Z += 0.7* amp * np.exp(-((X-np.pi)**2/0.5+(Y-np.pi)**2/0.5))
                    Z += 0.2*amp*(((X-np.pi)/8)**2 + ((Y-np.pi)/8)**2)

                    #add the x,y boundary energy barrier.
                    total_energy_barrier = np.zeros_like(X)
                    total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (X - 0)))) #left
                    total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (X - 2 * pi)))) #right
                    total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (Y - 0))))
                    total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (Y - 2 * pi))))
                    Z += total_energy_barrier

                    plt.figure()
                    plt.imshow(Z, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=amp *12/7, origin="lower")
                    plt.xlabel("x")
                    plt.xlim([-1, 2*np.pi+1])
                    plt.ylim([-1, 2*np.pi+1])
                    plt.ylabel("y")
                    plt.title("FES mode = funnel, pbc=False")
                    plt.colorbar()
                    plt.savefig(plot_path)
                    plt.close()
                    fes = Z

    return system, fes #return the system and the fes (2D array for plotting.)

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

def apply_bias(system, particle_idx, gaussian_param, pbc = False, name = "BIAS", num_gaussians = 20):
    """
    this applies a bias using customexternal force class. similar as apply_fes.
    note this leaves a set of global parameters Ag, x0g, y0g, sigma_xg, sigma_yg.
    as these parameters can be called and updated later.
    note this is done while preparing the system before assembling the context.
    """
    assert len(gaussian_param) == 5 * num_gaussians, "gaussian_param should be in A, x0, y0, sigma_x, sigma_y format."

    #unpack gaussian parameters gaussian_params = np.concatenate((a, bx, by, cx, cy))
    num_gaussians = len(gaussian_param)//5
    A = gaussian_param[:num_gaussians]
    x0 = gaussian_param[num_gaussians:2*num_gaussians]
    y0 = gaussian_param[2*num_gaussians:3*num_gaussians]
    sigma_x = gaussian_param[3*num_gaussians:4*num_gaussians]
    sigma_y = gaussian_param[4*num_gaussians:5*num_gaussians]

    #now we add the force for all gaussians. with num_gaussians terms.
    energy = "0"
    force = openmm.CustomExternalForce(energy)
    for i in range(num_gaussians):
        if pbc:
            raise NotImplementedError("pbc not implemented for gaussian potential.")
            energy = f"Ag{i}*exp(-periodicdistance(x,0,0, x0g{i},0,0)^2/(2*sigma_xg{i}^2) - periodicdistance(0,y,0, 0,y0g{i},0)^2/(2*sigma_yg{i}^2))"
            force = openmm.CustomExternalForce(energy)
        else:
            energy = f"Ag{i}*exp(-(x-x0g{i})^2/(2*sigma_xg{i}^2) - (y-y0g{i})^2/(2*sigma_yg{i}^2))" #in openmm unit, kj/mol, nm.
            force = openmm.CustomExternalForce(energy)

        #examine the current energy term within force.

        print(force.getEnergyFunction())

        force.addGlobalParameter(f"Ag{i}", A[i] * 4.184) #convert to kJ/mol
        force.addGlobalParameter(f"x0g{i}", x0[i]) #convert to nm
        force.addGlobalParameter(f"y0g{i}", y0[i])
        force.addGlobalParameter(f"sigma_xg{i}", sigma_x[i])
        force.addGlobalParameter(f"sigma_yg{i}", sigma_y[i])
        force.addParticle(particle_idx)
        #we append the force to the system.
        system.addForce(force)
    
    print("system added with bias.")
    return system

def update_bias(simulation, gaussian_param, name = "BIAS", num_gaussians = 20):
    """
    given the gaussian_param, update the bias
    note this requires the context object. or a simulation object.
    # the context object can be accessed by simulation.context.
    """
    assert len(gaussian_param) == 5 * num_gaussians, "gaussian_param should be in A, x0, y0, sigma_x, sigma_y format."

    #unpack gaussian parameters gaussian_params = np.concatenate((a, bx, by, cx, cy))
    num_gaussians = len(gaussian_param)//5
    A = gaussian_param[:num_gaussians]
    x0 = gaussian_param[num_gaussians:2*num_gaussians]
    y0 = gaussian_param[2*num_gaussians:3*num_gaussians]
    sigma_x = gaussian_param[3*num_gaussians:4*num_gaussians]
    sigma_y = gaussian_param[4*num_gaussians:5*num_gaussians]

    #now we update the GlobalParameter for all gaussians. with num_gaussians terms. and update them in the system.
    #note globalparameter does NOT need to be updated in the context.
    for i in range(num_gaussians):
        simulation.context.setParameter(f"Ag{i}", A[i] * 4.184) #convert to kJ/mol
        simulation.context.setParameter(f"x0g{i}", x0[i]) #convert to nm
        simulation.context.setParameter(f"y0g{i}", y0[i])
        simulation.context.setParameter(f"sigma_xg{i}", sigma_x[i])
        simulation.context.setParameter(f"sigma_yg{i}", sigma_y[i])
    
    print("system bias updated")
    return simulation
