#this is a langevin simulator in OPENMM.
# we put a particle in a box and simulate it with Langevin dynamics.
# the external force is defined using a function digitizing the phi/psi fes of dialanine.

import numpy as np
import matplotlib.pyplot as plt

import time

from tqdm import tqdm

import openmm
from openmm import unit
from openmm.app.topology import Topology
from openmm.app.element import Element
import mdtraj
import csv

import config
from dham import *
from util import *

def propagate(simulation,
              prop_index, 
              pos_traj,   #this records the trajectory of the particle. in shape: [prop_index, sim_steps, 3]
              steps=config.propagation_step,
              dcdfreq=config.dcdfreq_mfpt,
              stepsize=config.stepsize,
              num_bins=config.num_bins,
              pbc=config.pbc,
              time_tag = None,
              top=None,
              reach=None
              ):
    """
    here we use the openmm context object to propagate the system.
    save the CV and append it into the CV_total.
    use the DHAM_it to process CV_total, get the partially observed Markov matrix from trajectory.
    return the current position, the CV_total, and the partially observed Markov matrix.
    """
    
    file_handle = open(f"./trajectory/explore/{time_tag}_langevin_sim_explore_{prop_index}.dcd", 'bw')
    dcd_file = openmm.app.dcdfile.DCDFile(file_handle, top, dt = stepsize) #note top is no longer a global pararm, we need pass this.
    for _ in tqdm(range(int(steps/dcdfreq)), desc=f"Propagation {prop_index}"):
        simulation.integrator.step(dcdfreq)
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=pbc)
        dcd_file.writeModel(state.getPositions(asNumpy=True))
    file_handle.close()

    #save the top to pdb.
    with open(f"./trajectory/explore/{time_tag}_langevin_sim_explore_{prop_index}.pdb", 'w') as f:
        openmm.app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)
    
    #we load the pdb and pass it to mdtraj_top
    mdtraj_top = mdtraj.load(f"./trajectory/explore/{time_tag}_langevin_sim_explore_{prop_index}.pdb")

    #use mdtraj to get the coordinate of the particle.
    traj = mdtraj.load_dcd(f"./trajectory/explore/{time_tag}_langevin_sim_explore_{prop_index}.dcd", top = mdtraj_top)#top = mdtraj.Topology.from_openmm(top)) #this will yield error because we using imaginary element X.
    coor = traj.xyz[:,0,:] #[all_frames,particle_index,xyz] # we grep the particle 0.

    #we digitize the x, y coordinate into meshgrid (0, 2pi, num_bins)
    x = np.linspace(0, 2*np.pi, num_bins) #hardcoded.
    y = np.linspace(0, 2*np.pi, num_bins)
    #we digitize the coor into the meshgrid.
    coor_xy = coor.squeeze()[:,:2] #we only take the x, y coordinate.
    coor_x_digitized = np.digitize(coor_xy[:,0], x)#quick fix for digitized to 0 or maximum error. #note this is in coordinate space np.linspace(0, 2*np.pi, num_bins)
    coor_y_digitized = np.digitize(coor_xy[:,1], y)
    coor_xy_digitized = np.stack([coor_x_digitized, coor_y_digitized], axis=1) #shape: [all_frames, 2]

    #changed order = F, temporary fix for the DHAM?
    #print(x)
    coor_xy_digitized_ravel = np.array([np.ravel_multi_index(coor_temp, (num_bins, num_bins), order='F') for coor_temp in coor_xy_digitized]) #shape: [all_frames,]

    #we test.
    if False:
        coor_xy_digitized_ravel_unravel = np.array([np.unravel_index(x, (num_bins, num_bins), order='C') for x in coor_xy_digitized_ravel]) #shape: [all_frames, 2]

        x,y = np.meshgrid(np.linspace(0, 2*np.pi, num_bins), np.linspace(0, 2*np.pi, num_bins))

        plt.figure()
        #
        plt.xlim([0, 2*np.pi])
        plt.ylim([0, 2*np.pi])
        plt.savefig("./test.png")
        plt.close()
    #we append the coor_xy_digitized into the pos_traj.
    pos_traj[prop_index,:] = coor_xy_digitized_ravel.astype(np.int64)

    #we take all previous ravelled position from pos_traj and append it to the total list, feed into the DHAM.
    coor_xy_digital_ravelled_total = pos_traj[:prop_index+1,:] #shape: [prop_index+1, all_frames * 1]
    coor_xy_digital_ravelled_total = coor_xy_digital_ravelled_total.reshape(-1,1) #shape: [prop_index+1 * all_frames, 1]

    #here we load all the gaussian_params from previous propagations.
    #size of gaussian_params: [num_propagation, num_gaussian, 3] (a,b,c),
    # note for 2D this would be [num_propagation, num_gaussian, 5] (a,bx,by,cx,cy)
    gaussian_params = np.zeros([prop_index+1, config.num_gaussian, 5])
    for i in range(prop_index+1):
        gaussian_params[i,:,:] = np.loadtxt(f"./params/{time_tag}_gaussian_fes_param_{i}.txt").reshape(-1,5)
        print(f"gaussian_params for propagation {i} loaded.")

    #here we use the DHAM.
    F_M, MM = DHAM_it(coor_xy_digital_ravelled_total.reshape(prop_index+1, -1, 1), gaussian_params, T=300, lagtime=1, numbins=num_bins, time_tag=time_tag, prop_index=prop_index)
    cur_pos = coor_xy_digital_ravelled_total[-1] #the current position of the particle, in ravelled 1D form.
    
    #determine if the particle has reached the target state.
    end_state_xyz = config.end_state.value_in_unit_system(openmm.unit.md_unit_system)[0]
    end_state_xy = end_state_xyz[:2]
    for index_d, d in enumerate(coor_xy):
        #if the distance of current pos is the config.target_state, we set reach to index_d.
        target_distance = np.linalg.norm(d - end_state_xy)
        if target_distance < 0.1:
            reach = index_d * config.dcdfreq_mfpt

    return cur_pos, pos_traj, MM, reach, F_M

def get_working_MM(M):
    zero_rows = np.where(~M.any(axis=1))[0]
    zero_cols = np.where(~M.any(axis=0))[0]

    keep_indices = np.setdiff1d(range(M.shape[0]), np.union1d(zero_rows, zero_cols))
    M_work = M[np.ix_(keep_indices, keep_indices)]
    return M_work, keep_indices

def find_closest_index(working_indices, final_index, N):
    """
    returns the farest index in 1D.

    here we find the closest state to the final state.
    first we unravel all the index to 2D.
    then we use the lowest RMSD distance to find the closest state.
    then we ravel it back to 1D.
    note: for now we only find the first-encounted closest state.
          we can create a list of all the closest states, and then choose random one.
    """
    def rmsd_dist(a, b):
        return np.sqrt(np.sum((a-b)**2))
    working_x, working_y = np.unravel_index(working_indices, (N,N), order='C')
    working_states = np.stack((working_x, working_y), axis=1)
    final_state = np.unravel_index(final_index, (N,N), order='C')
    closest_state = working_states[0]
    for i in range(len(working_states)):
        if rmsd_dist(working_states[i], final_state) < rmsd_dist(closest_state, final_state):
            closest_state = working_states[i]
        
    closest_index = np.ravel_multi_index(closest_state, (N,N), order='C')
    return closest_index

def DHAM_it(CV, gaussian_params, T=300, lagtime=2, numbins=150, prop_index=0, time_tag=None):
    """
    intput:
    CV: the collective variable we are interested in. now it's 2d.
    gaussian_params: the parameters of bias potential. (in our case the 10-gaussian params)
     format: (a,bx, by,cx,cy)
    T: temperature 300

    output:
    the Markov Matrix
    Free energy surface probed by DHAM.
    """
    d = DHAM(gaussian_params)
    d.setup(CV, T, prop_index=prop_index, time_tag=time_tag)

    d.lagtime = lagtime
    d.numbins = numbins #num of bins, arbitrary.
    results = d.run(biased = True, plot=True)
    return results

if __name__ == "__main__":
    elem = Element(0, "X", "X", 1.0)
    top = Topology()
    top.addChain()
    top.addResidue("xxx", top._chains[0])
    top.addAtom("X", elem, top._chains[0]._residues[0])
    mass = 12.0 * unit.amu
    for i_sim in range(config.num_sim):
    #def simulate_once():
        print("system initializing")
        #print out all the config.
        print("config: ", config.__dict__)
        
        time_tag = time.strftime("%Y%m%d-%H%M%S")

        #print current time tag.
        print("time_tag: ", time_tag)

        system = openmm.System() #we initialize the system every
        system.addParticle(mass)
        #gaussian_param = np.loadtxt("./params/gaussian_fes_param.txt")
        system, fes = apply_fes(system = system, particle_idx=0, gaussian_param = None, pbc = config.pbc, amp = config.amp, name = "FES", mode=config.fes_mode, plot = True)
        z_pot = openmm.CustomExternalForce("1e3 * z^2") # very large force constant in z
        z_pot.addParticle(0)
        system.addForce(z_pot) #on z, large barrier

        #pbc section
        if config.pbc:
            a = unit.Quantity((2*np.pi*unit.nanometers, 0*unit.nanometers, 0*unit.nanometers))
            b = unit.Quantity((0*unit.nanometers, 2*np.pi*unit.nanometers, 0*unit.nanometers))
            c = unit.Quantity((0*unit.nanometers, 0*unit.nanometers, 1*unit.nanometers)) # atom not moving in z so we set it to 1 nm
            system.setDefaultPeriodicBoxVectors(a,b,c)

        #integrator
        integrator = openmm.LangevinIntegrator(300*unit.kelvin, 
                                            1.0/unit.picoseconds, 
                                            0.002*unit.picoseconds)

        num_propagation = int(config.sim_steps/config.propagation_step)
        frame_per_propagation = int(config.propagation_step/config.dcdfreq_mfpt)
        #this stores the digitized, ravelled, x, y coordinates of the particle, for every propagation.
        pos_traj = np.zeros([num_propagation, frame_per_propagation]) #shape: [num_propagation, frame_per_propagation]


        x,y = np.meshgrid(np.linspace(0, 2*np.pi, config.num_bins), np.linspace(0, 2*np.pi, config.num_bins))


        #save the top as pdb.
        with open(f"./trajectory/explore/{time_tag}_langevin_sim_explore.pdb", 'w') as f:
            openmm.app.PDBFile.writeFile(top, config.start_state, f)
            

        #we start propagation.
        #note num_propagation = config.sim_steps/config.propagation_step
        reach = None
        i_prop = 0
        #for i_prop in range(num_propagation):
        while reach is None:
            if i_prop >= num_propagation:
                print("propagation number exceeds num_propagation, break")
                break
            if i_prop == 0:
                print("propagation 0 starting")
                gaussian_params = random_initial_bias_2d(initial_position = config.start_state, num_gaussians = config.num_gaussian)
                np.savetxt(f"./params/{time_tag}_gaussian_fes_param_{i_prop}.txt", gaussian_params)
                #we apply the initial gaussian bias (v small) to the system
                system = apply_bias(system = system, particle_idx=0, gaussian_param = gaussian_params, pbc = config.pbc, name = "BIAS", num_gaussians = config.num_gaussian)

                #create simulation object, this create a context object automatically.
                # when we need to pass a context object, we can pass simulation instead.
                simulation = openmm.app.Simulation(top, system, integrator, config.platform)
                simulation.context.setPositions(config.start_state)
                simulation.context.setVelocitiesToTemperature(300*unit.kelvin)

                simulation.minimizeEnergy()
                if config.pbc:
                    simulation.context.setPeriodicBoxVectors(a,b,c)

                #now we propagate the system, i.e. run the langevin simulation.
                cur_pos, pos_traj, MM, reach, F_M = propagate(simulation = simulation,
                                                                    prop_index = i_prop,
                                                                    pos_traj = pos_traj,
                                                                    steps=config.propagation_step,
                                                                    dcdfreq=config.dcdfreq_mfpt,
                                                                    stepsize=config.stepsize,
                                                                    num_bins=config.num_bins,
                                                                    pbc=config.pbc,
                                                                    time_tag = time_tag,
                                                                    top=top,
                                                                    reach=reach
                                                                    )

                working_MM, working_indices = get_working_MM(MM)

                final_coor = config.end_state.value_in_unit_system(openmm.unit.md_unit_system)[0][:2]
                final_coor_digitized = np.digitize(final_coor, np.linspace(0, 2*np.pi, config.num_bins))
                final_index = np.ravel_multi_index(final_coor_digitized, (config.num_bins, config.num_bins), order='C')
                closest_index = find_closest_index(working_indices, final_index, config.num_bins)
                i_prop += 1
            else:

                print(f"propagation number {i_prop} starting")

                #find the most visited state in last propagation.
                last_traj = pos_traj[i_prop-1,:]
                most_visited_state = np.argmax(np.bincount(last_traj.astype(int))) #this is in digitized, ravelled form.

                gaussian_params = try_and_optim_M(working_MM,
                                                working_indices = working_indices,
                                                N = config.num_bins,
                                                num_gaussian = config.num_gaussian,
                                                start_index = most_visited_state,
                                                end_index = closest_index,
                                                plot = False,
                                                )
                
                #save the gaussian_params
                np.savetxt(f"./params/{time_tag}_gaussian_fes_param_{i_prop}.txt", gaussian_params)

                #apply the gaussian_params to openmm system.
                simulation = update_bias(simulation = simulation,
                                        gaussian_param = gaussian_params,
                                        name = "BIAS",
                                        num_gaussians=config.num_gaussian,
                                        )
                
                #we propagate system again
                cur_pos, pos_traj, MM, reach, F_M = propagate(simulation = simulation,
                                                                    prop_index = i_prop,
                                                                    pos_traj = pos_traj,
                                                                    steps=config.propagation_step,
                                                                    dcdfreq=config.dcdfreq_mfpt,
                                                                    stepsize=config.stepsize,
                                                                    num_bins=config.num_bins,
                                                                    pbc=config.pbc,
                                                                    time_tag = time_tag,
                                                                    top=top,
                                                                    reach=reach
                                                                    )
                
                if True:
                        #here we calculate the total bias given the optimized gaussian_params
                        x_total_bias, y_total_bias = np.meshgrid(np.linspace(0, 2*np.pi, config.num_bins), np.linspace(0, 2*np.pi, config.num_bins)) # shape: [num_bins, num_bins]
                        total_bias = get_total_bias_2d(x_total_bias,y_total_bias, gaussian_params) * 4.184 #convert to kcal/mol
                        plt.figure()
                        plt.imshow(total_bias, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=config.amp *12/7 * 4.184, origin="lower")
                        plt.colorbar()
                        plt.savefig(f"./figs/explore/{time_tag}_total_bias_{i_prop}.png")
                        plt.close()
                        total_bias_big = get_total_bias_2d(np.linspace(0, 2*np.pi, 100), np.linspace(0, 2*np.pi, 100), gaussian_params)* 4.184 #convert to kcal/mol
                        
                        #here we plot the reconstructed fes from MM.
                        # we also plot the unravelled most_visited_state and closest_index.
                        most_visited_state_unravelled = np.unravel_index(most_visited_state, (config.num_bins, config.num_bins), order='C')
                        closest_index_unravelled = np.unravel_index(closest_index, (config.num_bins, config.num_bins), order='C')
                        plt.figure()
                        plt.imshow(np.reshape(F_M,(config.num_bins, config.num_bins), order = "C")*4.184, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=config.amp *12/7 * 4.184, origin="lower")
                        plt.plot(x[most_visited_state_unravelled[0], most_visited_state_unravelled[1]], y[most_visited_state_unravelled[0], most_visited_state_unravelled[1]], marker='o', markersize=3, color="blue", label = "most visited state (local start)")
                        plt.plot(x[closest_index_unravelled[0], closest_index_unravelled[1]], y[closest_index_unravelled[0], closest_index_unravelled[1]], marker='o', markersize=3, color="red", label = "closest state (local target)")
                        plt.legend()
                        plt.colorbar()
                        plt.savefig(f"./figs/explore/{time_tag}_reconstructed_fes_{i_prop}.png")
                        plt.close()

                        #we plot here to check the original fes, total_bias and trajectory.
                    
                        #we add the total bias to the fes.
                        #fes += total_bias_big
                        plt.figure()
                        plt.imshow(fes, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=config.amp * 12/7 * 4.184, origin="lower")
                        plt.colorbar()
                        plt.xlabel("x")
                        #plt.xlim([-1, 2*np.pi+1])
                        #plt.ylim([-1, 2*np.pi+1])
                        plt.ylabel("y")
                        plt.title("FES mode = multiwell, pbc=False")
                        
                        #additionally we plot the trajectory.
                        # first we process the pos_traj into x, y coordinate.
                        # we plot all, but highlight the last prop_step points with higher alpha.

                        pos_traj_flat = pos_traj[:i_prop, :].astype(np.int64).squeeze() #note this is digitized and ravelled.
                        x_unravel, y_unravel = np.unravel_index(pos_traj_flat, (config.num_bins, config.num_bins), order='F') #note the traj is temporary ravelled in F order to adapt the DHAM. #shape: [all_frames, 2]
                        
                        pos_traj_flat_last = pos_traj[i_prop:, :].astype(np.int64).squeeze()
                        x_unravel_last, y_unravel_last = np.unravel_index(pos_traj_flat_last, (config.num_bins, config.num_bins), order='F')
                        
                        grid = np.linspace(0, 2*np.pi, config.num_bins)
                        plt.scatter(grid[x_unravel], grid[y_unravel], s=3.5, alpha=0.3, c='black')
                        plt.scatter(grid[x_unravel_last], grid[y_unravel_last], s=3.5, alpha=0.8, c='yellow')                                     
                        plt.savefig(f"./figs/explore/{time_tag}_fes_traj_{i_prop}.png")
                        plt.close()    

                #update working_MM and working_indices
                working_MM, working_indices = get_working_MM(MM)
                #update closest_index
                closest_index = find_closest_index(working_indices, final_index, config.num_bins)
                i_prop += 1

        #we have reached target state, thus we record the steps used.
        total_steps = i_prop * config.propagation_step + reach * config.dcdfreq_mfpt
        print("total steps used: ", total_steps)

        with open("./total_steps_mfpt.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([total_steps])

        #save the pos_traj
        np.savetxt(f"./visited_states/{time_tag}_pos_traj.txt", pos_traj)

    """from multiprocessing import Pool
    
    multi_process_result = []
    for _ in range(config.num_sim//config.NUMBER_OF_PROCESSES):
        with Pool(config.NUMBER_OF_PROCESSES) as p:
            multi_process_result.extend(p.map(simulate_once, range(config.NUMBER_OF_PROCESSES)))
"""
print("all done")
