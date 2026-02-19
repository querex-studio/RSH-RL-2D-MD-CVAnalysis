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

from openmm.unit import Quantity
from openmm import Vec3

import config
import csv
from util import *

#first we initialize the system.
# topology

#for amp applied on fes. note the gaussian parameters for fes is normalized.
if __name__ == "__main__":
    elem = Element(0, "X", "X", 1.0)
    top = Topology()
    top.addChain()
    top.addResidue("xxx", top._chains[0])
    top.addAtom("X", elem, top._chains[0]._residues[0])
    mass = 12.0 * unit.amu
    end_state_xyz = config.end_state.value_in_unit_system(openmm.unit.md_unit_system)[0]
        
    #starting point as [1.29,-1.29,0.0]
    for i_sim in range(config.num_sim):
        time_tag = time.strftime("%Y%m%d-%H%M%S")
        system = openmm.System()
        system.addParticle(mass)

        #potential setup
        #first we load the gaussians from the file.
        # params comes in A, x0, y0, sigma_x, sigma_y format.

        #gaussian_param = np.loadtxt("./params/gaussian_fes_param.txt") 

        system, fes = apply_fes(system = system, 
                        particle_idx=0, 
                        gaussian_param = None, 
                        pbc = config.pbc, 
                        name = "FES", 
                        amp=config.amp, 
                        mode = config.fes_mode,
                        plot = True)
        #save the fes to a matlab .mat file.
        if True:
            import scipy.io
            scipy.io.savemat(f"./fes_{time_tag}.mat", {'fes':fes})
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
                                            config.stepsize_unbias) #note the stepsize is 0.2 ps here.

        #before run, last check pbc:
        #print(system.getDefaultPeriodicBoxVectors())

        #CUDA
        platform = openmm.Platform.getPlatformByName('CUDA')

        #run the simulation
        simulation = openmm.app.Simulation(top, system, integrator, platform)
        simulation.context.setPositions(config.start_state)
        simulation.context.setVelocitiesToTemperature(300*unit.kelvin)

        #simulation.minimizeEnergy() #disabled for testing energy barrier.
        if config.pbc:
            simulation.context.setPeriodicBoxVectors(a,b,c)


        pos_traj = []#np.zeros([int(config.sim_steps_unbiased/config.dcdfreq), 3]) #note config.sim_steps_unbiased represents the maximum number of steps here in the unbiased case.
        reach = None
        i=0
        
        file_handle = open(f'trajectory/unbias/{time_tag}_unbias_traj.dcd', 'wb')
        dcd_file = openmm.app.DCDFile(file_handle, top, dt = config.stepsize_unbias)
        for i in tqdm(range(int(config.sim_steps_unbiased/config.dcdfreq))):
        #while reach is None:
            simulation.step(config.dcdfreq)
            state = simulation.context.getState(getPositions=True, enforcePeriodicBox=config.pbc)
            pos_traj.append(state.getPositions(asNumpy=True)[0,:])

            #we determine if we reached endstate.
            cur_pos = np.array(pos_traj[-1])
            
            #i+=1
            if np.linalg.norm(cur_pos - end_state_xyz) < 0.1:
                print("reached end state")
                reach = True
            
            if i % 1000 == 0: 
                print("simulation step: ", i * config.dcdfreq)
                ### VISUALIZATION ###
                x,y = np.meshgrid(np.linspace(0, 2*np.pi, 100), np.linspace(0, 2*np.pi, 100)) #fes in shape [100,100]

                plt.figure()
                plt.imshow(fes, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=config.amp * 12/7 * 4.184, origin = "lower")
                plt.colorbar()
                #we only take the most recent i*2000 steps.
                #plt.scatter(np.array(pos_traj)[::3,0], np.array(pos_traj)[::3,1], s=0.5, alpha=0.5, c='yellow')
                plt.scatter(np.array(pos_traj)[::5,0], np.array(pos_traj)[::5,1], s=3.5, alpha=0.5, c='black')
                plt.xlabel("x")
                plt.xlim([-1, 2*np.pi+1])
                plt.ylim([-1, 2*np.pi+1])
                plt.ylabel("y")
                plt.title(f"Unbiased Trajectory, pbc={config.pbc}")
                #plt.show()
                plt.savefig(f"./figs/unbias/unbias_traj_{time_tag}_simstep_{i*config.dcdfreq}.png")
                plt.close()
            
            #write dcd
            dcd_file.writeModel(state.getPositions(asNumpy=True))

    
        #we record the steps used to reach end state.
        steps_to_endstate = i * config.dcdfreq
        print(f"steps to end state: {steps_to_endstate}")

        with open("total_steps_unbiased.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([steps_to_endstate])
        
        #save the traj and plot it.
        pos_traj = np.array(pos_traj)
        np.savetxt(f"visited_states/{time_tag}_langevin_2D_unbias_traj.txt", pos_traj)

        #here we plot.
        
    print("all done")
