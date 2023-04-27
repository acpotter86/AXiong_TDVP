import numpy as np
import time as time
import utilities_MPS as mps
from utilities import SU4_Circuits
np.set_printoptions(precision=6,threshold=1e-6)
from hamiltonian import model_mpo
import functools as f
from scipy.integrate import solve_ivp

def evolution_variational(H0, H1, var_circuits, phys_sites, num_qubits, num_steps, time_step):
    # Each physical site parametrized by some number of circuits
    num_circuits = var_circuits.circ_len

    L = phys_sites
    N = num_qubits

    # Initialize parameter array
    var_params = np.zeros((num_steps+1, L, num_circuits))
    var_params[0] = var_circuits.get_params()

    # Collect matrix product of state at each time step
    var_circuits_all = [[] for n in range(num_steps+1)]
    var_circuits_all[0] = SU4_Circuits(L, var_params[0], N, 'direct')

    # Time evolution
    for t in range(num_steps):

        # Calculate derivatives overlap 
        var_params_deriv = np.zeros((L,num_circuits))
        
        # Calculate derivatives overlap 
        def deriv_overlap(i,j1,j2):
            bra = [np.conj(var_circuits.derivs_list()[i][j1]).T if i==l else np.conj(var_circuits.u_list()[l]).T for l in range(L)]
            ket = [var_circuits.derivs_list()[i][j2] if i==l else var_circuits.u_list()[l] for l in range(L-1,-1,-1)]
            dr = f.reduce(lambda a, b: a @ b, np.concatenate((bra, ket)))
            return 2 * np.real(dr[0][0])

        # Calculate energy gradient
        def H_gradient(i,j):
            H = (1-t/num_steps) * H0 + t/num_steps * H1
            bra = [np.conj(var_circuits.derivs_list()[i][j]).T if i==l else np.conj(var_circuits.u_list()[l]).T for l in range(L)]
            ket = [var_circuits.u_list()[l] for l in range(L-1, -1, -1)]
            dE = f.reduce(lambda a, b: a @ b, np.concatenate((bra, [H], ket)))
            return 2 * np.imag(dE[0][0])

        for i in range(L):
            # For each site there is a matrix C
            # [C]_{ij} is <du/dtheta i | du/dtheta j>
            C = [[deriv_overlap(i, j1, j2) for j2 in range(num_circuits)] for j1 in range(num_circuits)]  
            H_grad = [H_gradient(i, j) for j in range(num_circuits)]
                
            # Solve C d\theta = H_grad and obtain d\theta
            var_params_deriv[i] = np.linalg.pinv(C, rcond=1e-14) @ H_grad
            # var_params_deriv[i] = - np.around(H_grad, 14)
            
        # Update the next row of variational parameters
        var_params[t+1] = var_params[t] + time_step * var_params_deriv

        # update the variational circuits with new parameters
        var_circuits = var_circuits.update_params(var_params[t+1])
        var_circuits_all[t+1] = SU4_Circuits(L, var_params[t+1], N, 'direct')

    # return the variational parameters for plotting
    return var_params, var_circuits_all


def evolution_hamiltonian(H, var_circuits, num_steps, time_step=1):
    num_circuits = len(var_circuits)
    var_circuits_ket = [var_circuits[n].ket() for n in range(num_circuits)]

    matrices = [[] for i in range(num_steps+1)]
    matrices[0] = f.reduce(lambda a, b: a @ b, var_circuits_ket)

    # Time evolution
    for t in range(num_steps):

        # derivative is -j H dt |psi>
        matrices[t+1] = matrices[t] - 1j * time_step * H[0] @ matrices[t]

    # return the variational parameters for plotting
    return matrices


def evolution_adiabatic_mps_ode_solver(var_params_ini, phys_sites, num_qubits, t_total, step_size):
    # Each physical site parametrized by some number of circuits
    num_circuits = 8

    # Number of sites
    L = phys_sites
    N = num_qubits

    def get_var_params_deriv(t, var_params):
        # Reshape the input var parameters and reprepare state
        var_params = var_params.reshape(L, num_circuits)
        var_circuits = SU4_Circuits(L, var_params, N) 

        # Initialize array derivatives overlap 
        var_params_deriv = np.zeros((L, num_circuits))
        
        def deriv_overlap(i1, j1, i2, j2):
            return mps.derivative_overlap_finite(var_circuits.u_list(), var_circuits.derivs_list(), L, [(i1, j1),(i2, j2)])
        
        # Calculate the Hamiltonian at that time step
        H_mps = model_mpo.adiabatic_z_zz(L, t/t_total)

        # Calculate energy gradient
        _, H_gradient = mps.Energy_finite(var_circuits.u_list(), var_circuits.derivs_list(), L, H_mps, burnin=0)

        # Put overlap terms and energy gradient terms in matrices
        for i in range(L):
            # For each site there is a matrix C
            # [C]_{ij} is <du/dtheta i | du/dtheta j>
            C = np.around([[deriv_overlap(i, j1, i, j2) for j2 in range(num_circuits)] for j1 in range(num_circuits)], 14)     
            H_grad = np.around(H_gradient[i], 10)
                
            # Solve C d\theta = H_grad and obtain d\theta
            var_params_deriv[i] = np.real(np.linalg.inv(C) @ H_grad)

        return var_params_deriv.reshape(num_circuits * L)

    # Numerical integration of ODEs
    t_points = np.arange(0, t_total + step_size, step_size)

    sol = solve_ivp(get_var_params_deriv, [0, t_total], var_params_ini.reshape(num_circuits * L), method = 'RK23', first_step = 0.0001, t_eval=t_points, rtol = 0.3)
            
    # Update variational parameters for differen times
    var_params_all = [np.array([sol.y[l][t] for l in range(L*num_circuits)]).reshape(L, num_circuits) for t in range(len(sol.t))]

    # Update the variational circuits with new parameters
    var_circuits_all = [SU4_Circuits(L, var_params_all[t], N) for t in range(len(sol.t))]

    # Calculate the final energy
    H_fin = model_mpo.adiabatic_z_zz(L, 1)
    E_final, _ = mps.Energy_finite(var_circuits_all[-1].u_list(), var_circuits_all[-1].derivs_list(), L, H_fin, burnin=0)
 
    # return the variational parameters for plotting
    return var_params_all, var_circuits_all, np.real(E_final)


def evolution_adiabatic_mps(var_circuits, phys_sites, num_qubits, num_steps, time_step):
    # Each physical site parametrized by some number of circuits
    num_circuits = var_circuits.circ_len

    # Number of sites
    L = phys_sites
    N = num_qubits

    # Initialize parameter array
    var_params = np.zeros((num_steps+1, L, num_circuits))
    var_params[0] = var_circuits.get_params()

    # Collect matrix product of state at each time step
    var_circuits_all = [[] for n in range(num_steps+1)]
    var_circuits_all[0] = SU4_Circuits(L, var_params[0], N)

    # Time evolution
    for t in range(num_steps):
        # Calculate derivatives overlap 
        var_params_deriv = np.zeros((L, num_circuits))
        var_circuits = SU4_Circuits(L, var_params[t], N)
        
        def deriv_overlap(i1, j1, i2, j2):
            return mps.derivative_overlap_finite(var_circuits.u_list(), var_circuits.derivs_list(), L, [(i1, j1),(i2, j2)])
        
        # Calculate the Hamiltonian at that time step
        # Currently directly evolving with Heisenberg Hamiltonian
        # Change the last 1 to t/num_steps to do adiabatic evolution
        H_mps =model_mpo.adiabatic_x_zz(L, 0.5, 1, 1)
        # H_mps = model_mpo.z_chain(L)

        # Calculate energy gradient
        _, H_gradient = mps.Energy_finite(var_circuits.u_list(), var_circuits.derivs_list(), L, H_mps, burnin=0)

        # Put overlap terms and energy gradient terms in matrices
        for i in range(L):
            # For each site there is a matrix C
            # [C]_{ij} is <du/dtheta i | du/dtheta j>
            C = [[deriv_overlap(i, j1, i, j2) for j2 in range(num_circuits)] for j1 in range(num_circuits)]   
            H_grad = [H_gradient[i][j1] for j1 in range(num_circuits)]
                
            # Solve C d\theta = H_grad and obtain d\theta
            var_params_deriv[i] = np.linalg.pinv(C, rcond=1e-14) @ H_grad
        
        # Update the next row of variational parameters
        var_params[t+1] = var_params[t] + time_step * var_params_deriv

        # update the variational circuits with new parameters
        var_circuits_all[t+1] = SU4_Circuits(L, var_params[t+1], N)

    H_fin = model_mpo.adiabatic_x_zz(L, 0.5, 1, 1)
    # H_fin = model_mpo.z_chain(L)
    E_final, _ = mps.Energy_finite(var_circuits_all[-1].u_list(), var_circuits_all[-1].derivs_list(), L, H_fin, burnin=0)
 
    # return the variational parameters for plotting
    return var_params, var_circuits_all, E_final

