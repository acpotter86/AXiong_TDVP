# standard import
import numpy as np
import time as time
from random import*
import quimb.tensor as qtn
np.set_printoptions(precision=6,threshold=1e-6)


p0 = [[ 1.,  0.,],[ 0,  1]]
p1 = [[ 0.,  1.,],[ 1,  0]]
p2 = [[ 0.,  -1j],[ 1j, 0]]
p3 = [[ 1.,  0.,],[ 0, -1]]
paulis = [p0,p1,p2,p3]

def H(Hamiltonian,n,j):
    inds4 = f'p_out{j}',f'H{n+1}',f'pc_out{j}',f'H{n}'
    #if n>2:#burinin

    # H shape (2, 3, 2, 3) for Ising model
    H = qtn.Tensor(Hamiltonian,inds4,tags='H')
    #else:
        #H = qtn.Tensor(Ham,inds4,tags='H')
    return H


def H_contract(Hamiltonian,N,H_bvecl,H_bvecr):
    """
    Returns tensor contractions of Hamiltonian
    """
    TN_list = [H(Hamiltonian,n,j) for n,j in zip(range(N),range(N))]
    
    # for Hamiltonian
    # Left and right boundary conditions
    inds4 = 'H0',
    H_left = qtn.Tensor(H_bvecl,inds4,tags='Hl')
    inds5 = f'H{N}',
    H_right = qtn.Tensor(H_bvecr,inds5,tags='Hr')

    # tensor contractions
    TN0 = TN_list[0]
    for j in range(1,len(TN_list)):
        TN0 = TN_list[j] & TN0

    TN = H_left & H_right & TN0
    return TN

def expectation(tensor_list, basis, bdry_vec, burnin = 0, MPDO = None, input_type = 'unitary'):
    '''
    Construct a tn with package quimb:
    input a list of isometries or unitaries, return a tensor network that evaluates a desired observable
    output a network that over some measurement basis (array) - convention: 0/1/2/3 : IXYZ e. g.:[0,0,0,3] => IIIZ
    if a MPDO is provided as an observable, then the expectation will be evaluated with the provided MPDO instead
    notice that this only work for a depth 1 MPS; deeper circuits will require modifications 
    '''
    L = len(basis) # number of sites
    L_c = len(tensor_list) # unit cell length; in the finite MPS case this is L

    inds_d ='b_in', 'bc_in', 
    bdry_l = qtn.Tensor(bdry_vec,inds_d,tags='lbdry')
    if input_type == 'unitary':
        inds_d ='p_in0','b_in','p_out0','b_out0'
        t = qtn.Tensor(tensor_list[0],inds_d,tags='site0')
        inds_d ='pc_in0','bc_in','pc_out0','bc_out0'
        tc = qtn.Tensor(np.conj(tensor_list[0]),inds_d,tags='site0')

        tn = t&tc
        if L>1:
            for i in range (1,L):
                inds_d =f'p_in{i}',f'b_out{i-1}',f'p_out{i}',f'b_out{i}'
                t = qtn.Tensor(tensor_list[i%L_c],inds_d,tags=f'site{i}')                
                inds_d =f'pc_in{i}',f'bc_out{i-1}',f'pc_out{i}',f'bc_out{i}'
                tc = qtn.Tensor(np.conj(tensor_list[i%L_c]),inds_d,tags=f'site{i}')
                tn = tn&t&tc
        #density matrices; you can change this part so that we represent a thermal state!
        for i in range (L):
            inds_d =f'p_in{i}',  f'pc_in{i}'
            t = qtn.Tensor([[1,0],[0,0]],inds_d,tags=f'den{i}')   
            tn = tn&t

    #measurements
    if MPDO == None: # if no MPDO provided, measure on the basis provided
        for i in range (L):
            inds_d = f'p_out{i}',f'pc_out{i}',
            t = qtn.Tensor(paulis[basis[i]],inds_d,tags=f'site{i}')                
            tn = tn&t
    else:
        tn = tn&MPDO
    if burnin != 0 :
        for i in range (0,burnin):
            tn = tn.reindex({f'pc_out{i}':f'p_out{i}'})
    tn = tn.reindex({f'bc_out{L-1}':f'b_out{L-1}'})
    return tn&bdry_l 



def Energy_finite(u_list, d_list, L, MPDO, burnin=0):
    """
    computes VQE energy with a given MPDO and unitary derivatives, for finite MPS only
    since for the finite version, derivatives can be computed analytically with chain rule
    u_list: a list of unitaries
    d_list: a list of derivatives format - [[dU_1/dtheta_11,dU_1/dtheta_12...],[dU_2/dtheta_1,dU_21/dtheta_22...],...]
    for each site we asume a different unitary
    burnin is always set to 0 for now
    """
    basis = '0'*L
    N_para = len(d_list[0]) # number of parameters in each circuit
    b_dim = len(u_list[0])//2 # bond dimension
    tensor_list = [np.conj(U).T.reshape(2,b_dim,2,b_dim) for U in u_list]
    
    # define boundary condition
    bdry_vec = np.zeros([b_dim,b_dim])
    bdry_vec[0][0] = 1
    
    # define the whole tensor network
    E_tn = expectation(tensor_list, basis, bdry_vec, MPDO = MPDO, input_type = 'unitary', burnin=burnin)

    # check if the imaginary term is important
    E = E_tn^all
    dE = np.zeros((L,N_para))
    
    # assume L sites
    for l in range(L):
        for t in range(N_para): #loop over parameters, assuming each unitary is specified by same amount of parameters
            U_dr = np.conj(d_list[l][t]).T.reshape(2,b_dim,2,b_dim)
            
            # change each tensor in the tn from U to dU
            E_dr = E_tn.copy() # make a new copy
            E_dr.tensors[2*l].modify(data = U_dr)
            dr = 2*np.imag(E_dr^all) #factor of two b/c self/dual space
            dE[l][t] = dr
    return E,dE 

def derivative_overlap_finite(u_list, d_list, L, labels, MPDO = None, burnin=0):
    """
    computes wave function derivative overlaps, for finite MPS only
    <d psi/ d theta_i2j2 | d psi/ d theta_i1j1>
    for the finite version, derivatives can be computed analytically with chain rule
    u_list: a list of unitaries, each is an SU4 matrix (i.e. multiply all the gates at one site together)
    d_list: format - [[dU_1/dtheta_11,dU_1/dtheta_12...],[dU_2/dtheta_1,dU_21/dtheta_22...],...]
    labels: the labels of the desired parameters whose derivative overlap we calculate
    format - [(i1,j1),(i2,j2)]
    for each site we asume a different unitary
    burnin is always set to 0 for now
    """
    basis = [0]*L
    b_dim = len(u_list[0])//2 #bond dimension; // is division with floor function
    tensor_list = [np.conj(U.T).reshape(2,b_dim,2,b_dim) for U in u_list]
    
    # define boundary condition
    bdry_vec = np.zeros([b_dim,b_dim])
    bdry_vec[0][0] = 1
    
    # define the whole tensor network
    E_tn = expectation(tensor_list, basis, bdry_vec, MPDO = MPDO, input_type = 'unitary',burnin=burnin)
    [(i1,j1),(i2,j2)] = labels
    
    U1_dr = np.conj(d_list[i1][j1].T).reshape(2,b_dim,2,b_dim)
    U2_dr = np.conj(d_list[i2][j2].T).reshape(2,b_dim,2,b_dim)
    
    # change each tensor in the tn from U to dU
    E_tn.tensors[i1*2].modify(data = U1_dr)
    E_tn.tensors[i2*2+1].modify(data = np.conj(U2_dr)) # conjugate since it's in dual space

    dr = 2*np.real(E_tn^all) #factor of two b/c self/dual space
    return dr

