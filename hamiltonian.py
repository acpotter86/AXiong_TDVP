#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  30 09:20:10 2021
@contributors: Shahin, Yuxuan
"""

import numpy as np
from utilities_MPS import H_contract

class model_mpo(object): 
    def z_chain(N):
        # Pauli matrices 
        sigmax = np.array([[0., 1], [1, 0.]])
        sigmay = np.array([[0., -1j], [1j, 0.]])
        sigmaz = np.array([[1, 0.], [0., -1]])
        id = np.eye(2)

        # unit cell
        H = np.zeros((2, 2, 2, 2), dtype=np.complex_)
        H[0, 0] = H[1,1] = id
        H[1, 0] = -sigmaz
        H1 = np.einsum('abcd->cbda',H)

        H_bvecl = np.zeros(2)
        H_bvecr = np.zeros(2)
        H_bvecr[0] = 1.
        H_bvecl[-1] = 1.

        H_mps = H_contract(H1,N,H_bvecl,H_bvecr)
        H_mps = H_mps.reindex({f'p_out{i}':f'p_out{i}' for i in range(N)})
        H_mps = H_mps.reindex({f'pc_out{i}':f'pc_out{i}' for i in range(N)})
        return H_mps
    
    def zz_chain(N):
        # Pauli matrices 
        sigmax = np.array([[0., 1], [1, 0.]])
        sigmay = np.array([[0., -1j], [1j, 0.]])
        sigmaz = np.array([[1, 0.], [0., -1]])
        id = np.eye(2)
        
        # unit cell
        H = np.zeros((3, 3, 2, 2), dtype=np.complex_)
        H[0, 0] = H[2,2] = id
        H[1, 0] = sigmaz
        H[2, 1] = sigmaz
        H1 = np.einsum('abcd->cbda',H)
        return H1
    
    def x_z_adiabatic(N, r):
        # Pauli matrices 
        sigmax = np.array([[0., 1], [1, 0.]])
        sigmay = np.array([[0., -1j], [1j, 0.]])
        sigmaz = np.array([[1, 0.], [0., -1]])
        id = np.eye(2)
        
        # unit cell
        H = np.zeros((2, 2, 2, 2), dtype=np.complex_)
        H[0, 0] = H[1, 1] = id
        H[1, 0] = - ((1-r) * sigmax + r * sigmaz)
        H1 = np.einsum('abcd->cbda',H)

        H_bvecl = np.zeros(2)
        H_bvecr = np.zeros(2)
        H_bvecr[0] = 1.
        H_bvecl[-1] = 1.

        H_mps = H_contract(H1,N,H_bvecl,H_bvecr)
        H_mps = H_mps.reindex({f'p_out{i}':f'p_out{i}' for i in range(N)})
        H_mps = H_mps.reindex({f'pc_out{i}':f'pc_out{i}' for i in range(N)})
        return H_mps
    
    def adiabatic_x_zz(N, J, h, r):
        # Pauli matrices 
        sigmax = np.array([[0., 1], [1, 0.]])
        sigmay = np.array([[0., -1j], [1j, 0.]])
        sigmaz = np.array([[1, 0.], [0., -1]])
        id = np.eye(2)
        
        # unit cell
        H = np.zeros((3, 3, 2, 2), dtype=np.complex_)
        H[0, 0] = H[2,2] = id
        H[1, 0] = sigmaz
        H[2, 0] = - h * r * sigmax
        H[2, 1] = - J * sigmaz
        H1 = np.einsum('abcd->cbda',H)

        H_bvecl = np.zeros(3)
        H_bvecr = np.zeros(3)
        H_bvecr[0] = 1.
        H_bvecl[-1] = 1.

        H_mps = H_contract(H1,N,H_bvecl,H_bvecr)
        H_mps = H_mps.reindex({f'p_out{i}':f'p_out{i}' for i in range(N)})
        H_mps = H_mps.reindex({f'pc_out{i}':f'pc_out{i}' for i in range(N)})
        return H_mps


    """
    Matrix product operator (MPO) representation of
    model Hamiltonian (written in the thermal states
    convention).
    tensor leg index notation: p_out, b_out, p_in, b_in
    N = number of sites
    """ 
    def tfim_ising(J, g, h, N):
        """
        Unit-cell matrix product operator of Transverse 
        Field Ising model (TFIM). 
        """
        # Pauli matrices 
        sigmax = np.array([[0., 1], [1, 0.]])
        sigmay = np.array([[0., -1j], [1j, 0.]])
        sigmaz = np.array([[1, 0.], [0., -1]])
        id = np.eye(2)
    
       # structure of TFIM Ising model MPO unit cell
        H = np.zeros((3, 3, 2, 2), dtype=np.complex_)
        H[0, 0] = H[2, 2] = id
        H[1, 0] = sigmaz
        H[2, 0] = (-g * sigmaz - h * sigmax)/N
        H[2, 1] = (-J * sigmaz)/(N-1)
        H1 = np.einsum('abcd->cbda',H)
        return H1
    
    def xxz(J, Delta, hz, N):
        """
        Unit-cell matrix product operator of anisotropic 
        Heisenberg XXZ chain model
        """
        # Pauli matrices 
        sigmax = np.array([[0., 1], [1, 0.]])
        sigmay = np.array([[0., -1j], [1j, 0.]])
        sigmaz = np.array([[1, 0.], [0., -1]])
        id = np.eye(2)
    
        # structure of XXZ model MPO unit cell
        H = np.zeros((5, 5, 2, 2), dtype=np.complex_) 
        H[0, 0] = H[4, 4] = id
        H[1, 0] = sigmax
        H[2, 0] = sigmay
        H[3, 0] = sigmaz
        H[4, 0] = (hz * sigmaz)/N
        H[4, 1] = (J * sigmax)/(N - 1)
        H[4, 2] = (J * sigmay)/(N - 1)
        H[4, 3] = (J * Delta * sigmaz)/(N - 1)
        H1 = np.einsum('abcd->cbda',H)
        return H1
    
