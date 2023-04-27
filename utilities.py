import numpy as np
import functools as f

class Gate:
    """This class contains functions of common gates (expressed as matrices in z basis)
    """
    def __init__(self, matrix = np.array([[1,0],[0,1]])):
        self.matrix = matrix

    ## Pauli X, Y, Z gates
    def apply_op(self, Op):
        return Gate(Op @ self.matrix)

    def sx(self):
        return self.apply_op(np.array([[0,1], [1,0]]))

    def sy(self):
        return self.apply_op(np.array([[0,-1j], [1j,0]]))
      
    def sz(self):
        return self.apply_op(np.array([[1,0], [0,-1]]))

    def pauli(self, p):
        if p == "X":   
            return np.array([[0,1], [1,0]])
        elif p == "Y":   
            return np.array([[0,-1j], [1j,0]])
        elif p == "Z":   
            return np.array([[1,0], [0,-1]])
        elif p == "I":
            return np.array([[1,0],[0,1]])

    ## Hadamart, phase gate
    def hadamart(self):
        return self.apply_op(np.array([[1, 1],[1,-1]])/np.sqrt(2))

    def phase(self):
        return self.apply_op(np.array([[1, 0],[0, 1j]]))

    def phase(self, phi):
        temp = np.array([[np.exp(-1j * phi/2), 0],[0,np.exp(-1j * phi/2)]])
        return self.apply_op(temp)

    ## Rotations
    def rot_x(self, phi):
        return self.apply_op(np.cos(phi/2) * np.array([[1, 0],[0, 1]]) 
            - 1j * np.sin(phi/2) * np.array([[0, 1], [1, 0]]))

    def rot_y(self, phi):
        return self.apply_op(np.cos(phi/2) * np.array([[1, 0],[0, 1]]) 
            - 1j * np.sin(phi/2) * np.array([[0, -1j], [1j, 0]]))

    def rot_z(self, phi):
        return self.apply_op(np.cos(phi/2) * np.array([[1, 0],[0, 1]]) 
            - 1j * np.sin(phi/2) * np.array([[1, 0], [0, -1]]))

    def rot(self, p, phi):
        if p == "X":   
            return self.rot_x(phi)
        elif p == "Y":   
            return self.rot_y(phi)
        elif p == "Z":   
            return self.rot_z(phi)
        else:
            return self

    def to_mat(self):
        return self.matrix



class VarCircuit:
    """Represents e^{j phi_l P_l} for some Pauli gate P_l acting on sites idx1 and idx2
    """
    def __init__(self, var_param, chain_len, pauli1, idx1, pauli2="I", idx2=-1, rotation=True):
        """Initializes one layer of the variational circuit (e^{j phi_l P_l})
        Must multiply all layers of circuits together to obtain full state

        Args:
            var_param (float): variational parameter theta_l
            chain_len (int): total chain length 
            pauli1 (Gate): first Pauli operator
            idx1 (int): index where the first Pauli operator acts
            pauli2 (Gate, optional): _description_. Defaults to Gate().pauli("I").
            idx2 (int, optional): _description_. Defaults to -1.
        """
        G = Gate()
        I = G.pauli("I")

        # save variational parameter to be accessed
        self.param = var_param
        self.chain_len = chain_len
        self.pauli1 = pauli1
        self.pauli2 = pauli2
        self.idx1 = idx1
        self.idx2 = idx2
       
        # compute I ... P1 ... P2 ... I
        gates = [G.pauli(pauli1) if (i==idx1) else G.pauli(pauli2) if (i==idx2) else I for i in range(chain_len)]
        self.gates_prod = f.reduce(lambda a, b: np.kron(a,b), gates)
        
        # compute identity tensor I ..... I
        identity = [I for i in range(chain_len)]
        self.identity = f.reduce(lambda a, b: np.kron(a,b), identity)

        # compute e^(-i theta/2 I ... P1 ... P2 ... I)
        if rotation == True:
            self.matrix_ket = np.cos(var_param/2) * self.identity - 1j * np.sin(var_param/2) * self.gates_prod
        else:
            self.matrix_ket = self.gates_prod
            
        self.matrix_bra = np.conj(self.matrix_ket).T

    def get_param(self):
        return self.param

    def get_chain_len(self):
        return self.chain_len

    def update_param(self, new_var_param):
        return VarCircuit(new_var_param, self.chain_len, self.pauli1, self.idx1, self.pauli2, self.idx2)

    def deriv_ket(self):        
        # the derivative of this exponential wrt theta is just the tensor product of gates * -i/2
        return -1j/2 * self.gates_prod @ self.matrix_ket

    def deriv_bra(self):
        return 1j/2 * self.matrix_bra @ (np.conj(self.gates_prod).T)

    def bra(self):
        return self.matrix_bra

    def ket(self):
        return self.matrix_ket


class SU4_Circuits:
    def __init__(self, L, params_init, N, type = 'mps'):
        # L is number of sites, N is number of qubits
        self.L = L

        # params_init is an array of L by 15 values, ordered as following
        # Initialize circuit structure
        # if type=='mps':
        #     self.var_circuits = [[VarCircuit(params_init[i][0], N, "X", 0),     
        #                 VarCircuit(params_init[i][1], N, "Z", 0),  
        #                 VarCircuit(params_init[i][2], N, "X", 0),
        #                 VarCircuit(params_init[i][3], N, "X", 1),
        #                 VarCircuit(params_init[i][4], N, "Z", 1),  
        #                 VarCircuit(params_init[i][5], N, "X", 1),
        #                 VarCircuit(params_init[i][6], N, "Z", 0, "Z", 1), 
        #                 VarCircuit(params_init[i][7], N, "Y", 0, "Y", 1), 
        #                 VarCircuit(params_init[i][8], N, "X", 0, "X", 1),   
        #                 VarCircuit(params_init[i][9], N, "X", 0),     
        #                 VarCircuit(params_init[i][10], N, "Z", 0),  
        #                 VarCircuit(params_init[i][11], N, "X", 0),
        #                 VarCircuit(params_init[i][12], N, "X", 1),
        #                 VarCircuit(params_init[i][13], N, "Z", 1),  
        #                 VarCircuit(params_init[i][14], N, "X", 1)] for i in range(L)]
        # else: 
        #     self.var_circuits = [[VarCircuit(params_init[i][0], N, "X", N-L+i),
        #                 VarCircuit(params_init[i][1], N, "Z", N-L+i),  
        #                 VarCircuit(params_init[i][2], N, "X", N-L+i),
        #                 VarCircuit(params_init[i][3], N, "X", 0),     
        #                 VarCircuit(params_init[i][4], N, "Z", 0),  
        #                 VarCircuit(params_init[i][5], N, "X", 0),
        #                 VarCircuit(params_init[i][6], N, "Z", 0, "Z", N-L+i), 
        #                 VarCircuit(params_init[i][7], N, "Y", 0, "Y", N-L+i), 
        #                 VarCircuit(params_init[i][8], N, "X", 0, "X", N-L+i),   
        #                 VarCircuit(params_init[i][9], N, "X", N-L+i),
        #                 VarCircuit(params_init[i][10], N, "Z", N-L+i),  
        #                 VarCircuit(params_init[i][11], N, "X", N-L+i),
        #                 VarCircuit(params_init[i][12], N, "X", 0),     
        #                 VarCircuit(params_init[i][13], N, "Z", 0),  
        #                 VarCircuit(params_init[i][14], N, "X", 0)] for i in range(L)]

        if type=='mps':
            self.var_circuits = [[VarCircuit(params_init[i][0], N, "X", 0),  
                            VarCircuit(params_init[i][1], N, "Z", 0),
                            VarCircuit(params_init[i][2], N, "X", 0),  
                            VarCircuit(params_init[i][3], N, "X", 1),
                            VarCircuit(params_init[i][4], N, "Z", 1),
                            VarCircuit(params_init[i][5], N, "X", 1), 
                            VarCircuit(params_init[i][6], N, "X", 0, "X", 1),
                            VarCircuit(params_init[i][7], N, "Y", 0, "Y", 1),
                            VarCircuit(params_init[i][8], N, "Z", 0, "Z", 1),
                            VarCircuit(params_init[i][9], N, "X", 0),  
                            VarCircuit(params_init[i][10], N, "Z", 0),
                            VarCircuit(params_init[i][11], N, "X", 0)] for i in range(L)]
        else:
            self.var_circuits = [[VarCircuit(params_init[i][0], N, "X",N-L+i),
                            VarCircuit(params_init[i][1], N, "Z", N-L+i),  
                            VarCircuit(params_init[i][2], N, "X", N-L+i),
                            VarCircuit(params_init[i][3], N, "X", 0),  
                            VarCircuit(params_init[i][4], N, "Z", 0),
                            VarCircuit(params_init[i][5], N, "X", 0), 
                            VarCircuit(params_init[i][6], N, "X", 0, "X", N-L+i),
                            VarCircuit(params_init[i][7], N, "Y", 0, "Y", N-L+i),
                            VarCircuit(params_init[i][8], N, "Z", 0, "Z", N-L+i),
                            VarCircuit(params_init[i][9], N, "X", N-L+i),
                            VarCircuit(params_init[i][10], N, "Z", N-L+i),  
                            VarCircuit(params_init[i][11], N, "X", N-L+i)] for i in range(L)]
        
        self.var_params = params_init
        
        self.circ_len = len(self.var_circuits[0])
        
    def get_params(self):
        return self.var_params

    def update_params(self, params_list):
        # update the whole list of variational parameters
        self.var_params = params_list
        for i in range(self.L):
            for j in range(self.circ_len):
                self.var_circuits[i][j].update_param(params_list[i][j])
        return self

    def u_list(self):
        # list of u with all circuit matrices separated
        u_list_circuits = [[self.var_circuits[i][j].ket() for j in range(self.circ_len-1, -1, -1)] for i in range(self.L)]
        # list of u with all circuit matrices multiplied together
        u_list = [f.reduce(lambda a, b: a @ b, u_list_circuits[i]) for i in range(self.L)]
        return u_list
    
    def derivs_list(self):
        # list of derivatives with all circuit matrices separated
        derivs_circuits = [[[self.var_circuits[l][i].deriv_ket() if i==j else self.var_circuits[l][i].ket() for i in range(self.circ_len-1, -1, -1)] for j in range(self.circ_len)] for l in range(self.L)]
        derivs_list = [[f.reduce(lambda a, b: a @ b, derivs_circuits[l][i]) for i in range(self.circ_len)] for l in range(self.L)]
        return derivs_list

    def get_circuits(self):
        return self.var_circuits