import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
from numpy.lib import scimath as sm  # sqrt of negative number is complex
import pickle
import math, os
import cmath
from packaging import version
import matplotlib.ticker as ticker

def print_complex(name, number):
    string_to_print = "{} ({},{})".format(name, np.real(number), np.imag(number))
    return string_to_print

def dev_complex(number):
    string_to_print = "({},{})".format(np.real(number), np.imag(number))
    return string_to_print

def dev_complex22(number):
    num00 = number[0, 0]
    num01 = number[0, 1]
    num10 = number[1, 0]
    num11 = number[1, 1]
    string_to_print = f"{dev_complex(num00)} {dev_complex(num01)} {dev_complex(num10)} {dev_complex(num11)}"
    return string_to_print

def dev_complex2l(number):
    num00 = number[0][0]
    num01 = number[1][0]
    string_to_print = f"{dev_complex(num00)} {dev_complex(num01)}"
    return string_to_print

def is_number_in_ranges(ranges_list, number):
    """
    Checks if a number is within any of the ranges in a list of ranges.
    
    :param ranges_list: List of ranges, where each range is represented by a list [start, end].
    :param number: The number to check.
    :return: True if the number is in any of the ranges, False otherwise.
    """
    # If ranges_list is a single range, convert it to a list containing one range
    if isinstance(ranges_list[0], int):
        ranges_list = [ranges_list]
    
    for start, end in ranges_list:
        if start <= number <= end:
            return True
    return False

def np_multiply(X, Y):
    output = np.empty((2, 2), dtype=np.complex128)

    output[0][0] = X[0][0]*Y[0][0] + X[0][1]*Y[1][0]
    output[0][1] = X[0][0]*Y[0][1] + X[0][1]*Y[1][1]
    output[1][0] = X[1][0]*Y[0][0] + X[1][1]*Y[1][0]
    output[1][1] = X[1][0]*Y[0][1] + X[1][1]*Y[1][1]
        
    return output

def np_multiply_test(X, Y):
    # output = np.empty((2, 2), dtype=np.complex128)
    output = np.empty((2, 2))

    output[0][0] = X[0][0]*Y[0][0] + X[0][1]*Y[1][0]
    output[0][1] = X[0][0]*Y[0][1] + X[0][1]*Y[1][1]
    output[1][0] = X[1][0]*Y[0][0] + X[1][1]*Y[1][0]
    output[1][1] = X[1][0]*Y[0][1] + X[1][1]*Y[1][1]
        
    return output

def set_structure_values(values):
    if len(values) == 3:
        w_l = 5
        qw_l = round(values[0]/10, 1)
        qb_l = round(values[1]/10, 1)

        defe = round(values[2]/10, 1)

        w_r = 1
        qw_r = round(values[0]/10, 1)
        qb_r = round(values[1]/10, 1)

        structure = [w_l,qw_l,qb_l, defe, w_r,qw_r,qb_r]

    elif len(values) == 5:
        w_l = round(values[0], 0)
        qw_l = round(values[1]/10, 1)
        qb_l = round(values[2]/10, 1)

        defe = round(values[3]/10, 1)

        w_r = round(values[4], 0)
        qw_r = round(values[1]/10, 1)
        qb_r = round(values[2]/10, 1)

        structure = [w_l,qw_l,qb_l, defe, w_r,qw_r,qb_r]

    elif len(values) == 7:
        w_l = round(values[0], 0)
        qw_l = round(values[1]/10, 1)
        qb_l = round(values[2]/10, 1)

        defe = round(values[3]/10, 1)

        w_r = round(values[4], 0)
        qw_r = round(values[5]/10, 1)
        qb_r = round(values[6]/10, 1)

        structure = [w_l,qw_l,qb_l, defe, w_r,qw_r,qb_r]

    # modifica os valores do pocos e numero de pocos se um valor eh 0
    if 0 in [w_l, qw_l, qb_l]:
        w_l, qw_l, qb_l = 0, 0, 0
        structure = [w_l,qw_l,qb_l, defe, w_r,qw_r,qb_r]
    if 0 in [w_r, qw_r, qb_r]:
        w_r, qw_r, qb_r = 0, 0, 0
        structure = [w_l,qw_l,qb_l, defe, w_r,qw_r,qb_r]

    if w_r > w_l:
        structure = [w_r,qw_r,qb_r, defe, w_l,qw_l,qb_l]
    
    return structure

def set_id(structure):
    lr = "{:02.0f}".format(structure[0])
    lw = "{:05.2f}".format(structure[1])
    lb = "{:05.2f}".format(structure[2])
    
    de = "{:04.1f}".format(structure[3])

    rr = "{:02.0f}".format(structure[4])
    rw = "{:05.2f}".format(structure[5])
    rb = "{:05.2f}".format(structure[6])

    file_id   = lr+"x"+lw+"_"+lb+"__"+de+"__"+rr+"x"+rw+"_"+rb
    return file_id

def set_id_from_ga(structure):
    lr = "{:02.0f}".format(structure[0])
    lw = "{:05.2f}".format(structure[1]/10)
    lb = "{:05.2f}".format(structure[2]/10)
    
    de = "{:04.1f}".format(structure[3]/10)

    rr = "{:02.0f}".format(structure[4])
    rw = "{:05.2f}".format(structure[5]/10)
    rb = "{:05.2f}".format(structure[6]/10)

    file_id   = lr+"x"+lw+"_"+lb+"__"+de+"__"+rr+"x"+rw+"_"+rb
    return file_id


class QBMD_simulator():
    def __init__(self, structure, simulation_type = "test",
                pwf={"pwf_E_initial" : 0,
                        "pwf_E_final" : 800,
                        "pwf_dE" : 0.1,
                        "nn": 100},
                ppc={"ppc_E_initial" : 0,
                        "ppc_E_final" : 800,
                        "ppc_dE" : 0.1}, 
                output_folder="C:/PD/sims/"):
        
        self.structure = structure
        self.simulation_type = simulation_type
        # si for develop, definimos os arquivos para o develop
        if self.simulation_type == "develop":
            self.dev_mode = True
            self.dev = True
            
            
            # listas para incorporar os logs
            
            # START: para print_301_InicializarValores
            self.path_301_InicializarValores = "develop/temp_files_simulations/python/print_301_InicializarValores.txt"
            with open(self.path_301_InicializarValores, 'w') as file:
                pass
            
            # STRUCT: para print_302_Structure
            self.path_302_Structure = "develop/temp_files_simulations/python/print_302_Structure.txt"
            with open(self.path_302_Structure, 'w') as file:
                pass
            
            # STRUCT: para print_303_Potencial
            self.path_303_Potencial = "develop/temp_files_simulations/python/print_303_Potencial.txt"
            with open(self.path_303_Potencial, 'w') as file:
                pass
            
            # ETM_E: para print_304_ETM_Energies
            self.path_304_Energies = "develop/temp_files_simulations/python/print_304_Energies.txt"
            with open(self.path_304_Energies, 'w') as file:
                pass
            
            # TM: para print_305_TM
            self.path_305_TM = "develop/temp_files_simulations/python/print_305_transfer_matrix.txt"
            with open(self.path_305_TM, 'w') as file:
                pass
            
            self.path_306_ETM_wf = "develop/temp_files_simulations/python/print_306_Eigenstates_Transfer_Matrix_wf.txt"
            with open(self.path_306_ETM_wf, 'w') as file:
                pass
            
            self.path_307_OS = "develop/temp_files_simulations/python/print_307_oscillator_strength.txt"
            with open(self.path_307_OS, 'w') as file:
                pass
            
            self.path_308_PC = "develop/temp_files_simulations/python/print_308_pc.txt"
            with open(self.path_308_PC, 'w') as file:
                pass
            
            self.path_316_wf_tm = "develop/temp_files_simulations/python/print_316_wavefunction_transfer_matrix.txt"
            with open(self.path_316_wf_tm, 'w') as file:
                pass
            
            # para print_201_Eigenstates_Transfer_Matrix
            with open("develop/temp_files_simulations/python/print_201_Eigenstates_Transfer_Matrix.txt", 'w') as file:
                pass
            
        else:
            self.dev_mode = False
            self.dev = False
        
        # salvaoms os parametros de entrada
        self.output_folder = output_folder
        
        # ppc: parameters photocurrent
        self.ppc_E_initial = ppc["ppc_E_initial"]
        self.ppc_E_final = ppc["ppc_E_final"]
        self.ppc_dE = ppc["ppc_dE"]
        self.ppc_dE = ppc["ppc_dE"]

        # pwf: parameters wavefunction
        self.pwf_E_initial = pwf["pwf_E_initial"]       #(meV) 
        self.pwf_E_final = pwf["pwf_E_final"]     #(meV) 
        self.pwf_dE = pwf["pwf_dE"]     #(meV)
        self.nn = pwf["nn"]     ## The number of states you want to calculate
        self.pwf_En = (self.pwf_E_final - self.pwf_E_initial)/self.pwf_dE
        self.pwf_E = self.pwf_E_initial

        # geramos os nomes da amostra e pastas para salvar os arquivos
        self.process_names()

        # definimos as constantes
        self.pi = math.pi
        self.me = 9.10938356E-31 #[kg]
        self.hbar = 1.054571817E-34 #[J][s]
        self.e_charge = 1.602176634E-19 #[C]

        self.eo_InGaAs = 14.3 #constante dieletrica InGaAs
        self.eo_InAlAs = 12.7 #constante dieletrica InAlAs
        self.m_InGaAs = 0.0436*self.me 
        self.m_InAlAs = 0.0836*self.me 
        self.nm = 1E-9 #[m]
        self.Vo=0.503*self.e_charge #[J]
        self.barreira = 0.503						
        #fonte:DOI: 10.1103/PhysRevB.67.085318	
        self.Nonparabw = 1.3E-18
        self.Ew = self.hbar*self.hbar/(2*self.m_InGaAs*self.Nonparabw)    # gap energy of InGaAs in J#
        self.Eb = self.Ew*self.m_InAlAs/self.m_InGaAs                # gap energy of InAlAs in J #

        self.a0_InGaAs = 0.53*self.eo_InGaAs/self.m_InGaAs 
        self.a0_InAlAs = 0.53*self.eo_InAlAs/self.m_InAlAs 

        self.tol = 1.0e-4 # tolerancia
        # self.tol = self.e_charge * 1E-3 * 1.0E-12
        
        # self.tol = self.e_charge * 1E-3 * 1.0E-10
        self.max_interactions = 1000

        self.ic = 0.0 + 1.0j

        self.m_qw = self.m_InGaAs
        self.m_qb = self.m_InAlAs
        self.Fd = 1*1E5*self.e_charge # Amplitude of the external electric field [J]/[M]
        
        self.dx = round(0.1*self.nm,11)   ## step size dx
        self.beta = self.dx*self.dx/12
        self.hbar2 = self.hbar*self.hbar
        
        if self.dev:
            self.log_START(f"### Constantes Python ###")
            self.log_START(f"pi: {self.pi}")
            self.log_START(f"me: {self.me}")
            self.log_START(f"hbar: {self.hbar}")
            self.log_START(f"e_charge: {self.e_charge}")
            self.log_START(f"eo_InGaAs: {self.eo_InGaAs}")
            self.log_START(f"eo_InAlAs: {self.eo_InAlAs}")
            self.log_START(f"m_InGaAs: {self.m_InGaAs}")
            self.log_START(f"m_InAlAs: {self.m_InAlAs}")
            self.log_START(f"nm: {self.nm}")
            self.log_START(f"Vo: {self.Vo}")
            self.log_START(f"barreira: {self.barreira}")
            self.log_START(f"Nonparabw: {self.Nonparabw}")
            self.log_START(f"Ew: {self.Ew}")
            self.log_START(f"Eb: {self.Eb}")
            self.log_START(f"a0_InGaAs: {self.a0_InGaAs}")
            self.log_START(f"a0_InAlAs: {self.a0_InAlAs}")
            self.log_START(f"tol: {self.tol}")
            self.log_START(f"tol: {self.tol}")
            self.log_START(f"max_interactions: {self.max_interactions}")
            self.log_START(f"ic: {dev_complex(self.ic)}")
            self.log_START(f"m_qw: {self.m_qw}")
            self.log_START(f"m_qb: {self.m_qb}")
            self.log_START(f"Fd: {self.Fd}")

        ## ------------------------------------------------------------------
        ##                 dimension of the heterostructure [m]
        ## ------------------------------------------------------------------
        ## quantum bragg mirror to left of central quantum well
        self.num_qb_left  = self.structure[0]		##Number of well/barrier of the Bragg mirror
        self.L_qw_left    = self.structure[1] 	    ##Width of lateral quantum well (nm)
        self.L_qb_left    = self.structure[2] 	    ##Width of quantum barrier (nm)

        ## central quantum well
        self.L_cqw        = self.structure[3] 	##Width of central quantum well (nm)

        ## quantum bragg mirror to right of central quantum well
        self.num_qb_right = self.structure[4] 		##Number of well/barrier of the Bragg mirror
        self.L_qw_right   = self.structure[5] 	##Width of lateral quantum well (nm)
        self.L_qb_right   = self.structure[6] 	##Width of quantum barrier (nm)

        
        ##------------------------------------------------------------------
        ## Transformando as informacoes da super-rede(numero de pocos/barreiras e espessuras) em caracteres para criar a pasta da simulacao  

        

        return None
    
    def log(self, log_id=301, text='', n=0, ranges=[-1,1]):
        if log_id == 305:
            log_path = self.path_305_TM
        elif log_id == 306:
            log_path = self.path_306_ETM_wf
        elif log_id == 307:
            log_path = self.path_307_OS
        elif log_id == 308:
            log_path = self.path_308_PC
        elif log_id == 316:
            log_path = self.path_316_wf_tm
        else:
            return
        if is_number_in_ranges(ranges, n):
            with open(log_path, 'a') as file:
                file.write(text + "\n")
            
    def log_matrix22(self, log_id, matrix, matrix_name, n=0, ranges=[-1,1]):
        if log_id == 305:
            log_path = self.path_305_TM
        elif log_id == 306:
            log_path = self.path_306_ETM_wf
        elif log_id == 307:
            log_path = self.path_307_OS
        elif log_id == 308:
            log_path = self.path_308_PC
        elif log_id == 316:
            log_path = self.path_316_wf_tm
        else:
            return
        if is_number_in_ranges(ranges, n):
            with open(log_path, 'a') as file:
                self.log(log_id, f"* {matrix_name} [0,0]: {dev_complex(matrix[0,0])}")
                self.log(log_id, f"*** {matrix_name} [0,1]: {dev_complex(matrix[0,1])}")
                self.log(log_id, f"*** {matrix_name} [1,0]: {dev_complex(matrix[1,0])}")
                self.log(log_id, f"*** {matrix_name} [1,1]: {dev_complex(matrix[1,1])}")
                
    def log_START(self, text='', n=0, ranges=[-1,1]):
        with open(self.path_301_InicializarValores, 'a') as file:
            file.write(text + "\n")
    
    def log_STRUCT(self, text='', n=0, ranges=[-1,1]):
        with open(self.path_302_Structure, 'a') as file:
            file.write(text + "\n")
    
    def log_POT(self, text='', n=0, ranges=[-1,1]):
        with open(self.path_303_Potencial, 'a') as file:
            file.write(text + "\n")
            
    def log_ETM_E(self, text='', n=0, ranges=[-1,1]):
        with open(self.path_304_Energies, 'a') as file:
            file.write(text + "\n")
    
    def log_ETM_E_matrix22(self, matrix, matrix_name, n=0, ranges=[-1,1]):
        if is_number_in_ranges(ranges, n):
            self.log_ETM_E(f"* {matrix_name} [0,0]: {dev_complex(matrix[0,0])}")
            self.log_ETM_E(f"*** {matrix_name} [0,1]: {dev_complex(matrix[0,1])}")
            self.log_ETM_E(f"*** {matrix_name} [1,0]: {dev_complex(matrix[1,0])}")
            self.log_ETM_E(f"*** {matrix_name} [1,1]: {dev_complex(matrix[1,1])}")
    
    def log_ETM(self, text='', n=0, ranges=[-1,1]):
        if is_number_in_ranges(ranges, n):
            with open("develop/temp_files_simulations/python/print_201_Eigenstates_Transfer_Matrix.txt", 'a') as file:
                file.write(text + "\n")
        
    def log_ETM_matrix22(self, matrix, matrix_name, n=0, ranges=[-1,1]):
        if is_number_in_ranges(ranges, n):
            with open("develop/temp_files_simulations/python/print_201_Eigenstates_Transfer_Matrix.txt", 'a') as file:
                self.log_ETM(f"* {matrix_name} [0,0]: {dev_complex(matrix[0,0])}")
                self.log_ETM(f"*** {matrix_name} [0,1]: {dev_complex(matrix[0,1])}")
                self.log_ETM(f"*** {matrix_name} [1,0]: {dev_complex(matrix[1,0])}")
                self.log_ETM(f"*** {matrix_name} [1,1]: {dev_complex(matrix[1,1])}")
    
    def process_names(self):
        self.left_rep = self.structure[0]
        self.left_wells = self.structure[1]
        self.left_barriers = self.structure[2]
        
        self.defect = self.structure[3]
        
        self.right_rep = self.structure[4]
        self.right_wells = self.structure[5]
        self.right_barriers = self.structure[6]

        lr = "{:02.0f}".format(self.structure[0])
        lw = "{:05.2f}".format(self.structure[1])
        lb = "{:05.2f}".format(self.structure[2])
        
        de = "{:04.1f}".format(self.structure[3])

        rr = "{:02.0f}".format(self.structure[4])
        rw = "{:05.2f}".format(self.structure[5])
        rb = "{:05.2f}".format(self.structure[6])

        self.file_id   = lr+"x"+lw+"_"+lb+"__"+de+"__"+rr+"x"+rw+"_"+rb
        self.title_img = lr+"x"+lw+"_"+lb+"  "+de+"  "+rr+"x"+rw+"_"+rb
        self.pkl_path = self.output_folder + "sims/" + self.file_id + ".pkl"
        self.png_path = self.output_folder + "images/" + self.file_id + ".png"
        self.exist_pkl = os.path.exists(self.pkl_path)
        
        os.makedirs(self.output_folder + "images/",exist_ok=True)
        # os.makedirs(self.output_folder + "gifs/",exist_ok=True)
        # os.makedirs(self.output_folder + "gifs/img/",exist_ok=True)
        os.makedirs(self.output_folder + "sims/",exist_ok=True)

    def calculate_structure_limits(self):
        #--------------------------------------------------------------------
        # quantum bragg mirror to left of central quantum well
        self.L_qw_left    = round(self.L_qw_left*self.nm,11) 	#Width of lateral quantum well
        self.L_qb_left    = round(self.L_qb_left*self.nm,11) 	#Width of quantum barrier
        self.L_bm_left    = round(-self.num_qb_left*(self.L_qb_left+self.L_qw_left),11) #The thickness of left superlattice
        # self.L_bm_left    = round(self.num_qb_left*(self.L_qb_left+self.L_qw_left),11) #The thickness of left superlattice

        # quantum bragg mirror to right of central quantum well
        self.L_qb_right   = round(self.L_qb_right*self.nm,11) 	#Width of quantum barrier
        self.L_qw_right   = round(self.L_qw_right*self.nm,11) 	#Width of lateral quantum well
        self.L_bm_right   = round(self.num_qb_right*(self.L_qb_right+self.L_qw_right),11)

        # central quantum well
        self.L_cqw        = round(self.L_cqw*self.nm,11) 	#Width of central quantum well
        
        if self.dev:
            self.log_STRUCT(f"### Estrutura Python ###")
            self.log_STRUCT(f"L_qw_left: {self.L_qw_left}")
            self.log_STRUCT(f"L_qb_left: {self.L_qb_left}")
            self.log_STRUCT(f"L_bm_left: {self.L_bm_left}")

            self.log_STRUCT(f"L_qb_right: {self.L_qb_right}")
            self.log_STRUCT(f"L_qw_right: {self.L_qw_right}")
            self.log_STRUCT(f"L_bm_right: {self.L_bm_right}")

            self.log_STRUCT(f"L_cqw: {self.L_cqw}")
            
        
        
        
        
        #This "if" is used to find the largest superlattice (right or left side). The largest superlattice will have a barrier thickness of 50 nm, 
        #and the barrier thickness of the other superlattice will be calculated.
        self.strcuture_barrier = round(50*self.nm,11)
        # print("left: {}  || right: {}".format(round(self.L_bm_left/self.nm, 2), round(self.L_bm_right/self.nm,2)))
        if (-self.L_bm_left >= self.L_bm_right):
            if self.dev: self.log_STRUCT(f"\n-L_bm_left >= L_bm_right")
            self.L0_left      = -self.strcuture_barrier		#Thickness of the first barrier 
            self.L_left       = (self.L_bm_left + self.L0_left)
            self.L0_right     = -self.L_left - self.L_cqw - self.L_bm_right  #Thickness of the last barrier
            self.L_right      = self.L_cqw + self.L_bm_right + self.L0_right + self.L_cqw

        else:
            if self.dev: self.log_STRUCT(f"\nelse -L_bm_left >= L_bm_right")
            self.L0_right     = self.strcuture_barrier
            self.L_right	  = self.L_cqw + self.L_bm_right + self.L0_right  + self.L_cqw
            self.L0_left      = -self.L_right - self.L_bm_left
            self.L_left       = self.L_bm_left + self.L0_left
        
        self.n = int((self.L_right-self.L_left)/self.dx) + 2
        if self.dev:
            self.log_STRUCT(f"depois de identificar the largest superlattice")
            self.log_STRUCT(f"L0_left: {self.L0_left}")
            self.log_STRUCT(f"L_left: {self.L_left}")
            self.log_STRUCT(f"L0_right: {self.L0_right}")
            self.log_STRUCT(f"L_right: {self.L_right}")
            self.log_STRUCT(f"dx: {self.dx}")
            self.log_STRUCT(f"L_right+L_left: {self.L_right+self.L_left}")
            self.log_STRUCT(f"n: {self.n}")

    def calculate_structure_steps(self):
        """
        this function create the steps to limit the index in the structure and mass array
        """
        self.structure_steps = []

        self.structure_steps.append(-round(self.L0_left/self.dx))
        for _ in range(self.num_qb_left):
            self.structure_steps.append(round(self.L_qw_left/self.dx)+self.structure_steps[-1])
            self.structure_steps.append(round(self.L_qb_left/self.dx)+self.structure_steps[-1])
        self.x_cut = self.structure_steps[-1]
        self.structure_steps.append(round(self.L_cqw/self.dx)+self.structure_steps[-1])
        for _ in range(self.num_qb_right):
            self.structure_steps.append(round(self.L_qb_right/self.dx)+self.structure_steps[-1])
            self.structure_steps.append(round(self.L_qw_right/self.dx)+self.structure_steps[-1])
        # self.structure_steps.append(round(self.L0_right/self.dx)+self.structure_steps[-1])
        self.structure_steps.append(round(self.n))
        if self.dev_mode:
            barrier = True
            for struc in self.structure_steps:
                if barrier:
                    # self.list_logs_structure_limits.append(str(struc) + " barrier")
                    # print(str(struc) + " barrier")
                    barrier = False
                else:
                    # self.list_logs_structure_limits.append(str(struc) + " well")
                    # print(str(struc) + " well")
                    barrier = True

    def create_structure_array(self):
        self.x = np.arange(0,self.n,1)
        self.x = self.x * self.dx
        self.x = self.x + self.L_left
        self.x = np.round(self.x, 11)

        if self.dev: self.log_STRUCT(f"\nvetor x da super-rede")
        if self.dev:
            for i in range(len(self.x)):
                self.log_STRUCT(f"*j: {i+1} | x[j]: {self.x[i]}")

    def mass(self, m_qw, m_qb):
        m = np.empty(self.n)
        m[:] = np.nan
        barrier = True
        lim0 = 0
        for lim in self.structure_steps:
            if barrier:
                m[lim0:lim] = m_qb
                barrier = False
            else:
                m[lim0:lim] = m_qw
                barrier = True
            lim0 = lim
        # print("--------")
        # print(self.structure_steps)
        # print("self.n:", self.n)
        # print("step-1:", self.structure_steps[-1])
        # print(m[:2], m[-2:])
        return m 

    def potencial(self):
        if self.dev: self.log_POT(f"### Potencial Python ###")
        self.L_bm_left = self.num_qb_left*(self.L_qb_left+self.L_qw_left)    # Comprimento do espelho de Bragg a esquerda do poco quantico de defeito
        self.L0        = self.L_bm_left							 # L0 e a comprimento do ultimo poco quantico da super-rede	
        self.L0_left   = self.L0                                   # Guardando esse valor para ser usado no looping da construcao da super-rede

        self.v = np.empty(self.n)
        self.v[:] = np.nan
        barrier = True
        lim0 = 0
        for lim in self.structure_steps:
            if barrier:
                self.v[lim0:lim] = self.Vo
                barrier = False
            else:
                self.v[lim0:lim] = 0
                barrier = True
            lim0 = lim
        if self.dev_mode:
            for v, pot in enumerate(self.v):
                x = -self.L_left + (self.dx*v)

        self.v[0] = 10000*self.e_charge
        self.v[-1] = 10000*self.e_charge
        if self.dev:
            for i in range(len(self.v)):
                self.log_POT(f"*j: {i+1} | v[j]: {self.v[i]}")
                
    def norm_psi(self, psi):
        vec_probability = np.real(psi * np.conjugate(psi))
        norm = np.trapz(vec_probability, self.x)

        vec_probability = vec_probability / norm
        # Definition of the amplitude (100 meV) and energy
        vec_probability = 0.05 * vec_probability / max(vec_probability)

        # Normalization of the Wavefunction
        psi = psi / sm.sqrt(norm)
        return psi, vec_probability

    def funcWave_Numerov2(self, Ener, m):
        psi = np.empty(self.n, dtype=np.complex_)
        psi[:] = np.nan

        psi_left = np.empty(self.n, dtype=np.complex_)
        psi_left[:] = np.nan

        psi_right = np.empty(self.n, dtype=np.complex_)
        psi_right[:] = np.nan

        ki = np.empty(self.n, dtype=np.complex_)
        ki[:] = np.nan

        psi_left[0] = 0
        psi_left[1] = 0.1

        psi_right[-1] = 0
        psi_right[-2] = 0.1
        # self.list_of_logs.append("----------------------------------")
        # self.list_of_logs.append("----------------------------------")
        # self.list_of_logs.append("----------------------------------")
        # self.list_of_logs.append(" numerov2")

        beta = self.dx*self.dx/12
        # self.list_of_logs.append("beta " + str(beta))
        # calculando a funcao de onda do inicio da estrutura ate o meio da estrutura
        for i in range(1, self.x_cut):
            am = (m[i-1] + m[i])/2 # media das massas do eletron no ponto i-1
            aux1 = (2*am*(Ener-self.v[i-1]))/self.hbar/self.hbar
            ki[i-1] = sm.sqrt(aux1)
            
            am = (m[i] + m[i+1])/2 # media das massas do eletron no ponto i
            aux1 = (2*am*(Ener-self.v[i]))/self.hbar/self.hbar
            ki[i] = sm.sqrt(aux1)

            am = (m[i+1] + m[i+2])/2 # media das massas do eletron no ponto i+1
            aux1 = (2*am*(Ener-self.v[i+1]))/self.hbar/self.hbar
            ki[i+1] = sm.sqrt(aux1)

            a = 2 - 10*beta*ki[i]**2
            b = 1 + beta*ki[i-1]**2
            c = 1 + beta*ki[i+1]**2

            psi_left[i+1] = (a*psi_left[i] - b*psi_left[i-1])/c
        # for i in range(5):
        #     self.list_of_logs.append("psi_left " + str(psi_left[i+1]))
            


        # calculando a funcao de onda do final da estrutura ate o meio da estrutura
        for i in range(self.n-2, self.x_cut, -1):
            am = (m[i-2] + m[i-1])/2 # media das massas do eletron no ponto i-1
            aux1 = (2*am*(Ener-self.v[i-1]))/self.hbar/self.hbar
            ki[i-1] = sm.sqrt(aux1)

            am = (m[i] + m[i-1])/2 # media das massas do eletron no ponto i
            aux1 = (2*am*(Ener-self.v[i]))/self.hbar/self.hbar
            ki[i] = sm.sqrt(aux1)

            am = (m[i] + m[i+1])/2 # media das massas do eletron no ponto i+1
            aux1 = (2*am*(Ener-self.v[i+1]))/self.hbar/self.hbar
            ki[i+1] = sm.sqrt(aux1)

            a = 2 - 10*beta*ki[i]**2
            b = 1 + beta*ki[i-1]**2
            c = 1 + beta*ki[i+1]**2

            psi_right[i-1] = (a*psi_right[i] - c*psi_right[i+1])/b	
        # for i in range(self.n-2, self.n-7, -1):
        #     self.list_of_logs.append("psi_right " + str(i) + "  " + str(psi_right[i]))
        
        # for i in range(self.x_cut+5,self.x_cut, -1):
        #     self.list_of_logs.append("psi_right " + str(i) + "  " + str(psi_right[i]))
                    
        k = self.nn + 1

        #A funcao de onda a direita apresenta uma paridade inversa que a funcao de onda da esquerda
        #para resolver o problema estou multiplicando por -1 as funcoes de onda impares
        for i in range(self.n-1,self.x_cut, -1):
            psi_right[i-1] = (-1)**(k+1)*psi_right[i-1]

        # self.list_of_logs.append("************* paridade inversa")
        # for i in range(self.n-2, self.n-7, -1):
        #     self.list_of_logs.append("psi_right " + str(i) + "  " + str(psi_right[i]))
        
        # for i in range(self.x_cut+5,self.x_cut, -1):
        #     self.list_of_logs.append("psi_right " + str(i) + "  " + str(psi_right[i]))
        
        #normalizando a funcao de onda da esquerda com a amplitude da funcao de onda da direita
        # self.list_of_logs.append("************* normalizar")
        # self.list_of_logs.append("self.x_cut " + str(self.x_cut))
        # self.list_of_logs.append("psi_right[self.x_cut-2] " + str(psi_right[self.x_cut-2]))
        # self.list_of_logs.append("psi_right[self.x_cut-1] " + str(psi_right[self.x_cut-1]))
        # self.list_of_logs.append("psi_right[self.x_cut] " + str(psi_right[self.x_cut]))
        # self.list_of_logs.append("psi_right[self.x_cut+1] " + str(psi_right[self.x_cut+1]))
        # self.list_of_logs.append("psi_right[self.x_cut+2] " + str(psi_right[self.x_cut+2]))
        # self.list_of_logs.append("psi_left[self.x_cut] " + str(psi_left[self.x_cut]))
        for i in range(0,self.x_cut):
            # self.list_of_logs.append("psi_left[i] " + str(psi_left[i]))
            psi_left[i] = psi_left[i]*psi_right[self.x_cut]/psi_left[self.x_cut]
            # self.list_of_logs.append("psi_left[i] " + str(psi_left[i]))
            
        
        #juntando as funcoes de onda esquerda e direita em apenas uma funcao de onda
        for i in range(0,self.x_cut):
            psi[i] = psi_left[i]

        for i in range(self.x_cut,self.n):
            psi[i] = psi_right[i]
        return psi

    def numerov(self, Ener, m):
        psi = np.empty(self.n, dtype=np.complex_)
        psi[:] = np.nan
        psi[0] = 0
        psi[1] = 0.1
        new_m = (m+np.roll(m, 1))/2

        new_m_p1 = np.roll(new_m, -1)
        
        new_Ener = Ener-self.v
        new_Ener_n1 = np.roll(new_Ener, 1)
        new_Ener_p1 = np.roll(new_Ener, -1)

        aux1_a = sm.sqrt(2*(new_m*new_Ener_n1)/self.hbar2)
        aux1_b = sm.sqrt(2*(new_m_p1*new_Ener)/self.hbar2)
        aux1_c = sm.sqrt(2*(new_m_p1*new_Ener_p1)/self.hbar2)

        a = 2 - 10*self.beta*aux1_b**2
        b = 1 + self.beta*aux1_a**2
        c = 1 + self.beta*aux1_c**2

        for i in range(1, self.n-1):
            psi[i+1] = (a[i]*psi[i] - b[i]*psi[i-1])/c[i]
        return psi

    def wavefunction_numerov(self):
        prints = False
        if self.simulation_type == "develop": prints = False
        self.result_wavefunction = []
        self.pwf_E
        m_qw = self.m_InGaAs*(1 + self.pwf_E/self.Ew)
        m_qb = self.m_InAlAs*(1  + (self.pwf_E-self.Vo)/self.Eb)
        m = self.mass(m_qw, m_qb)
        psi = self.numerov(self.pwf_E, m)
        psi_old = psi[-1]
        Ener = 0
        Ener_old = self.pwf_E
        psi0 = []
        Energy = []
        
        ne = 0
        for j in range(1, int(self.pwf_En)):   # numero de autovalores do problema
            psi_old = psi[-1].copy()
            #   calculando a funcao de onda do eletron para energia Ener  
            Ener = round(self.pwf_E_initial + self.pwf_dE*j, 11)
            Ener = Ener*self.e_charge*1E-3  #[J]
            m_qw = self.m_InGaAs*(1 + Ener/self.Ew)
            m_qb = self.m_InAlAs*(1  + (Ener-self.Vo)/self.Eb)
            m = self.mass(m_qw, m_qb)
            psi = self.numerov(Ener, m)
            psi_New = psi[-1].copy()
            
            #   condicao necessaria para o "do while" operar (procure a regiao em que a multiplicacao passou por zero)	
            mult1 = np.real(psi_old*psi_New)
            if prints: print("ener: {:06.3f} || multi1: {:.2e} || {:.2e} {:.2e}".format(Ener*1000/self.e_charge, mult1, np.real(psi_old), np.real(psi_New)))
            if (mult1 <= 0):
                if prints: print("ener: ", round(Ener*1000/self.e_charge, 4))
                if prints: print()
                self.list_logs_numerov.append("---" + str(Ener))
                Ener_new = Ener
                k = 0

                continuar = True
                while continuar:
                    if prints: print("k:", k)
                    # self.list_of_logs.append(str(k) + " |||||||||||||||||||||||||||||||||||||||||||||||||| ")
                    #        15  continue 
                    #	------------------------------------------------------
                    #	calculando o funcao de onda do eletron com energia Ener_Old
                    Ener = Ener_old	  
                    m_qw = self.m_InGaAs*(1 + Ener/self.Ew)
                    m_qb = self.m_InAlAs*(1  + (Ener-self.Vo)/self.Eb)
                    m = self.mass(m_qw, m_qb)
                    psi = self.numerov(Ener, m)
                    psi_old = psi[-1].copy()

                    # self.list_of_logs.append("self.m_InGaAs " + str(k) + " || " + str(self.m_InGaAs))
                    # self.list_of_logs.append("self.m_InAlAs " + str(k) + " || " + str(self.m_InAlAs))
                    # self.list_of_logs.append("Ener " + str(k) + " || " + str(Ener))
                    # self.list_of_logs.append("self.Ew " + str(k) + " || " + str(self.Ew))
                    # self.list_of_logs.append("self.Vo " + str(k) + " || " + str(self.Vo))
                    # self.list_of_logs.append("self.Eb " + str(k) + " || " + str(self.Eb))

                    
                    # self.list_of_logs.append("m_qw " + str(k) + " || " + str(m_qw))
                    # self.list_of_logs.append("m_qb " + str(k) + " || " + str(m_qb))
                    # self.list_of_logs.append("m " + str(k) + " || " + str(m))
                    # self.list_of_logs.append("psi " + str(k) + " || " + str(psi))
                    # self.list_of_logs.append("psi_old " + str(k) + " || " + str(psi_old))
                    
                    #	------------------------------------------------------
                    #	calculando o funcao de onda do eletron com energia Ener_New
                    Ener = Ener_new	  
                    m_qw = self.m_InGaAs*(1 + Ener/self.Ew)
                    m_qb = self.m_InAlAs*(1  + (Ener-self.Vo)/self.Eb)
                    m = self.mass(m_qw, m_qb)
                    psi = self.numerov(Ener, m)
                    psi_New = psi[-1].copy()
                    # self.list_of_logs.append("psi_New " + str(k) + " || " + str(psi_New))
                    
                    #	------------------------------------------------------
                    #	condicao para saber se as psiN (ener_old e ener_new) passar por zero ou nao
                    mult1 = np.real(psi_old*psi_New)
                    #	------------------------------------------------------	

                    #	------------------------------------------------------	
                    #	Inicio do metodo de bissecao para encontrar os autovalores da funcao de onda		
                    if (mult1 < 0):
                        #caso a condicao nao tenha passado por zero, o valor ener sera a media entre ener_old e ener_new	
                        Ener = (Ener_old + Ener_new)/2.0
                        # self.list_of_logs.append("Ener " + str(k) + " || " + str(Ener))
                    
                    #	------------------------------------------------------
                    #	calculando o funcao de onda para nova energia Ener	  
                    m_qw = self.m_InGaAs*(1 + Ener/self.Ew)
                    m_qb = self.m_InAlAs*(1  + (Ener-self.Vo)/self.Eb)
                    m = self.mass(m_qw, m_qb)
                    psi = self.numerov(Ener, m)
                    psi_New = psi[-1].copy()
                    # self.list_of_logs.append("psi_New " + str(k) + " || " + str(psi_New))

                    #	------------------------------------------------------
                    #	condicao para saber se as psiN (ener_old e ener_new) passou por zero ou nao	
                    mult1 = np.real(psi_old*psi_New)
                    #	------------------------------------------------------	
                    
                    if (mult1 < 0):
                        Ener_new = Ener
                    else:
                        Ener_old = Ener

                    k = k + 1 #contador de vezes que o metodo da bissecao foi utilizado
                    if (k == 100):
                        continuar = False
                        # go to 16 #evitando que o loop se torne infinito 
                        # Caso a amplitude a funcao de onda em psi(n) for menor que a tolerancia, o problema foi resolvido 	
                    if (abs(psi_New) > self.tol and continuar):
                        continuar = True
                        #goto 15 
                    else:
                        continuar = False

                #16  	continue
                # Quando ocorre o loop infinito significa que a funcao de onda esta explodindo no final da estrutura
                #Para contornar esse problema, o calculo da funcao de onda precisa ser dividido em duas partes: um loop que vai do 
                #inicio da estrutura i = 0 ate no meio da estrutura n/2 e outro loop saindo do final da estrutura 
                # no ponto n ate no meio da estrutura n/2
                ne = ne + 1
                if (Ener < self.Vo):
                    m_qw = self.m_InGaAs*(1 + Ener/self.Ew)
                    m_qb = self.m_InAlAs*(1  + (Ener-self.Vo)/self.Eb)
                    m = self.mass(m_qw, m_qb)
                    psi = self.funcWave_Numerov2(Ener, m)
                    for val in psi:
                        self.list_logs_numerov.append(str(val))

                #write(*,*)"The energy is", 1E3*Ener/e_charge
                psi0.append([ne, psi, Ener])
                Energy.append([ne, Ener])
                # print("ne y self.nn", ne, self.nn)
                if (ne == self.nn):
                    break
                    # pass
                    # go to 17 # se o numero de autoestados encontrados (ne) for igual ao numero de autoestado desejados(nn), finalize o loop
                    # Aqui estou calculando novamente a funcao de onda completa pois quando uso o numerov2 eu coloco psi nas bordas e zero e com isso 
                    # a condicao mult1 de calculo da funcao de onda nao e sastifeita
                else:
                    Ener = self.pwf_E_initial + self.pwf_dE*j#
                    m_qw = self.m_InGaAs*(1 + Ener/self.Ew)
                    m_qb = self.m_InAlAs*(1  + (Ener-self.Vo)/self.Eb)
                    m = self.mass(m_qw, m_qb)
                    psi = self.numerov(Ener, m)
                
        #17 continue  
        #---------------------------------------------------------------
        #	Normalizando as funcoes de onda	
        
        for i in range(0,ne):
            psi = psi0[i][1]
            EEEE = psi0[i][2]
            psi, vec_probability = self.norm_psi(psi)
            psi0[i] = [i, psi, vec_probability, EEEE]
        self.psi0 = psi0
        self.Energy = Energy
        if self.dev_mode:
            for i in range(len(self.psi0[0][2])):
                line_text = "{}".format(round(self.x[i]*1e+9,2))
                for j, [ne, psi, vec_probability, EEEE] in enumerate(self.psi0):
                    Energy = round(EEEE/self.e_charge,6)*1000
                    line_text = line_text + " " + "{:.6}".format(round((vec_probability[i]*400*1)+Energy,1))
                self.list_logs_wavefunctions.append(line_text)

    # Transfer matrix subroutine
    def transfer_matrix(self, Ener):
        # if self.dev: self.log(305, f"### transfer_matrix Python ###")
        T0 = np.eye(2, dtype=np.complex128)
        T = np.eye(2, dtype=np.complex128)
        
        # if self.dev: self.log_matrix22(305, T0, 'T0')

        for i in range(0, self.n-1):
            if Ener > self.v[i]:
                ki = np.sqrt((2 * self.m[i] * (Ener - self.v[i])) / self.hbar2)
            else:
                ki = self.ic * np.sqrt((2 * self.m[i] * (self.v[i] - Ener)) / self.hbar2)
            # if self.dev: self.log(305, f"{i+1} - ki: {dev_complex(ki)}", i, [518, 520])

            if Ener > self.v[i + 1]:
                ki_next = np.sqrt((2 * self.m[i + 1] * (Ener - self.v[i + 1])) / self.hbar2)
            else:
                ki_next = self.ic * np.sqrt((2 * self.m[i + 1] * (self.v[i + 1] - Ener)) / self.hbar2)
            # if self.dev: self.log(305, f"** {i+1} - ki_next: {dev_complex(ki_next)}")

            P_inicial = np.array([[np.exp(self.ic * ki * (self.dx) / 2), 0], [0, np.exp(-self.ic * ki * (self.dx) / 2)]], dtype=np.complex128)
            P_final = np.array([[np.exp(self.ic * ki_next * (self.dx) / 2), 0], [0, np.exp(-self.ic * ki_next * (self.dx) / 2)]], dtype=np.complex128)

            beta = (ki * self.m[i + 1]) / (ki_next * self.m[i])
            Dn = np.array([[0.5 * (1 + beta), 0.5 * (1 - beta)], [0.5 * (1 - beta), 0.5 * (1 + beta)]], dtype=np.complex128)
            

            # aux = np.matmul(P_final, Dn)
            # Ti = np.matmul(aux, P_inicial)
            # T = np.matmul(Ti, T0)
            
            aux = np_multiply(P_final, Dn)
            Ti = np_multiply(aux, P_inicial)
            T = np_multiply(Ti, T0)
            T0 = T
            # if self.dev: self.log(305, f"beta: {dev_complex(beta)}")
            
            # if self.dev: self.log_matrix22(305, P_final, 'P_final', i, [518, 520])
            # if self.dev: self.log_matrix22(305, Dn, 'Dn', i, [518, 520])
            # if self.dev: self.log_matrix22(305, aux, 'aux', i, [518, 520])
            # if self.dev: self.log(305, '##############', i, [518, 520])
            
            # if self.dev: self.log_matrix22(305, P_inicial, 'P_inicial', i, [518, 520])
            # if self.dev: self.log_matrix22(305, Ti, 'Ti', i, [518, 520])
            # if self.dev: self.log(305, '##############', i, [518, 520])
            
            # if self.dev: self.log_matrix22(305, T0, 'T0', i, [518, 520])
            # if self.dev: self.log_matrix22(305, T, 'T', i, [518, 520])

        return T

    ##############################################################
    # Eigenstates Transfer Matrix subroutine
    def Eigenstates_Transfer_Matrix_energy(self):
        if self.dev: self.log_ETM_E(f"### Eigenstates_Transfer_Matrix_energy Python ###")
        # self.Energy = np.empty(self.n)
        self.Energy = []
        self.psi0 = []
        interaction = 0
        if self.dev: self.log_ETM('inicia Eigenstates_Transfer_Matrix Python')
            
        Ener_old = (self.pwf_E_initial + 1 * self.pwf_dE)  # [meV]
        if self.dev: self.log_ETM_E(f"Ener_old: {Ener_old} [meV]")
        Ener_old = Ener_old * self.e_charge * 1E-3  # [J]
        if self.dev: self.log_ETM_E(f"Ener_old: {Ener_old} [J]")
        Ener = Ener_old
        m_qw = self.m_InGaAs*(1 + Ener/self.Ew)
        m_qb = self.m_InAlAs*(1  + (Ener-self.Vo)/self.Eb)
        if self.dev: self.log_ETM_E(f"m_qw: {m_qw}")
        if self.dev: self.log_ETM_E(f"m_qb: {m_qb}")
        self.m = self.mass(m_qw, m_qb)
        # if self.dev: self.log_ETM_E(f"masa")
        # for _i, _m in enumerate(self.m):
        #     if self.dev: self.log_ETM_E(f"**{_i + 1} : {_m}")
            
        T = self.transfer_matrix(Ener)
        if self.dev: self.log_ETM_E_matrix22(T, 'T')
        
        T22_old = np.real(T[1, 1])
        nn = 0  # Resetting the eigenstate counter

        if self.dev: self.log_ETM_E(f'primeiro for {int(self.pwf_En-1)}')
        for jn in range(2, int(self.pwf_En)):   # numero de autovalores do problema
            # -------------------------------------------------------------------
	        # Calculando a ener_new para comecar o loomp
	
            Ener_new = (self.pwf_E_initial + jn * self.pwf_dE)  # [meV]
            if self.dev: print(f"{jn + 1} de {int(self.pwf_En)} -- {Ener_new}", end="\r")
            # print(f"ene {jn + 1} de {int(self.pwf_En)} -- {Ener_new}")
            if self.dev: self.log_ETM_E(f'{jn} de {int(self.pwf_En-1)} / {Ener_new}')
            # if self.dev: self.log_ETM(f'Ener_new: {Ener_new}')
            Ener_new = Ener_new * self.e_charge * 1E-3  # [J]
            Ener = Ener_new
            
            m_qw = self.m_InGaAs*(1 + Ener/self.Ew)
            m_qb = self.m_InAlAs*(1  + (Ener-self.Vo)/self.Eb)
            self.m = self.mass(m_qw, m_qb)
            T = self.transfer_matrix(Ener)
            T22_new = np.real(T[1, 1])


            # !-----------------------------------------------------------------
	        # !           Metodo da bissecao para encontrar as autoenergias	
	        # !-----------------------------------------------------------------
            
            mult2 = T22_old * T22_new
            if mult2 < 0:
                while abs(Ener_new - Ener_old) > self.tol:
                    Ener = (Ener_old + Ener_new) / 2
                    m_qw = self.m_InGaAs*(1 + Ener/self.Ew)
                    m_qb = self.m_InAlAs*(1  + (Ener-self.Vo)/self.Eb)
                    self.m = self.mass(m_qw, m_qb)
                    T = self.transfer_matrix(Ener)
                    T22_new = np.real(T[1, 1])
                    mult2 = T22_old * T22_new
                    if mult2 < 0:
                        Ener_new = Ener
                    else:
                        Ener_old = Ener
                    interaction += 1
                    if interaction > self.max_interactions:
                        break
                else:
                    nn += 1
                    # self.Energy.append([nn, Ener * 1e3 / self.e_charge])
                    # print(nn, Ener * 1e3 / self.e_charge, 'Eigenstates_Transfer_Matrix_energy - 1016')
                    Ener = Ener # - (self.pwf_dE * self.e_charge / 1e3)
                    self.Energy.append([nn, Ener])
                    if self.dev: self.log_ETM_E(f'{nn} Ener: {Ener}')
                    if self.dev: print(f'{nn} Ener: {Ener}')

            Ener = self.pwf_E_initial + self.pwf_dE * jn
            Ener = Ener * self.e_charge * 1E-3  # [J]
            m_qw = self.m_InGaAs*(1 + Ener/self.Ew)
            m_qb = self.m_InAlAs*(1  + (Ener-self.Vo)/self.Eb)
            self.m = self.mass(m_qw, m_qb)
            T = self.transfer_matrix(Ener)
            T22_old = np.real(T[1, 1])
            Ener_old = Ener
            
            ##################################################################
            # para colocar no arquivo das energias
            # print(f"PC 0864 - |{self.Energy[-1]}| {jn} de {int(self.pwf_En)}")

    def Eigenstates_Transfer_Matrix_wf(self):
        nn = len(self.Energy)
        # nn = 1
        if self.dev: self.log(306, f'### Eigenstates_Transfer_Matrix_energy Python ###')
        if self.dev: self.log(306, f'segundo for / nn: {nn}')
        if self.dev: print(f'segundo for / nn: {nn}')
        for i in range(0, nn):
            if self.dev: print(f"{i + 1} de {nn}", end="\r")
            if self.dev: self.log(306, f'i: {i+1}')
            Ener = self.Energy[i][1]
            if self.dev: self.log(306, f'Ener: {Ener} [j]')
            if self.dev: self.log(306, f'Ener: {1E3*Ener/self.e_charge} [meV]')
          # print(f"{i + 1} de {nn} -- {1E3*Ener/self.e_charge}")
            m_qw = self.m_InGaAs*(1 + Ener/self.Ew)
            m_qb = self.m_InAlAs*(1  + (Ener-self.Vo)/self.Eb)
            if self.dev: self.log(306, f'm_qw: {m_qw}')
            if self.dev: self.log(306, f'm_qb: {m_qb}')
            self.m = self.mass(m_qw, m_qb)
            # print('Ener < self.Vo:', Ener, ' < ', self.Vo)
            if Ener < self.Vo:
                # print(f'below barrier | nn: {nn} | Ener: {1E3*Ener/self.e_charge}')
                if self.dev: self.log(306, f'below barrier | nn: {nn} | Ener: {1E3*Ener/self.e_charge}')
                psi = self.new_WaveFunc_transfer_matrix_split(nn, Ener, i)
            else:
                # print(f'above barrier | nn: {nn} | Ener: {1E3*Ener/self.e_charge}')
                if self.dev: self.log(306, f'above barrier | nn: {nn} | Ener: {1E3*Ener/self.e_charge}')
                psi = self.new_wavefunction_transfer_matrix(nn, Ener)
            # self.psi0.append([i, self.psi[:], self.Energy[i][1]])
            self.psi0.append([i, psi[:], self.Energy[i][1]])
            
            
        if self.dev: self.log_ETM('fim Eigenstates_Transfer_Matrix')
        # test normalizar deixar 4 valores no psi0
        
            
        for i in range(len(self.psi0)):
            psi = self.psi0[i][1]
            EEEE = self.psi0[i][2]
            
            # psi_norm = [(c.real**2 + c.imag**2)* 0.4e-6 for c in psi]
            psi_norm = [(c.real**2 + c.imag**2) for c in psi]
            # psi_norm = (psi2 * 0.4e-6) #  + (1E3*EEEE/self.e_charge)
            self.psi0[i] = [i, psi, psi_norm, EEEE]
            
            for _psi, _psi_norm, _x in zip(psi, psi_norm, self.x):
                if self.dev: self.log(306, f"{round(_x/self.nm,3)} {_psi_norm}")
                
    def test_falta_arreglar_new_wavefunction_transfer_matrix(self, n, Ener):
      # print('-- new_wavefunction_transfer_matrix')
        psi = np.empty(self.n, dtype=np.complex_)
        psi[:] = np.nan
        
        T0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
        WaveFunc = np.array([[-1.0 + 0.0j, 0.0 + 0.0j],
                                [1.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex128)
        
        psi = np.zeros(self.n)
        ki = np.zeros(self.n, dtype=np.complex128)
        
        
        
        for i in range(self.n-1):

            if Ener > self.v[i]:
                ki[i] = np.sqrt((2 * self.m[i] * (Ener - self.v[i])) /(self.hbar2))
            else:
                ki[i] = self.ic * np.sqrt((2 * self.m[i] * (self.v[i] - Ener)) /(self.hbar2))

            if Ener > self.v[i]:
                ki[i+1] = np.sqrt((2 * self.m[i+1] * (Ener - self.v[i+1])) /(self.hbar2))
            else:
                ki[i+1] = self.ic * np.sqrt((2 * self.m[i+1] * (self.v[i+1] - Ener)) /(self.hbar2))

            P_inicial = np.array([[np.exp(self.ic * ki[i] * self.dx / 2), 0.0], [0.0, np.exp(-self.ic * ki[i] * self.dx / 2)]], dtype=np.complex128)
            P_final = np.array([[np.exp(self.ic * ki[i+1] * self.dx / 2), 0.0], [0.0, np.exp(-self.ic * ki[i+1] * self.dx / 2)]], dtype=np.complex128)

            beta = (ki[i] * self.m[i+1]) / (ki[i+1] * self.m[i])
            Dn = np.array([[0.5 * (1 + beta), 0.5 * (1 - beta)], [0.5 * (1 - beta), 0.5 * (1 + beta)]], dtype=np.complex128)

            aux = np_multiply(P_final, Dn)
            Ti = np_multiply(aux, P_inicial)
            WaveFunc = np_multiply(Ti, WaveFunc)
            
            
            # # comecando o calculo de funcao de onda
            # # IEEE Journal of Quantum electronics, vol. 45, no 9 september 2009
            # WaveFunc = ((P_final @ Dn) @ P_inicial) @ WaveFunc
            # # esta matriz esta invertida com o exemplo do artigo acima. Eu faco a onda incidir
            # # pela direita e sair pela esquerda. O contrario do artigo
            psi[i-1] = WaveFunc[0, 0] + WaveFunc[1, 0]
        
        

            # # print(f"WaveFunc.shape: {WaveFunc.shape}, {WaveFunc}")
            # # print(f"len(self.psi): {len(psi)}")
            # # print(f"WaveFunc[0, 0]: {WaveFunc[0, 0]}")
            # # print(f"WaveFunc[1, 0]: {WaveFunc[1, 0]}")
            # # print(f"xxx: {xxx}")
            # # psi[i-1] = WaveFunc[0, 0] + WaveFunc[1, 0]
            # psi[i-1] = WaveFunc[0] + WaveFunc[1]

        prod = np.conj(psi) * psi
        soma = np.trapz(prod, dx=self.dx)

        psi /= np.sqrt(soma)
        self.psi = psi
        
    def new_wavefunction_transfer_matrix(self, n, Ener):
        if self.dev: self.log(316, f'### wavefunction_transfer_matrix python ###')
        psi = np.empty(self.n, dtype=np.complex_)
        psi[:] = np.nan
        T0 = np.array([[1+0j, 0+0j], [0+0j, 1+0j]])
        
        
        WaveFunc = np.array([[-1.0 + 0.0j, 0 + 0j],
                           [1.0 + 0.0j, 0 + 0j]], dtype=np.complex128)
        
        if self.dev: self.log_matrix22(316, WaveFunc, 'WaveFunc')
        
        for i in range(self.n-1):
            if self.dev: self.log(316, f'{i+1} de {self.n-1}')
            if Ener > self.v[i]:
                ki = np.sqrt((2 * self.m[i] * (Ener - self.v[i])) /(self.hbar2))
            else:
                ki = self.ic * np.sqrt((2 * self.m[i] * (self.v[i] - Ener)) /(self.hbar2))

            if Ener > self.v[i+1]:
                ki_next = np.sqrt((2 * self.m[i+1] * (Ener - self.v[i+1])) /(self.hbar2))
            else:
                ki_next = self.ic * np.sqrt((2 * self.m[i+1] * (self.v[i+1] - Ener)) /(self.hbar2))
                
            # if self.dev: self.log(316, f'**m {self.m[i]}')
            # if self.dev: self.log(316, f'**v {self.v[i]}')
            # if self.dev: self.log(316, f'**m_next {self.m[i+1]}')
            # if self.dev: self.log(316, f'**v_next {self.v[i+1]}')
            # if self.dev: self.log(316, f'**ki {dev_complex(ki)}')
            # if self.dev: self.log(316, f'**ki_next {dev_complex(ki_next)}')

            P_inicial = np.array([[np.exp(self.ic * ki * self.dx / 2), 0.0], [0.0, np.exp(-self.ic * ki * self.dx / 2)]], dtype=np.complex128)
            P_final = np.array([[np.exp(self.ic * ki_next * self.dx / 2), 0.0], [0.0, np.exp(-self.ic * ki_next * self.dx / 2)]], dtype=np.complex128)

            beta = (ki * self.m[i+1]) / (ki_next * self.m[i])
            Dn = np.array([[0.5 * (1 + beta), 0.5 * (1 - beta)], [0.5 * (1 - beta), 0.5 * (1 + beta)]], dtype=np.complex128)

            aux = np_multiply(P_final, Dn)
            Ti = np_multiply(aux, P_inicial)
            if i > 1910:
                if self.dev: self.log_matrix22(316, Ti, 'Ti')
                if self.dev: self.log_matrix22(316, WaveFunc, 'WaveFunc')
            
            WaveFunc = np_multiply(Ti, WaveFunc)
            
            # if self.dev: self.log_matrix22(316, P_inicial, 'P_inicial')
            # if self.dev: self.log_matrix22(316, P_final, 'P_final')
            # if self.dev: self.log_matrix22(316, Dn, 'Dn')
            # if self.dev: self.log_matrix22(316, aux, 'aux')
            
            if i > 1910:
                if self.dev: self.log_matrix22(316, WaveFunc, '=WaveFunc')
            
            
            psi[i] = WaveFunc[0, 0] + WaveFunc[1, 0]
            psi[i] = psi[i].real
            if i > 1910:
                if self.dev: self.log(316, f'**psi[i] {dev_complex(psi[i])}')

        psi = np.nan_to_num(psi, nan=0)
        # psi = psi.real
        prod = np.conj(psi) * psi
        soma = np.trapz(prod, dx=self.dx)
        
        if self.dev: self.log(316, f'soma: {soma}')
        for i in range(self.n):
            if self.dev: self.log(316, f'{i+1} | {dev_complex(psi[i])}')

        psi /= np.sqrt(soma)
        
        
        self.psi = psi
        return psi


    def new_WaveFunc_transfer_matrix_split(self, n, Ener, k):
        if self.dev: self.log(306, f"start WaveFunc_transfer_matrix_split | n: {self.n} | Ener: {Ener}")
        T0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
        WaveFunc_L = np.array([[-1.0 + 0.0j, 0.0 + 0.0j],
                                [1.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex128)
        
        WaveFunc_R = np.array([[-1.0 + 0.0j, 0.0 + 0.0j],
                                [1.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex128)

        psi = np.zeros(self.n)
        psi_L = np.zeros(self.n, dtype=np.complex128)
        psi_R = np.zeros(self.n, dtype=np.complex128)
        ki_L = np.zeros(self.n, dtype=np.complex128)
        ki_R = np.zeros(self.n, dtype=np.complex128)
        
        lim_1 = [497, 522]
        if self.dev: self.log_ETM(f"primer sub do")
        if self.dev: self.log_ETM(f"x_cut: {self.x_cut}")
        for i in range(0, self.x_cut+1):
            # if self.dev: self.log(306, f"i x_cut: {i+1} Ener: {Ener} | Ener: {Ener * 1e3 / self.e_charge}")
            # if self.dev: self.log_ETM(f"i x_cut: {i+1}")

            if Ener > self.v[i]:
                ki_L[i] = np.sqrt((2 * self.m[i] * (Ener - self.v[i])) /(self.hbar2))
            else:
                ki_L[i] = self.ic * np.sqrt((2 * self.m[i] * (self.v[i] - Ener)) /(self.hbar2))
            # if lim_1[0] < i < lim_1[1]:
            #     if self.dev: self.log_ETM(f"ki_L[i]: {dev_complex(ki_L[i])}")
            
            if Ener > self.v[i+1]:
                # if lim_1[0] < i < lim_1[1]:
                #     if self.dev: self.log_ETM(f"optun")
                ki_L[i + 1] = np.sqrt((2 * self.m[i+1] * (Ener - self.v[i+1])) /(self.hbar2))
            else:
                # if lim_1[0] < i < lim_1[1]:
                #     if self.dev: self.log_ETM(f"optdo")
                ki_L[i + 1] = self.ic * np.sqrt((2 * self.m[i+1] * (self.v[i+1] - Ener)) /(self.hbar2))
            # if lim_1[0] < i < lim_1[1]:
            #     if self.dev: self.log_ETM(f"ki_L[i+u]: {dev_complex(ki_L[i + 1])}")
            
            P_inicial_L = np.array([[np.exp(self.ic * ki_L[i] * (self.dx) / 2), 0.0], [0.0, np.exp(-self.ic * ki_L[i] * (self.dx) / 2)]], dtype=np.complex128)
            P_final_L = np.array([[np.exp(self.ic * ki_L[i + 1] * (self.dx) / 2), 0.0], [0.0, np.exp(-self.ic * ki_L[i + 1] * (self.dx) / 2)]], dtype=np.complex128)
            
            # if lim_1[0] < i < lim_1[1]:
            #     if self.dev: self.log_ETM(f"* P_inicial_L: {dev_complex22(P_inicial_L)}")
            # if lim_1[0] < i < lim_1[1]:
            #     if self.dev: self.log_ETM(f"* P_final_L: {dev_complex22(P_final_L)}")

            beta = (ki_L[i] * self.m[i+1]) / (ki_L[i + 1] * self.m[i])
            Dn = np.array([[0.5 * (1 + beta), 0.5 * (1 - beta)], [0.5 * (1 - beta), 0.5 * (1 + beta)]], dtype=np.complex128)
            Dn[0, 0] = 0.5 * (1 + beta)
            Dn[0, 1] = 0.5 * (1 - beta)
            Dn[1, 0] = 0.5 * (1 - beta)
            Dn[1, 1] = 0.5 * (1 + beta)
            
            # if self.dev: self.log_ETM_matrix22(WaveFunc_L, 'WaveFunc_L')
            
            # aux = np.matmul(P_final_L, Dn)
            # Ti = np.matmul(aux, P_inicial_L)
            # WaveFunc_L = np.matmul(Ti, WaveFunc_L)
            aux = np_multiply(P_final_L, Dn)
            Ti = np_multiply(aux, P_inicial_L)
            WaveFunc_L = np_multiply(Ti, WaveFunc_L)
            
            psi_L[i] = WaveFunc_L[0, 0] + WaveFunc_L[1, 0]
            
            # if self.dev: self.log(306, f"* beta: {dev_complex(beta)}")
            # if self.dev: self.log_matrix22(306, P_final_L, 'P_final_L')
            # if self.dev: self.log_matrix22(306, Dn, 'Dn')
            # if self.dev: self.log_matrix22(306, aux, 'aux')
            
            # if self.dev: self.log_matrix22(306, P_inicial_L, 'P_inicial_L')
            # if self.dev: self.log_matrix22(306, Ti, 'Ti')
            
            # if self.dev: self.log_matrix22(306, WaveFunc_L, 'WaveFunc_L')
            
            # if self.dev: self.log(306, f"* psi_L[i]: {dev_complex(psi_L[i])}")

        # Calculando a funcao de onda da direta para esquerda com incremento -dx	
        for i in range(self.n-1, self.x_cut-1, -1):
            # if self.dev: self.log_ETM(f"i x_cut: {i+2}")

            if Ener > self.v[i]:
                ki_R[i] = np.sqrt((2 * self.m[i] * (Ener - self.v[i])) /(self.hbar2))
            else:
                ki_R[i] = self.ic * np.sqrt((2 * self.m[i] * (self.v[i] - Ener)) /(self.hbar2))

            if Ener > self.v[i - 1]:
                ki_R[i - 1] = np.sqrt((2 * self.m[i - 1] * (Ener - self.v[i - 1])) /(self.hbar2))
            else:
                ki_R[i - 1] = self.ic * np.sqrt((2 * self.m[i - 1] * (self.v[i - 1] - Ener)) /(self.hbar2))

            P_inicial_R = np.array([[np.exp(self.ic * ki_R[i] * (-self.dx) / 2), 0.0], [0.0, np.exp(-self.ic * ki_R[i] * (-self.dx) / 2)]], dtype=np.complex128)
            P_final_R = np.array([[np.exp(self.ic * ki_R[i - 1] * (-self.dx) / 2), 0.0], [0.0, np.exp(-self.ic * ki_R[i - 1] * (-self.dx) / 2)]], dtype=np.complex128)

            beta = (ki_R[i] * self.m[i - 1]) / (ki_R[i - 1] * self.m[i])
            Dn = np.array([[0.5 * (1 + beta), 0.5 * (1 - beta)], [0.5 * (1 - beta), 0.5 * (1 + beta)]], dtype=np.complex128)
            Dn[0, 0] = 0.5 * (1 + beta)
            Dn[0, 1] = 0.5 * (1 - beta)
            Dn[1, 0] = 0.5 * (1 - beta)
            Dn[1, 1] = 0.5 * (1 + beta)

            aux = np_multiply(P_final_R, Dn)
            Ti = np_multiply(aux, P_inicial_R)
            WaveFunc_R = np_multiply(Ti, WaveFunc_R)
            psi_R[i] = WaveFunc_R[0, 0] + WaveFunc_R[1, 0]
            # if self.dev: self.log_ETM(f"* {i+2} psi_R[i]: {dev_complex(psi_R[i])}")
            
        for i in range(self.n-1, self.x_cut-1, -1):
            psi_R[i] = (-1) ** (k+1) * psi_R[i]

        for i in range(0, self.x_cut + 1):
            # if self.dev: self.log(306, f"{i+1}")
            # if self.dev: self.log_ETM(f"** psi_L[i]: {dev_complex(psi_L[i])}")
            # if self.dev: self.log_ETM(f"** psi_R[x_cut]: {dev_complex(psi_R[self.x_cut])}")
            # if self.dev: self.log_ETM(f"** psi_L[x_cut]: {dev_complex(psi_L[self.x_cut])}")
            psi_L[i] = psi_L[i] * psi_R[self.x_cut] / psi_L[self.x_cut]
            # if self.dev: self.log(306, f"** psi_L[i]: {dev_complex(psi_L[i])}")

        for i in range(0, self.x_cut + 1):
            psi[i] = psi_L[i]

        for i in range(self.x_cut, self.n):
            psi[i] = psi_R[i]


        prod = np.conj(psi) * psi
        soma = np.trapz(prod, dx=self.dx)
        # if self.dev: self.log_ETM(f"soma: {soma}")

        psi /= np.sqrt(soma)
        # for i in range(len(psi)):
        #     if self.dev: self.log(306, f"* {i+1} psi[i]: {dev_complex(psi[i])}")

        self.psi = psi
        # for i in range(len(self.psi)):
        #     if self.dev: self.log_ETM(f"* {i+1} psi[i]: {self.psi[i]}")
        return psi
        
    
    ##############################################################
    
    def oscillator_strength_old(self):
        self.oscstr = []
        for p in range(len(self.psi0)):
            psi_ = np.conjugate(self.psi0[p][1])
            prod_ = psi_ * self.x * self.psi0[0][1]
            difE = (self.Energy[p][1] - self.Energy[0][1])
            soma = np.trapz(prod_, self.x)
            f_ = 2*self.me*difE*abs(soma)**2/self.hbar/self.hbar
            self.oscstr.append([f_, difE, self.Energy[p][1], self.Energy[0][1]])

    def oscillator_strength(self):
        if self.dev: self.log(307, f'### oscillator_strength Python ###')
        self.oscstr = []
        # new_sim.oscstr.append([0, 0, 0, 0])
            
        i = 0
        Ener = round(self.psi0[i][3]*1000/self.e_charge,6)
        Ener = Ener*self.e_charge*1E-3  #[J]
        m_qw = self.m_InGaAs*(1 + Ener/self.Ew)
        m_qb = self.m_InAlAs*(1  + (Ener-self.Vo)/self.Eb)
        
        m = self.mass(m_qw, m_qb)
        
        # meff = np.arange(0,new_sim.n,1)
        meff = np.zeros((self.n, len(self.psi0)))
        meff[:,0] = m
                
        mass_deri = np.zeros((self.n))
        psi0_deri = np.zeros((self.n))
        
        if self.dev: self.log(307, f'Ener: {Ener} | {Ener*1000/self.e_charge}')
        if self.dev: self.log(307, f'm_qw: {m_qw}')
        if self.dev: self.log(307, f'm_qb: {m_qb}')
        
        for i in range(1, len(self.psi0)):
          # print(f"{i + 1} de {len(self.psi0)}", end="\r")
            if self.dev: self.log(307, f"{i+1} de {len(self.psi0)}")
            Ener = round(self.psi0[i][3]*1000/self.e_charge,6)
            Ener = Ener*self.e_charge*1E-3  #[J]
            m_qw = self.m_InGaAs*(1 + Ener/self.Ew)
            m_qb = self.m_InAlAs*(1  + (Ener-self.Vo)/self.Eb)
            if self.dev: self.log(307, f'**Ener: {Ener} | {Ener*1000/self.e_charge}')
            if self.dev: self.log(307, f'**m_qw: {m_qw}')
            if self.dev: self.log(307, f'**m_qb: {m_qb}')
            
            m = self.mass(m_qw, m_qb)
            meff[:,i] = m
            
            # calculando a primeira derivada da funcao de onda do estado excitado e da massa efetiva 
            
            # Derivada à direita no ponto i = 1
            mass_deri[0] = (meff[1,i] - meff[0,i])/self.dx
            psi0_deri[0] = (self.psi0[i][1][1] - self.psi0[i][1][0])/self.dx
            
            #! Derivada à esquerda no ponto i = n
            mass_deri[-1] = (meff[-1,0] - meff[-2,1])/self.dx
            psi0_deri[-1] = (self.psi0[i][1][-1] - self.psi0[i][1][-2])/self.dx
            
            #! Derivada central no ponto i
            for k in range(1, self.n-1):
                mass_deri[k] = (meff[k+1,i] - meff[k-1,i])/(2*self.dx)
                # mass_deri[k] = (meff[i,k+1] - meff[i,k-1])/(2*new_sim.dx)
                psi0_deri[k] = (self.psi0[i][1][k+1] - self.psi0[i][1][k-1])/(2*self.dx)
            
            #!-----------------------------------------------------------------------------------------------------
            #! Calculo da forca do oscilador considerando os efeitos da nao parabolicidade 
            psi_ = np.conjugate(self.psi0[0][1])

            sub_prod_1 = ((psi_)*self.psi0[i][1]*mass_deri[:]/meff[:,0]/meff[:,0])
            sub_prod_2 = (psi_* psi0_deri[:]/ meff[:,0])
            sub_prod_3 = (psi_ * psi0_deri[:]/ meff[:,i])
            # prod_ = - ((psi_)*new_sim.psi0[0][1][i]*mass_deri[:]/meff[:,0]/meff[:,0]) + (psi_* psi0_deri[:]/ meff[:,0]) + (psi_ * psi0_deri[:]/ meff[:,i])
            prod_ = - sub_prod_1 + sub_prod_2 + sub_prod_3

            soma = np.trapz(prod_, self.x)
            difE = (self.Energy[i][1] - self.Energy[0][1])

            
            f_ = m_qw*self.hbar*self.hbar*abs(soma)**2/2/difE
            if self.dev: self.log(307, f'**f: {f_}')
            self.oscstr.append([f_, difE, self.Energy[i][1], self.Energy[0][1]])
      # print("fim oscillator_strength")

    def pc(self):
      # print("starting pc")
        if self.dev: self.log(308, f'### pc Python ###')
        self.v[0] = self.Vo
        self.v[-1] = self.Vo
        # E_initial = 473.3 #(meV) 
        # E_final   = 473.5 #(meV) 
        # E_initial = 309.0 #(meV) 
        # E_final   = 309.4 #(meV) 
        # E_initial = 300.0 #(meV) 
        # E_final   = 350.0 #(meV) 
        # self.ppc_E_final, self.ppc_E_initial = E_final, E_initial

        if self.simulation_type == "develop": prints = True
        E0 = self.Energy[0][1]
        psi_0 = self.psi0[0][1]
        psi_0_p1 = np.roll(psi_0, -1)

        self.E0 = round(self.psi0[0][3]*1000/self.e_charge,6)
        
        if self.ppc_E_initial == "auto":
            self.ppc_E_initial = int((self.barreira*1000) - self.E0 - 10)
            # self.ppc_E_initial = int((self.barreira*1000) - self.E0 - 5)
            # self.ppc_E_initial = int((self.barreira*1000) - self.E0 - 1)

        if self.ppc_E_final == "auto":
            self.ppc_E_final = int((self.barreira*1000) - self.E0 + 305)
        elif isinstance(self.ppc_E_final, str) and self.ppc_E_final[0] == "+":
            self.ppc_E_final = int((self.barreira*1000) - self.E0 + int(self.ppc_E_final[1:]))

        En = int(round((self.ppc_E_final - self.ppc_E_initial)/self.ppc_dE))
        
        v_p1 = np.roll(self.v, -1)
        x_p1 = np.roll(self.x, -1)

        ic = 0.0 + 1.0j
        self.Je = []
        self.Jd = []
        self.E_pc = []

        
        Ener_ = np.arange(self.ppc_E_initial, self.ppc_E_final, self.ppc_dE)
        Ener_ = Ener_*self.e_charge*1E-3

        m_qw = self.m_InGaAs*(1 + Ener_/self.Ew)
        m_qb = self.m_InAlAs*(1  + (E0 + Ener_-self.Vo)/self.Eb)
        

        for jn in range(En-1):
          # print(f"{jn + 1} de {En-1}", end="\r")
            if self.dev: self.log(308, f"jn: {jn+1} de {En-1} || Ener: {round(Ener_[jn+1]/(self.e_charge*1E-3), 2)}")
            # if self.dev: self.log_ETM(f"m_qw: {m_qw[jn]}")
            # if self.dev: self.log_ETM(f"m_qb: {m_qb[jn]}")
            m_ = self.mass(m_qw[jn+1], m_qb[jn+1])
            m_p1 = np.roll(m_, -1)

            T0 = np.array([[1+0j, 0+0j], [0+0j, 1+0j]])
            F  = np.array([0+0j, 0+0j])

            aux1_0 = ((2*m_)*(Ener_[jn+1]+E0-self.v))/(self.hbar2)
            ki_ = sm.sqrt(aux1_0)

            aux1_p1 = ((2*m_p1)*(Ener_[jn+1]+E0-v_p1))/(self.hbar2)
            ki_p1 = sm.sqrt(aux1_p1)
            
            pi_n = np.exp(-ic*ki_p1*self.x)
            pi_p = np.exp(+ic*ki_p1*self.x)

            pf_p = np.exp(+ic*ki_*self.x)
            pf_n = np.exp(-ic*ki_*self.x)

            beta_ = (ki_*m_p1)/(ki_p1*m_)
            Dn_p = 0.5*(1 + beta_)
            Dn_n = 0.5*(1 - beta_)
            cte_ = (m_*self.Fd)/(2*ic*self.hbar*self.hbar*ki_)

            beta_plus_  = -cte_*(np.exp(+ic*ki_*self.x)*self.x*psi_0-np.exp(+ic*ki_*x_p1)*x_p1*psi_0_p1)*self.dx
            beta_minus_ = +cte_*(np.exp(-ic*ki_*self.x)*self.x*psi_0-np.exp(-ic*ki_*x_p1)*x_p1*psi_0_p1)*self.dx

            ##################################################################
            P_inicial = np.empty((self.n,2,2), dtype=np.complex_)
            # P_inicial[:] = np.nan
            P_inicial[:,0,0] = pi_n
            P_inicial[:,0,1] = 0+0j
            P_inicial[:,1,0] = 0+0j
            P_inicial[:,1,1] = pi_p


            P_final = np.empty((self.n,2,2), dtype=np.complex_)
            # P_final[:] = np.nan
            P_final[:,0,0] = pf_p
            P_final[:,0,1] = 0+0j
            P_final[:,1,0] = 0+0j
            P_final[:,1,1] = pf_n

            Dn = np.empty((self.n,2,2), dtype=np.complex_)
            # Dn[:] = np.nan
            Dn[:,0,0] = Dn_p
            Dn[:,0,1] = Dn_n
            Dn[:,1,0] = Dn_n
            Dn[:,1,1] = Dn_p

            ##################################################################
            
            for i in range(self.n-2, -1, -1):
                # if self.dev: self.log(308, f"*int__i: {i+1}")
                
                # if self.dev: self.log(308, f"ic: {dev_complex(ic)}")
                # if self.dev: self.log(308, f"ki[i+1]: {dev_complex(ki_p1[i])}")
                # if self.dev: self.log(308, f"x[i]: {self.x[i]}")
                # if self.dev: self.log(308, f"prod: {dev_complex(-ic*ki_p1[i]*self.x[i])}")
                # if self.dev: self.log(308, f"exp: {dev_complex(cmath.exp(-ic*ki_p1[i]*self.x[i]))}")
                # if self.dev: self.log(308, f"pi_n: {dev_complex(pi_n[i])}")
                
                # pi_n = np.exp(-ic*ki_p1*self.x)
                
                # if self.dev: self.log(308, f"aux1_0: {dev_complex(aux1_0[i])}")
                # if self.dev: self.log(308, f"aux1_p1: {dev_complex(aux1_p1[i])}")
                
                # if self.dev: self.log(308, f"* beta: {dev_complex(beta_[i])}")
                
                
                # if self.dev: self.log_matrix22(308, T0, 'T0')
                # if self.dev: self.log_matrix22(308, P_inicial[i], 'P_inicial')
                # if self.dev: self.log_matrix22(308, P_final[i], 'P_final')
                # if self.dev: self.log_matrix22(308, Dn[i], 'Dn')
            
                
                AUX = np_multiply(P_inicial[i],Dn[i])
                Ti = np_multiply(AUX,P_final[i])
                T = np_multiply(T0,Ti)
                T0 = T.copy()
                
                #	----------------------------------------------------
                #	Calculo da interacao eletron-foton na interface i
                #   ----------------------------------------------------

                F[0] = F[0] + (T[0][0]*beta_minus_[i] + T[0][1]*beta_plus_[i])
                F[1] = F[1] + (T[1][0]*beta_minus_[i] + T[1][1]*beta_plus_[i])

                #	------------------------------------------------------------
                #	Calculo da fotocorrente em funcao da energia do foton (E-E0)
                #   ------------------------------------------------------------
            
            ctecur_left  = (self.hbar*ki_[0])/m_[0]
            ctecur_right = (self.hbar*ki_[self.n-1])/m_[self.n-1]

            self.Je.append(ctecur_left*self.Fd*((-F[1]/T[1][1])*np.conjugate(-F[1]/T[1][1])))
            self.Jd.append(ctecur_right*self.Fd*(F[0]-((F[1]*T[0][1])/T[1][1]))*np.conjugate((F[0]-(F[1]*T[0][1]/T[1][1]))))
            # if self.dev: self.log(308, f"**je: {self.Je[-1]}")
            # if self.dev: self.log(308, f"**jd: {self.Jd[-1]}")
            _pc = np.real(self.Jd[-1] - self.Je[-1])
            if self.dev: self.log(308, f"**pc: {_pc}")
            self.E_pc.append(round(Ener_[jn+1]*1000/self.e_charge, 6))

            # if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("T[0][1]: ", T[0][1]))
            # if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("T[1][1]: ", T[1][1]))
            # if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("F[0]: ", F[0]))
            # if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("F[1]: ", F[1]))

            # if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("je: ", self.Je[-1]))
            # if self.dev: self.log_ETM(f"je: {self.Je[-1]}")
            # if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("jd: ", self.Jd[-1]))
            # if self.dev: self.log_ETM(f"jd: {self.Jd[-1]}")

            # if self.dev_mode: self.list_logs_calculation_pc.append("pc: " + str(np.real(np.array(self.Jd[-1]) - np.array(self.Je[-1]))))

    def develop_indice_to_print(self, i):
        to_print = False
        if i in [1045, 1044, 1043, 1042]:
            to_print = True
        to_print = True
        to_print = False
        return to_print
        
    def pc_org(self):
        prints = False
        if self.simulation_type == "develop": prints = True
        if prints: print("develop mode")

        self.list_of_logs.append("self.n " + str(self.n))
        self.Fd
        E0 = self.Energy[0][1]
        psi_0 = self.psi0[0][1]
        E_initial = 475.0 #(meV) 
        E_final   = 500.0 #(meV) 
        E_final   = 495.0 #(meV) 

        E_initial = 475.2 #(meV) 
        E_final   = 475.4 #(meV) 


        self.pc_e0 = E_initial
        self.pc_e1 = E_final
        
        dE        = 0.1    #(meV)
        En        = int(round((E_final - E_initial)/dE))
        # En        = 1
        ic = 0.0 + 1.0j
        self.Je = []
        self.Jd = []
        self.E_pc = []

        for jn in range(En):
            self.list_of_logs.append(str(jn) + " --------------------------------------")
            if prints: print("inicia com", jn, "de", En)
            Ener= (E_initial + jn*dE) #(meV)
            if self.dev_mode: self.list_logs_calculation_pc.append("---------------------------------------------")
            if self.dev_mode: self.list_logs_calculation_pc.append("Ener: " + str(Ener))
            Ener= Ener*self.e_charge*1E-3  #[J]
            
            
            # non-parabolicity 	  
            m_qw = self.m_InGaAs*(1 + Ener/self.Ew)
            if self.dev_mode: self.list_logs_calculation_pc.append("m_qw: " + str(m_qw))
            m_qb = self.m_InAlAs*(1  + (E0+ Ener-self.Vo)/self.Eb)
            if self.dev_mode: self.list_logs_calculation_pc.append("m_qb: " + str(m_qb))
            m_ = self.mass(m_qw, m_qb)

            if self.dev_mode: self.list_logs_calculation_pc.append("--- inicia massa ---")
            # for index_m, sub_m in enumerate(m_):
                # if self.dev_mode: self.list_logs_calculation_pc.append("m[{}]: {}".format(index_m, sub_m))
            if self.dev_mode: self.list_logs_calculation_pc.append("--- termina massa ---")

            T0 = np.array([[1+0j, 0+0j], [0+0j, 1+0j]])
            F  = np.array([0+0j, 0+0j])
            ki = [None]*self.n

            for i in range(self.n-2, -1, -1):
            # for i in range(n-2, n-6, -1):
                if self.dev_mode: self.list_logs_calculation_pc.append("i: "+ str(i))
                aux1 = ((2*m_[i])*(Ener+E0-self.v[i]))/self.hbar/self.hbar
                # if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("aux1_1: ", aux1))
                ki[i] = sm.sqrt(aux1)
                # if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("ki[i]: ", ki[i]))

                aux1 = (2*m_[i+1]*(Ener+E0-self.v[i+1]))/self.hbar/self.hbar
                ki[i+1] = sm.sqrt(aux1)

                #Matriz propagacao para i=1
                Pi11 = np.exp(-ic*ki[i+1]*(self.x[i]))
                Pi12 = 0+0j
                Pi21 = 0+0j
                Pi22 = np.exp(+ic*ki[i+1]*(self.x[i]))
                P_inicial = np.array([[Pi11, Pi12], [Pi21, Pi22]])
                
                
                Pf11 = np.exp(+ic*ki[i]*(self.x[i]))
                Pf12 = 0+0j
                Pf21 = 0+0j
                Pf22 = np.exp(-ic*ki[i]*(self.x[i]))
                P_final = np.array([[Pf11, Pf12], [Pf21, Pf22]])
            
                #Matriz descontinuidade para i=1 
                if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>ki[i]: ", ki[i]))
                if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>m_[i+1]: ", m_[i+1]))
                if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>ki[i+1]: ", ki[i+1]))
                if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>m_[i]: ", m_[i]))
                beta = (ki[i]*m_[i+1])/(ki[i+1]*m_[i])
                Dn11 = 0.5*(1 + beta)
                Dn12 = 0.5*(1 - beta)
                Dn21 = 0.5*(1 - beta)
                Dn22 = 0.5*(1 + beta)
                Dn = np.array([[Dn11, Dn12], [Dn21,  Dn22]])
                # if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>-ic: ", -ic))
                if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>beta: ", beta))
                if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>Dn11: ", Dn11))
                if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>1+beta: ", 1 + beta))
                if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>Dn12: ", Dn12))
                if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>1-beta: ", 1 - beta))
                # if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>P_inicial[0][0]: ", P_inicial[0][0]))
                # if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>P_inicial[0][1]: ", P_inicial[0][1]))
                # if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>P_inicial[1][0]: ", P_inicial[1][0]))
                # if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>P_inicial[1][1]: ", P_inicial[1][1]))
                
                # if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>Dn[0][0]: ", Dn[0][0]))
                # if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>Dn[0][1]: ", Dn[0][1]))
                # if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>Dn[1][0]: ", Dn[1][0]))
                # if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>Dn[1][1]: ", Dn[1][1]))

                
                if self.develop_indice_to_print(i):
                    Pi11 = np.exp(-ic*ki[i+1]*(self.x[i]))
                    self.list_of_logs.append("   " + str(i) + " -ic " + str(-ic))
                    self.list_of_logs.append("   " + str(i) + " ki[i+1] " + str(ki[i+1]))
                    self.list_of_logs.append("   " + str(i) + " self.x[i] " + str(self.x[i]))
                    self.list_of_logs.append("   " + str(i) + " -ic*ki[i+1] " + str(-ic*ki[i+1]))
                    self.list_of_logs.append("   " + str(i) + " -ic*ki[i+1]*(self.x[i]) " + str(-ic*ki[i+1]*(self.x[i])))
                    self.list_of_logs.append("   " + str(i) + " np.exp(-ic*ki[i+1]*(self.x[i])) " + str(np.exp(-ic*ki[i+1]*(self.x[i]))))

                    self.list_of_logs.append("   " + str(i) + " ---------------------------------------- ")
                    self.list_of_logs.append("   " + str(i) + " P_inicial [0][0] " + str(P_inicial[0][0]))
                    self.list_of_logs.append("   " + str(i) + " P_inicial [0][1] " + str(P_inicial[0][1]))
                    self.list_of_logs.append("   " + str(i) + " P_inicial [1][0] " + str(P_inicial[1][0]))
                    self.list_of_logs.append("   " + str(i) + " P_inicial [1][1] " + str(P_inicial[1][1]))
                    self.list_of_logs.append("   " + str(i) + " P_final [0][0] " + str(P_final[0][0]))
                    self.list_of_logs.append("   " + str(i) + " P_final [1][1] " + str(P_final[1][1]))
                    self.list_of_logs.append("   " + str(i) + " Dn [0][0] " + str(Dn[0][0]))
                    self.list_of_logs.append("   " + str(i) + " Dn [0][1] " + str(Dn[0][1]))
                    self.list_of_logs.append("   " + str(i) + " Dn [1][0] " + str(Dn[1][0]))
                    self.list_of_logs.append("   " + str(i) + " Dn [1][1] " + str(Dn[1][1]))
                
                
                
                AUX = np_multiply(P_inicial,Dn)
                Ti = np_multiply(AUX,P_final)
                T = np_multiply(T0,Ti)
                T0 = T.copy()
                # if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>AUX[0][0]: ", AUX[0][0]))
                # if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>AUX[0][1]: ", AUX[0][1]))
                # if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>AUX[1][0]: ", AUX[1][0]))
                # if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("<>AUX[1][1]: ", AUX[1][1]))

        #		----------------------------------------------------
        #		Calculo da interacao eletron-foton na interface i
        #       ----------------------------------------------------
                cte = (m_[i]*self.Fd)/(2*ic*self.hbar*self.hbar*ki[i])



                # list_of_logs.append("psi " + str(psi0_pc[i]))
                beta_plus  = -cte*((np.exp(+ic*ki[i]*self.x[i])*self.x[i]*psi_0[i])-(np.exp(+ic*ki[i]*self.x[i+1])*self.x[i+1]*psi_0[i+1]))*self.dx
                beta_minus = +cte*(np.exp(-ic*ki[i]*self.x[i])*self.x[i]*psi_0[i]-np.exp(-ic*ki[i]*self.x[i+1])*self.x[i+1]*psi_0[i+1])*self.dx

                
                F[0] = F[0] + (T[0][0]*beta_minus + T[0][1]*beta_plus)
                F[1] = F[1] + (T[1][0]*beta_minus + T[1][1]*beta_plus)

                

            #	------------------------------------------------------------
            #	Calculo da fotocorrente em funcao da energia do foton (E-E0)
            #   ------------------------------------------------------------
            
            ctecur_left  = (self.hbar*ki[0])/m_[0]
            ctecur_right = (self.hbar*ki[self.n-1])/m_[self.n-1]

            self.Je.append(ctecur_left*self.Fd*((-F[1]/T[1][1])*np.conjugate(-F[1]/T[1][1])))
            self.Jd.append(ctecur_right*self.Fd*(F[0]-((F[1]*T[0][1])/T[1][1]))*np.conjugate((F[0]-(F[1]*T[0][1]/T[1][1]))))
            self.E_pc.append(round(Ener*1000/self.e_charge, 6))
            
            if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("T[0][1]: ", T[0][1]))
            if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("T[1][1]: ", T[1][1]))
            if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("F[0]: ", F[0]))
            if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("F[1]: ", F[1]))

            if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("je: ", self.Je[-1]))
            if self.dev_mode: self.list_logs_calculation_pc.append(print_complex("jd: ", self.Jd[-1]))

            if self.dev_mode: self.list_logs_calculation_pc.append("pc: " + str(np.real(np.array(self.Jd[-1]) - np.array(self.Je[-1]))))

            valll = 1/sm.sqrt((np.real(T[0][1])**2 + np.imag(T[0][1])))

            if self.dev_mode: self.list_logs_transmission.append(str(Ener/(self.e_charge*1E-3)) + " " + str(valll))
            

            self.list_of_logs.append(" self.Je[-1] " + str(self.Je[-1]))

    def save_pkl(self):
        afile = open(self.pkl_path, 'wb')
        pickle.dump(self, afile)
        afile.close()

    def fitness_function(self, prints = False, optimization_parameter = 'oscpc', target=[300, 1000]):
        self.fitness = 0
        
        self.max_oscstr_under_barrier
        self.max_e_oscstr_under_barrier
        self.max_oscstr_above_barrier
        self.max_e_oscstr_above_barrier
        
        self.max_photocurrent
        self.max_e_photocurrent
        self.min_photocurrent
        self.min_e_photocurrent
        self.max_abs_photocurrent
        self.max_e_abs_photocurrent
        self.min_abs_photocurrent
        self.min_e_abs_photocurrent
        
        def valid_energy(value, target):
            limit_min = target[0] - target[1]
            limit_max = target[0] + target[1]
            if value > limit_max or value < limit_min:
                return False
            return True

        #new fitness version, only PC
        self.fitness = 0
        if optimization_parameter == "osc":
            self.fitness = self.max_oscstr_above_barrier
            if not valid_energy(self.max_e_oscstr_above_barrier, target):
                self.fitness = 0
        elif optimization_parameter == "pc":
            if self.max_abs_photocurrent > 0:
                # self.fitness = ((300/(-(math.log10(abs(self.max_abs_photocurrent))))+0))-20
                self.fitness = (math.log10(abs(self.max_abs_photocurrent)))+30
                if not valid_energy(self.max_e_abs_photocurrent, target):
                    self.fitness = 0
        elif optimization_parameter == "oscpc":
            if self.max_abs_photocurrent > 0:
                fitness_pc = (math.log10(abs(self.max_abs_photocurrent)))+30
            else:
                fitness_pc = 0
            fitness_oscstr = self.max_oscstr_above_barrier
            if prints:
                print(f"fitness_pc: {fitness_pc}")
                print(f"fitness_oscstr: {fitness_oscstr}")
            
            self.fitness = fitness_pc + (fitness_oscstr * 20)
            if prints:
                print(f"fitness: {self.fitness}")

            if not valid_energy(self.max_e_abs_photocurrent, target):
                self.fitness = 0
            if not valid_energy(self.max_e_oscstr_above_barrier, target):
                self.fitness = 0


        if prints:
            print("max1 PC: {:.2e} - E: {:02.1f} (meV)".format(self.max_abs_photocurrent, self.max_e_abs_photocurrent))
            print("max2 PC: {:.2e} - E: {:02.1f} (meV)".format(self.min_abs_photocurrent, self.min_e_abs_photocurrent))
            print("OscStr: {:.2f} - E: {:02.1f} (meV)".format(self.max_oscstr_above_barrier, self.max_e_oscstr_above_barrier))
            print("E0: {:02.1f} (meV)".format(self.E0))
            print("aptidao do individuo: ", round(self.fitness, 3))

    def process_results(self):
        self.x_nm = self.x*1e+9
        self.E0 = round(self.psi0[0][3]*1000/self.e_charge,6)
        
        # structure
        self.result_potential = self.v*1000/self.e_charge

        # for wave functions
        self.result_wavefunction = []
        for i, [ne, psi, vec_probability, EEEE] in enumerate(self.psi0):
            wave = np.array(vec_probability)
            # até -2 porque tem um pico no final
            wave = (wave-min(wave[:-2]))/max(wave[:-2])
            # wave = (wave-min(wave[:]))/max(wave[:])
            Energy = round(EEEE*1000/self.e_charge,6)
            if i == 0:
                self.result_wavefunction.append([Energy, (wave*50)+Energy])
            elif Energy > self.barreira*1000:
                self.result_wavefunction.append([Energy, (wave*20)+Energy])
            else:
                self.result_wavefunction.append([Energy, (wave*25)+Energy])

        

        # Oscillator Strength
        self.max_oscstr_under_barrier = 0
        self.max_e_oscstr_under_barrier = 0
        self.max_oscstr_above_barrier = 0
        self.max_e_oscstr_above_barrier = 0
        self.max_oscstr = 0
        self.max_e_oscstr = 0

        self.max_oscstr_under_index = 0
        self.max_oscstr_above_index = 0
        self.result_oscstr = []
        for i, [oscstr, E, Ef, Ei] in enumerate(self.oscstr):
            self.result_oscstr.append([oscstr, E*1000/self.e_charge, Ef*1000/self.e_charge, Ei*1000/self.e_charge])
            if Ef/self.e_charge > self.barreira:
                if oscstr > self.max_oscstr_above_barrier:
                    self.max_oscstr_above_barrier = oscstr
                    self.max_oscstr_above_index = i+1
                    self.max_e_oscstr_above_barrier = E*1000/self.e_charge
                    
            else:
                if oscstr > self.max_oscstr_under_barrier:
                    self.max_oscstr_under_barrier = oscstr
                    self.max_oscstr_under_index = i+1
                    self.max_e_oscstr_under_barrier = E*1000/self.e_charge

        if self.max_oscstr_under_barrier > self.max_oscstr_above_barrier:
            self.max_oscstr = self.max_oscstr_under_barrier
            self.max_e_oscstr = self.max_e_oscstr_under_barrier
        else:
            self.max_oscstr = self.max_oscstr_above_barrier
            self.max_e_oscstr = self.max_e_oscstr_above_barrier



        # potocurrent
        self.result_pc = np.real(np.array(self.Jd) - np.array(self.Je))

        # to replace nan by 0
        for i in range(len(self.result_pc)):
            if np.isnan(self.result_pc[i]):
                self.result_pc[i] = 0

        # max PC
        self.max_photocurrent = np.amax(self.result_pc)
        self.max_e_photocurrent = self.E_pc[np.argmax(self.result_pc)]
        self.min_photocurrent = np.amin(self.result_pc)
        self.min_e_photocurrent = self.E_pc[np.argmin(self.result_pc)]

        if abs(self.max_photocurrent) > abs(self.min_photocurrent):
            self.max_abs_photocurrent = abs(self.max_photocurrent)
            self.max_e_abs_photocurrent = self.max_e_photocurrent
            self.min_abs_photocurrent = abs(self.min_photocurrent)
            self.min_e_abs_photocurrent = self.min_e_photocurrent
        else:
            self.max_abs_photocurrent = abs(self.min_photocurrent)
            self.max_e_abs_photocurrent = self.min_e_photocurrent
            self.min_abs_photocurrent = abs(self.max_photocurrent)
            self.min_e_abs_photocurrent = self.max_e_photocurrent
        return None

    def process_results_old(self):
        self.x_nm = self.x*1e+9
        self.E0 = round(self.psi0[0][3]*1000/self.e_charge,6)
        
        # structure
        self.result_potential = self.v*1000/self.e_charge

        # for wave functions
        self.result_wavefunction = []
        for i, [ne, psi, vec_probability, EEEE] in enumerate(self.psi0):
            Energy = round(EEEE/self.e_charge,6)
            if i == 0:
                self.result_wavefunction.append([Energy*1000, (vec_probability*1000*1)+Energy*1000])
            elif Energy > self.barreira:
                self.result_wavefunction.append([Energy*1000, (vec_probability*1000/4)+Energy*1000])
            else:
                self.result_wavefunction.append([Energy*1000, (vec_probability*1000/2)+Energy*1000])

        

        # Oscillator Strength
        self.max_oscstr_under_barrier = 0
        self.max_e_oscstr_under_barrier = 0
        self.max_oscstr_above_barrier = 0
        self.max_e_oscstr_above_barrier = 0
        self.max_oscstr = 0
        self.max_e_oscstr = 0

        self.max_oscstr_under_index = 0
        self.max_oscstr_above_index = 0
        self.result_oscstr = []
        for i, [oscstr, E, Ef, Ei] in enumerate(self.oscstr):
            self.result_oscstr.append([oscstr, E*1000/self.e_charge, Ef*1000/self.e_charge, Ei*1000/self.e_charge])
            if Ef/self.e_charge > self.barreira:
                if oscstr > self.max_oscstr_above_barrier:
                    self.max_oscstr_above_barrier = oscstr
                    self.max_oscstr_above_index = i+1
                    self.max_e_oscstr_above_barrier = E*1000/self.e_charge
                    
            else:
                if oscstr > self.max_oscstr_under_barrier:
                    self.max_oscstr_under_barrier = oscstr
                    self.max_oscstr_under_index = i+1
                    self.max_e_oscstr_under_barrier = E*1000/self.e_charge

        if self.max_oscstr_under_barrier > self.max_oscstr_above_barrier:
            self.max_oscstr = self.max_oscstr_under_barrier
            self.max_e_oscstr = self.max_e_oscstr_under_barrier
        else:
            self.max_oscstr = self.max_oscstr_above_barrier
            self.max_e_oscstr = self.max_e_oscstr_above_barrier



        # potocurrent
        self.result_pc = np.real(np.array(self.Jd) - np.array(self.Je))

        # to replace nan by 0
        for i in range(len(self.result_pc)):
            if np.isnan(self.result_pc[i]):
                self.result_pc[i] = 0

        # max PC
        self.max_photocurrent = np.amax(self.result_pc)
        self.max_e_photocurrent = self.E_pc[np.argmax(self.result_pc)]
        self.min_photocurrent = np.amin(self.result_pc)
        self.min_e_photocurrent = self.E_pc[np.argmin(self.result_pc)]

        if abs(self.max_photocurrent) > abs(self.min_photocurrent):
            self.max_abs_photocurrent = abs(self.max_photocurrent)
            self.max_e_abs_photocurrent = self.max_e_photocurrent
            self.min_abs_photocurrent = abs(self.min_photocurrent)
            self.min_e_abs_photocurrent = self.min_e_photocurrent
        else:
            self.max_abs_photocurrent = abs(self.min_photocurrent)
            self.max_e_abs_photocurrent = self.min_e_photocurrent
            self.min_abs_photocurrent = abs(self.max_photocurrent)
            self.min_e_abs_photocurrent = self.max_e_photocurrent
        return None


    def finish_dictionary(self, individual):
        structure = set_structure_values(individual)
        id = set_id(structure)
        lw, lqw, lqb, c, rw, rqw, rqb = self.structure

        self.newdata = {"id": id,
                    "structure": self.structure,
                    "lw": lw,
                    "lqw": lqw,
                    "lqb": lqb,
                    "c": c,
                    "rw": rw,
                    "rqw": rqw,
                    "rqb": rqb,
                    "fit": self.fitness,
                    "E0": self.E0,
                    "pc1": self.max_abs_photocurrent,
                    "pc1e": self.max_e_abs_photocurrent,
                    "pc2": self.min_abs_photocurrent,
                    "pc2e": self.min_e_abs_photocurrent,
                    "oscstr": self.max_oscstr,
                    "oscstre": self.max_e_oscstr,
                    "oscstr_under": self.max_oscstr_under_barrier,
                    "oscstre_under": self.max_e_oscstr_under_barrier,
                    "oscstr_above": self.max_oscstr_above_barrier,
                    "oscstre_above": self.max_e_oscstr_above_barrier
                }
        

        self.result_dictionary = {"individual" : individual,
                            "fitness" : self.fitness,
                            "structure" : self.structure,
                            "E0" : self.E0,
                            "pc1" : self.max_abs_photocurrent,
                            "pc1e" : self.max_e_abs_photocurrent,
                            "pc2" : self.min_abs_photocurrent,
                            "pc2e" : self.min_e_abs_photocurrent,
                            "oscstr_above" : self.max_oscstr_above_barrier,
                            "oscstre_above" : self.max_e_oscstr_above_barrier,
                            "oscstr_under" : self.max_oscstr_under_barrier,
                            "oscstre_under" : self.max_e_oscstr_under_barrier,
                            "oscstr" : self.max_oscstr,
                            "oscstre" : self.max_e_oscstr}
        
    def plot_structure_os(self):
        fig, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [16, 4]})
        plt.subplots_adjust(wspace = 0.0, top = 0.85)
        # plt.gcf().text(0.01, 0.01, "Fortran em Python")
        lim_E_min, lim_E_max = -50, 700
        # lim_E_min, lim_E_max = 492, 493
        # lim_E_min, lim_E_max = 490, 491
        # lim_E_min, lim_E_max = 480, 501

        # for pc, ele in zip(self.result_pc, self.E_pc):
        #     print(ele, pc)

        ax1 = self.plot_wave_function(ax1, lim_E_min, lim_E_max)
        ax2 = self.plot_oscillator_strength(ax2, lim_E_min, lim_E_max)

        for ax in [ax1, ax2]:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(13)


        
        
        # set titles
        ax1.set_title("(a)", x=0.0, y=1.1, size=9)
        ax2.set_title('(b)', x=0.0, y=1.1, size=9)

        # set main title
        lw, lqw, lqb, de, rw, rqw, rqb = self.structure
        title_es = "Left || {:02.0f} - QW: {:04.1f} nm - QB: {:04.1f} nm".format(lw, lqw, lqb) + \
            "\nRight || {:02.0f} - QW: {:04.1f} nm - QB: {:04.1f} nm".format(rw, rqw, rqb) + \
            "\nMain QW: {:04.1f} nm\n".format(de) + \
            "Oscillator Strength: {:.2f} - E: {:02.1f} (meV)".format(self.max_oscstr, self.max_e_oscstr)
        fig.suptitle(title_es, x=0.42, fontsize=8)
        fig.suptitle(title_es, x=0.42, fontsize=10)

        # save image
        if self.simulation_type == 'os_test':
            plt.savefig('temp_files/0001.png', dpi=150)
        elif self.simulation_type.startswith('os_test_'):
            name = self.simulation_type.replace('os_test_', '')
            plt.savefig('temp_files/{}.png'.format(name), dpi=150)
        
        # clear plots
        plt.cla()               # clears an axis, i.e. the currently active axis in the current figure. It leaves the other axes untouched.
        plt.clf()               # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
        plt.close()             # closes a window, which will be the current window, if not specified otherwise. 
        plt.close('all') 

        del(fig)
        del(ax1)
        del(ax2)

    def plot_structure(self):
        fig, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [16, 4]})
        plt.subplots_adjust(wspace = 0.0, top = 0.8)
        # plt.gcf().text(0.01, 0.01, "Fortran em Python")
        lim_E_min, lim_E_max = -50, 700
        # lim_E_min, lim_E_max = 492, 493
        # lim_E_min, lim_E_max = 490, 491
        # lim_E_min, lim_E_max = 480, 501

        # for pc, ele in zip(self.result_pc[2999:4000], self.E_pc[2999:4000]):
        #     print(ele, pc)

        ax1 = self.plot_wave_function(ax1, lim_E_min, lim_E_max)
        ax2 = self.plot_oscillator_strength(ax2, lim_E_min, lim_E_max)
        ax3 = ax2.twiny()
        # ax3 = self.plot_photocurrent_w(ax3, lim_E_min, lim_E_max)
        ax3 = self.plot_photocurrent(ax3, lim_E_min, lim_E_max)

        
        # ax3.set_xscale('log')
        # ax3.set(xlim=[-1e43, 1e44])
        # ax3.set(xlim=[-1e43, 1e43])

        # set titles
        ax1.set_title("(a)", x=0.0, y=1.1, size=9)
        ax2.set_title('(b)', x=0.0, y=1.1, size=9)

        # set main title
        lw, lqw, lqb, de, rw, rqw, rqb = self.structure
        title_es = "Left || {:02.0f} - QW: {:04.1f} nm - QB: {:04.1f} nm".format(lw, lqw, lqb) + \
            "\nRight || {:02.0f} - QW: {:04.1f} nm - QB: {:04.1f} nm".format(rw, rqw, rqb) + \
            "\nMain QW: {:04.1f} nm\n".format(de) + \
            "Photocurrent: {:.2e} || PC energy: {:02.1f} (meV)\n".format(self.max_abs_photocurrent, self.max_e_abs_photocurrent) + \
            "Oscillator Strength: {:.2f} || E: {:02.1f} (meV)".format(self.max_oscstr_above_barrier, self.max_e_oscstr_above_barrier)
        # "max1 PC: {:.2e} || PC energy: {:02.1f} (meV)\n".format(self.max_abs_photocurrent, self.max_e_abs_photocurrent) + \
        # "max2 PC: {:.2e} || PC energy: {:02.1f} (meV)\n".format(self.min_abs_photocurrent, self.min_e_abs_photocurrent) + \
        fig.suptitle(title_es, x=0.42, fontsize=10)

        # save image
        if "test_qqqq" in self.simulation_type:
            plt.savefig('temp_files/000' + self.simulation_type[-1] + '.png', dpi=150)
        elif self.simulation_type == "test":
            plt.savefig('temp_files/0001.png', dpi=150)
        elif self.simulation_type.startswith('test_'):
            name = self.simulation_type.replace('test_', '')
            plt.savefig('temp_files/{}.png'.format(name), dpi=150)
        elif self.simulation_type == "image_folder":
            plt.savefig(self.png_path, dpi=150)
        elif self.simulation_type == "image_to_map":
            plt.savefig(self.png_path, dpi=150)
        elif self.simulation_type == "develop":
            plt.savefig('temp_files/dev_python.png', dpi=150)
        elif self.simulation_type.startswith("gif_"):
            image_number = int(self.simulation_type.replace("gif_", ""))
            file_name = self.output_folder + "gifs/img/" + str(image_number).zfill(4) + ".png"
            plt.savefig(file_name, dpi=150)
        elif self.simulation_type == "save":
            pass
        
        self.plot_wf = ax1
        self.plot_os = ax2
        self.plot_pc = ax3
        # clear plots
        plt.cla()               # clears an axis, i.e. the currently active axis in the current figure. It leaves the other axes untouched.
        plt.clf()               # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
        plt.close()             # closes a window, which will be the current window, if not specified otherwise. 
        plt.close('all') 

        del(fig)
        del(ax1)
        del(ax2)
        del(ax3)

    def plot_structure_paper(self, plot='os_pc', options={}):
        plot_options = options['plot_options']
        font_base = 10
        font_axis_number = font_base
        font_axis_text = font_base + 0
        font_subtitle = font_base + 2
        font_title = font_base + 4
        
        
        fig, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [8, 4]})
        plt.subplots_adjust(wspace = 0.0, top = 0.7)

        
        lim_E_min, lim_E_max = -50, 700

        ax1 = self.plot_wave_function(ax1, lim_E_min, lim_E_max)
        if plot_options.lim_x: ax1.set(xlim=plot_options.lim_x)
        
        if plot == 'os':
            ax2 = self.plot_oscillator_strength(ax2, lim_E_min, lim_E_max)
        elif plot == 'pc':
            ax2 = self.plot_oscillator_strength_empty(ax2, lim_E_min, lim_E_max)
            ax3 = ax2.twiny()
            ax3 = self.plot_photocurrent(ax3, lim_E_min, lim_E_max)
            # ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        elif plot == 'os_pc':
            ax2 = self.plot_oscillator_strength(ax2, lim_E_min, lim_E_max)
            ax3 = ax2.twiny()
            ax3 = self.plot_photocurrent(ax3, lim_E_min, lim_E_max)

        if plot == 'os':
            axes_ = [ax1, ax2]
        elif plot == 'pc':
            axes_ = [ax1, ax2, ax3]
        elif plot == 'os_pc':
            axes_ = [ax1, ax2, ax3]
        
        arrow = True
        if "paper_0_base" in self.simulation_type:
            x_arrow = 15
            max_osc_transition_e = self.result_oscstr[self.max_oscstr_above_index-1][1]
            max_osc_e = self.result_oscstr[self.max_oscstr_above_index-1][2]
            style = "Simple, tail_width=0.5, head_width=10, head_length=10"
            kw = dict(arrowstyle=style, color="r")
            arrow = patches.FancyArrowPatch((x_arrow, self.E0), (x_arrow, max_osc_e),
                                connectionstyle="arc3,rad=0.3", **kw)
            
            ax1.add_patch(arrow)

            ax1.text(x_arrow+7, ((max_osc_e + self.E0 + 30)/2), f'{round(max_osc_transition_e,2)} meV',
                verticalalignment='center', horizontalalignment='left',
                color='r', fontsize=font_base-5, rotation=45)
            
        ax2.grid(axis='y', color='0.70')

        for ax in axes_:
            # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.yaxis.offsetText] + ax.get_xticklabels() + ax.get_yticklabels()):
                # item.set_fontsize(16)
            
            for item in ([ax.xaxis.label, ax.yaxis.label]):
                item.set_fontsize(font_axis_text)

            for item in ([ax.title]):
                item.set_fontsize(font_subtitle)

            for item in ([ax.yaxis.offsetText]):
                item.set_fontsize(font_axis_number)

            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(font_axis_number)

            ax.title.set_fontsize(font_title)

        if plot_options.ref_pc or plot_options.ref_oscstr:
            # pc = f"{self.newdata['pc1']:.2e} - {self.newdata['pc1e']:02.1f}"
            # osc_ab = f"{self.newdata['oscstr_above']:02.self.newdata} - {self.newdata['oscstre_above']:02.1f}"
            sim_pc = self.newdata['pc1']
            sim_os = self.newdata['oscstr_above']
            ref_pc = plot_options.ref_pc
            ref_os = plot_options.ref_oscstr
            
            
            
            title_es = f"________________________________________"
            title_es += f"\n|             sim  |    ref   | gain   |"
            if plot_options.ref_oscstr:
                gain_os = (sim_os-ref_os)/ref_os
                title_es += f"\n|OscStr:     {sim_os:04.3f} |    {ref_os:04.3f} | {gain_os:04.3f}X |"
            if plot_options.ref_pc:
                gain_pc = (sim_pc-ref_pc)/ref_pc
                title_es += f"\n|    PC:  {sim_pc:.2e} | {ref_pc:.2e} | {gain_pc:04.3f}X |"
                
            # title_es = f"|        sim |    ref  | gain"
            # if plot_options.ref_oscstr:
            #     gain_os = (sim_os-ref_os)/ref_os
            #     title_es += f"\n|OS:    {sim_os:04.3f}|    {ref_os:04.3f}| {gain_os:04.3f}X"
            # if plot_options.ref_pc:
            #     gain_pc = (sim_pc-ref_pc)/ref_pc
            #     title_es += f"\n|PC: {sim_pc:.2e}| {ref_pc:.2e}| {gain_pc:04.3f}X"
            ax1.set_title(title_es, x=0.01, fontsize=9, family='monospace', horizontalalignment='left')
        
        # set main title
        lw, lqw, lqb, de, rw, rqw, rqb = self.structure
        title_es = "Left || {:02.0f} - QW: {:04.1f} nm - QB: {:04.1f} nm".format(lw, lqw, lqb) + \
            "\nRight || {:02.0f} - QW: {:04.1f} nm - QB: {:04.1f} nm".format(rw, rqw, rqb) + \
            "\nMain QW: {:04.1f} nm\n".format(de) + \
            "Photocurrent: {:.2e} || PC energy: {:02.1f} (meV)\n".format(self.max_abs_photocurrent, self.max_e_abs_photocurrent) + \
            "Oscillator Strength: {:.2f} || E: {:02.1f} (meV)".format(self.max_oscstr_above_barrier, self.max_e_oscstr_above_barrier)
        fig.suptitle(title_es, x=0.40, fontsize=10)

        
        
        # save image
        if self.simulation_type.startswith('paper_test_'):
            name = self.simulation_type.replace('paper_test_', '')
            plt.savefig('temp_files/{}.png'.format(name), dpi=150)
        elif self.simulation_type == "save":
            pass
        
        plt.cla()               # clears an axis, i.e. the currently active axis in the current figure. It leaves the other axes untouched.
        plt.clf()               # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
        plt.close()             # closes a window, which will be the current window, if not specified otherwise. 
        plt.close('all') 

        del(fig)
        
    def plot_structure_paper_final_paper(self, plot='os_pc', options={}):
        plot_options = options['plot_options']
        font_base = 24
        font_axis_number = font_base
        font_axis_text = font_base + 2
        font_subtitle = font_base + 2
        font_title = font_base + 4

        fig, (ax1, ax2) = plt.subplots(1,2,
                                       gridspec_kw={'width_ratios': [8, 5]},
                                       figsize=(12, 6))
        plt.subplots_adjust(wspace = 0.40,
                            top = 0.9, bottom = 0.15,
                            left = 0.1, right = 0.875)
        
        lim_E_min, lim_E_max = -50, 700

        ax1 = self.plot_wave_function(ax1, lim_E_min, lim_E_max)
        if plot_options.lim_x: ax1.set(xlim=plot_options.lim_x)
        
        if plot == 'os':
            if plot_options.lim_x_pc:
                lim_E_min, lim_E_max = plot_options.lim_x_pc
            # print("******")
            # print(lim_E_min, lim_E_max)
            ax2 = self.plot_oscillator_strength_horizontal(ax2, lim_E_min, lim_E_max)
            # ax2.set(xlim=[300, 501])
        elif plot == 'pc':
            if plot_options.lim_x_pc:
                lim_E_min, lim_E_max = plot_options.lim_x_pc
            ax3 = ax2.twinx()
            # ax3 = self.plot_photocurrent_w(ax3, lim_E_min, lim_E_max)
            ax3 = self.plot_photocurrent_horizontal(ax3, lim_E_min, lim_E_max)
            ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        elif plot == 'os_pc':
            if plot_options.lim_x_pc:
                lim_E_min, lim_E_max = plot_options.lim_x_pc
            ax2 = self.plot_oscillator_strength_horizontal(ax2, lim_E_min, lim_E_max)
            ax3 = ax2.twinx()
            
            # Definir un formateador para los ticks del eje x
            def format_func(value, tick_number):
                return f'{value:.2f}'

            # Aplicar el formateador al eje x
            ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
            # ax3 = self.plot_photocurrent_w(ax3, lim_E_min, lim_E_max)
            ax3 = self.plot_photocurrent_horizontal(ax3, lim_E_min, lim_E_max)

            ax2_ylim = list(ax2.get_ylim())
            ax3_ylim = list(ax3.get_ylim())
            # print(f"ax2_ylim: {ax2_ylim}")
            # print(f"ax3_ylim: {ax3_ylim}")

            
            ax2_ylim[0] = (ax3_ylim[0] * ax2_ylim[1]) / ax3_ylim[1]
            ax2.set(ylim=ax2_ylim)
            ax3.set(ylim=ax3_ylim)

        ax1.set_yticks(range(0, 700+1, 100))
        ax1.set_xticks(range(-150, 151, 30))
        if plot_options.lim_x: ax1.set(xlim=plot_options.lim_x)

        ax2.set_xticks(list(range(lim_E_min, lim_E_max+1, 100)))
        ax2.minorticks_on()
        ax2.set_xticks(list(range(lim_E_min+50, lim_E_max+1, 100)), minor=True)

        axe_ticks_y = ax2.get_yticks()
        # print(axe_ticks_y)
        # Eliminar el primer tick si no es igual a 0
        for idx, axe_tick in enumerate(axe_ticks_y):
            if axe_tick < 0:
                ax2.yaxis.get_major_ticks()[idx].set_visible(False)

        # ax2.set_yticks(range(0, 700+1, 100))

        
        ax2.grid(axis='x', color='0.70')
        ax2.grid(axis='x', which="minor", color='0.85')

        ax2.grid(axis='y', color='0.70')

        
        # set titles
        ax1.set_title("(a)", x=0.05, y=1.02, size=font_subtitle)
        ax2.set_title('(b)', x=0.05, y=1.02, size=font_subtitle)

        if plot == 'os':
            axes_ = [ax1, ax2]
        elif plot == 'pc':
            axes_ = [ax1, ax2, ax3]
        elif plot == 'os_pc':
            axes_ = [ax1, ax2, ax3]
        
        arrow = True
        if "paper_0_base" in self.simulation_type:
            x_arrow = 15
            max_osc_transition_e = self.result_oscstr[self.max_oscstr_above_index-1][1]
            max_osc_e = self.result_oscstr[self.max_oscstr_above_index-1][2]
            style = "Simple, tail_width=0.5, head_width=10, head_length=10"
            kw = dict(arrowstyle=style, color="r")
            arrow = patches.FancyArrowPatch((x_arrow, self.E0), (x_arrow, max_osc_e),
                                connectionstyle="arc3,rad=0.3", **kw)
            
            ax1.add_patch(arrow)

            ax1.text(x_arrow+7, ((max_osc_e + self.E0 + 30)/2), f'{round(max_osc_transition_e,2)} meV',
                verticalalignment='center', horizontalalignment='left',
                color='r', fontsize=font_base-5, rotation=45)


        

        for ax in axes_:
            # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.yaxis.offsetText] + ax.get_xticklabels() + ax.get_yticklabels()):
                # item.set_fontsize(16)
            
            for item in ([ax.xaxis.label, ax.yaxis.label]):
                item.set_fontsize(font_axis_text)

            for item in ([ax.title]):
                item.set_fontsize(font_subtitle)

            for item in ([ax.yaxis.offsetText]):
                item.set_fontsize(font_axis_number)

            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(font_axis_number)

            ax.title.set_fontsize(font_title)

        if plot_options.ref_pc or plot_options.ref_oscstr:
            # pc = f"{self.newdata['pc1']:.2e} - {self.newdata['pc1e']:02.1f}"
            # osc_ab = f"{self.newdata['oscstr_above']:02.self.newdata} - {self.newdata['oscstre_above']:02.1f}"
            sim_pc = self.newdata['pc1']
            sim_os = self.newdata['oscstr_above']
            ref_pc = plot_options.ref_pc
            ref_os = plot_options.ref_oscstr
            
            
            
            title_es = f"             sim  |    ref   | gain"
            if plot_options.ref_oscstr:
                gain_os = (sim_os-ref_os)/ref_os
                title_es += f"\nOscStr:     {sim_os:04.3f} |    {ref_os:04.3f} | {gain_os:04.3f}X"
            if plot_options.ref_pc:
                gain_pc = (sim_pc-ref_pc)/ref_pc
                title_es += f"\n    PC:  {sim_pc:.2e} | {ref_pc:.2e} | {gain_pc:04.3f}X"
            fig.suptitle(title_es, x=0.25, fontsize=10, family='monospace', horizontalalignment='left')
        
        
        # save image
        if self.simulation_type.startswith('paper_test_'):
            name = self.simulation_type.replace('paper_test_', '')
            plt.savefig('temp_files/{}.png'.format(name), dpi=150)
        elif self.simulation_type == "save":
            pass
        
        plt.cla()               # clears an axis, i.e. the currently active axis in the current figure. It leaves the other axes untouched.
        plt.clf()               # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
        plt.close()             # closes a window, which will be the current window, if not specified otherwise. 
        plt.close('all') 

        del(fig)


    def plot_structure_paper_comparison(self, plot='os_pc', options={}):
        font_base = 24
        font_axis_number = font_base
        font_axis_text = font_base + 2
        font_subtitle = font_base + 2
        font_title = font_base + 4

        def cauchy_distribution(x, x0, gamma):
            """
            Cauchy distribution (Lorentz distribution) probability density function (PDF).
            
            :param x: The value at which to evaluate the PDF.
            :param x0: Location parameter.
            :param gamma: Scale parameter.
            :return: The PDF value at x.
            """
            return 1.0 / (np.pi * gamma * (1 + ((x - x0) / gamma)**2))
        
        fig, (ax1, ax2) = plt.subplots(1,2,
                                       gridspec_kw={'width_ratios': [4, 4]},
                                       figsize=(12, 6))
        plt.subplots_adjust(wspace = 0.05,
                            top = 0.9, bottom = 0.15,
                            left = 0.1, right = 0.850)
        
        
        lim_E_min, lim_E_max = -50, 700

        ax1 = self.plot_wave_function(ax1, lim_E_min, lim_E_max)
        if 'limx' in options.keys():
            if options['limx'] != '':
                ax1.set(xlim=options['limx'])

        if True: # imprimir a pc
            for _pc, _pc_e in zip (self.result_pc[:], self.E_pc[:]):
                print(_pc, _pc_e)

        if True: # plot comparison
            with open('reference_data/2024_QBMD_GA_paper/files/sim_data.txt', 'r') as sim_txt:
                lines_sim = sim_txt.readlines()
            with open('reference_data/2024_QBMD_GA_paper/files/real_Photocurrent_P458_77K_P0.0V.txt', 'r') as real_txt:
                lines_real = real_txt.readlines()
            
            
            e_sim = []
            pc_sim = []
            for line in lines_sim:
                e, pc = line.split()
                e_sim.append(float(e))
                pc_sim.append(float(pc))

            pc_sim = np.array(pc_sim)
            e_sim = np.array(e_sim)
            max_pc_sim = max(pc_sim)
            pc_sim = pc_sim/max_pc_sim


            # criamos um array de zeros
            pdf = np.array(pc_sim) * 0
            for e, pc in zip(e_sim, pc_sim):
                partial_pdf = cauchy_distribution(e_sim, e, (0.06*e))
                partial_pdf *= pc
                pdf += partial_pdf
            max_pdf = max(pdf)
            pdf = pdf/max_pdf

            e_real = []
            pc_real = []
            for line in lines_real:
                e, pc = line.split()
                e_real.append(float(e))
                pc_real.append(float(pc))

            


            plt.plot(e_sim, pc_sim, label="Simulation", color='#0000cc')
            # plt.plot(e_sim, pdf, label="Simulation (Lorentz distribution)")
            # plt.plot(e_real, pc_real, label="Measured for a real sample at 77 K")

            ############ para interpolar
            if True:
                e_real = e_real[135:701]
                pc_real = pc_real[135:701]
                
                e_sim = e_sim[120:-329]
                pdf = pdf[120:-329]

                pc_real = np.interp(e_sim, e_real, pc_real)
                e_real = e_sim
                plt.plot(e_real, pc_real, label="Measured", color='#ff0000')
                
              # print(f"real: {len(e_real)} - {e_real[0]}:{e_real[-1]}")
              # print(f"sim: {len(e_sim)} - {e_sim[0]}:{e_sim[-1]}")
            gammas = np.arange(0.04, 0.071, 0.001)
          # print(gammas)
            # best => 0.057
            if True:
                gammas = [0.057]
                for gamma in gammas:
                    # criamos um array de zeros
                    pdf = np.array(pdf) * 0
                    for e, pc in zip(e_sim, pc_sim):
                        partial_pdf = cauchy_distribution(e_sim, e, (gamma*e))
                        partial_pdf *= pc
                        pdf += partial_pdf
                    max_pdf = max(pdf)
                    pdf = pdf/max_pdf
                    plt.plot(e_sim, pdf, label="Lorentzian\nfitted\ncurve", color='#00aa00')
                    erro = np.abs(pc_real - pdf)
                  # print(f"{round(gamma,3):.3f} {sum(erro)/1:.4f}")


            ax2.legend(fontsize=font_base-9)
            ax2.set(xlabel = 'ΔE (meV)')
            ax2.set(ylabel = 'Photocurrent intensity (a.u)')
            ax2.grid(axis='x', color='0.75')
            ax2.grid(axis='x', which="minor", color='0.90')
            
            ax2.set_xticks(range(200, 551, 100))
            ax2.xaxis.set_minor_locator(MultipleLocator(20))
            ax2.set(xlim = [250, 520])
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")



                
        ax1.set_title("(a)", x=0.05, y=1.02, size=9)
        ax2.set_title('(b)', x=0.05, y=1.02, size=9)

        
        
        # ax1.set_yticks(range(0, 700+1, 100))
        ax1.set_xticks(range(-150, 151, 30))
        ax1.set(xlim=options['limx'])

        
        for ax in [ax1, ax2]:
            # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.yaxis.offsetText] + ax.get_xticklabels() + ax.get_yticklabels()):
                # item.set_fontsize(16)
            
            for item in ([ax.xaxis.label, ax.yaxis.label]):
                item.set_fontsize(font_axis_text)

            for item in ([ax.title]):
                item.set_fontsize(font_subtitle)

            for item in ([ax.yaxis.offsetText]):
                item.set_fontsize(font_axis_number)

            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(font_axis_number)

            ax.title.set_fontsize(font_title)
            


        # save image
        if self.simulation_type.startswith('paper_test_'):
            name = self.simulation_type.replace('paper_test_', '')
            plt.savefig('temp_files/{}.png'.format(name), dpi=150)
        elif self.simulation_type == "save":
            pass
        
        plt.cla()               # clears an axis, i.e. the currently active axis in the current figure. It leaves the other axes untouched.
        plt.clf()               # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
        plt.close()             # closes a window, which will be the current window, if not specified otherwise. 
        plt.close('all') 

        del(fig)



    def process_results_wf(self):
        self.x_nm = self.x*1e+9
        self.E0 = round(self.psi0[0][3]*1000/self.e_charge,6)
        
        # structure
        self.result_potential = self.v*1000/self.e_charge

        # for wave functions
        self.result_wavefunction = []
        for i, [ne, psi, vec_probability, EEEE] in enumerate(self.psi0):
            Energy = round(EEEE/self.e_charge,6)
            if i == 0:
                self.result_wavefunction.append([Energy*1000, (vec_probability*1000*2)+Energy*1000])
            elif Energy > self.barreira:
                self.result_wavefunction.append([Energy*1000, (vec_probability*1000/2)+Energy*1000])
            else:
                self.result_wavefunction.append([Energy*1000, (vec_probability*1000/1)+Energy*1000])

        

        # Oscillator Strength
        max_oscstr_under_barrier = 0
        max_oscstr_above_barrier = 0
        self.max_oscstr_under_index = 0
        self.max_oscstr_above_index = 0
        self.result_oscstr = []
        for i, [oscstr, E, Ef, Ei] in enumerate(self.oscstr):
            self.result_oscstr.append([oscstr, E*1000/self.e_charge, Ef*1000/self.e_charge, Ei*1000/self.e_charge])
            if Ef/self.e_charge > self.barreira:
                if oscstr > max_oscstr_above_barrier:
                    max_oscstr_above_barrier = oscstr
                    self.max_oscstr_above_index = i
            else:
                if oscstr > max_oscstr_under_barrier:
                    max_oscstr_under_barrier = oscstr
                    self.max_oscstr_under_index = i

    def plot_structure_wf(self):
        fig, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [16, 4]})
        plt.subplots_adjust(wspace = 0.0, top = 0.8)
        plt.gcf().text(0.01, 0.01, "Fortran em Python")
        
        
        # plot WF
        for E, wf in self.result_wavefunction:
            ax1.plot(self.x_nm, wf, color='#aaaaaa', linewidth=0.7)

        # plot structure
        ax1.plot(self.x_nm, self.result_potential, color='#5555ff', linewidth=0.8)
        # plot WF max under the barrier
        ax1.plot(self.x_nm, self.result_wavefunction[self.max_oscstr_under_index+0][1], color='#ffaaaa', linewidth=1.5)
        # plot WF max over the barrier
        ax1.plot(self.x_nm, self.result_wavefunction[self.max_oscstr_above_index+0][1], color='#ff0000', linewidth=1.5)
        # plot WF E0
        ax1.plot(self.x_nm, self.result_wavefunction[0][1], color='#555555', linewidth=1.5)


        # plot Oscillator Strength
        for i in range(1, len(self.result_oscstr)):
            ax2.plot(self.result_oscstr[i][0], self.result_oscstr[i][1], '.', color='red', markersize=4)

        ax3 = ax2.twiny()




        ######## set axes names and plots limits
        lim_E_min, lim_E_max = -50, 800
        # lim_E_min, lim_E_max = 540, 550
        lim_x1 = [min(self.x_nm), max(self.x_nm)]
        # lim_x1 = [-100, 100]
        lim_y1 = [lim_E_min, lim_E_max]
        lim_y2 = [lim_E_min-self.E0, lim_E_max-self.E0]
        lim_y3 = [lim_E_min-self.E0, lim_E_max-self.E0]


        ax1.set(xlabel="Length (nm)", ylabel="Energy (meV)")
        ax1.set(xlim=lim_x1, ylim=lim_y1)

        ax2.set(ylim=lim_y2)
        ax2.set_title('(b)', x=0.0, y=1, size=9)
        ax2.set_xlabel('OscStr', color="red")
        ax2.set_ylabel('ΔE (meV)', fontdict=None, labelpad=0)
        ax2.tick_params(axis='x', labelcolor="red")
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")


        # ax3.set(ylim=lim_y3)
        ax3.set_xlabel('PC (a.u)', color="blue", x=0.5, labelpad=14)
        ax3.tick_params(axis='x', labelcolor="blue", labelsize=8)
        ax3.xaxis.offsetText.set_fontsize(8)



        # set titles

        ax1.set_title("(a)", x=0.0, y=1.1, size=9)
        
        ax2.set_title('(b)', x=0.0, y=1.1, size=9)
        ax2.yaxis.set_label_position("right")


        # set main title
        
        lw = self.structure[0]
        lqw = self.structure[1]
        lqb = self.structure[2]

        de = self.structure[3]
        
        rw = self.structure[4]
        rqw = self.structure[5]
        rqb = self.structure[6]


        title_es = "Left || {:02.0f} - QW: {:04.1f} nm - QB: {:04.1f} nm\n Right || {:02.0f} - QW: {:04.1f} nm - QB: {:04.1f} nm\nDefect: {:04.1f} nm".format(lw, lqw, lqb, rw, rqw, rqb, de)


        fig.suptitle(title_es, x=0.4, fontsize=10)

        if "test_" in self.simulation_type:
            plt.savefig('temp_files/000' + self.simulation_type[-1] + '.png', dpi=150)

        # clear plots
        plt.cla()               # clears an axis, i.e. the currently active axis in the current figure. It leaves the other axes untouched.
        plt.clf()               # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
        plt.close()             # closes a window, which will be the current window, if not specified otherwise. 
        plt.close('all') 

        del(fig)
        del(ax1)
        del(ax2)

    def plot_wave_function(self, ax_graph, lim_E_min, lim_E_max):
        # to plot all wave functions
        for E, wf in self.result_wavefunction:
            ax_graph.plot(self.x_nm, wf, color='#555555', linewidth=0.5)
        
        # plot structure
        ax_graph.plot(self.x_nm, self.result_potential, color='#5555ff', linewidth=1.0)
        # to fill under the structure
        ax_graph.fill_between(self.x_nm,self.result_potential, color='#e2e2ff')
        ax_graph.fill_between(self.x_nm,-100, color='#e2e2ff')
        # plot WF max under the barrier
        # ax_graph.plot(self.x_nm, self.result_wavefunction[self.max_oscstr_under_index+0][1], color='#ffaaaa', linewidth=1.5)
        
        # plot WF max over the barrier
        ax_graph.plot(self.x_nm, self.result_wavefunction[self.max_oscstr_above_index-0][1], color='#ff0000', linewidth=1.5)
        # print(self.result_wavefunction[self.max_oscstr_above_index-0][0] - self.E0)
        
        # plot WF E0
        ax_graph.plot(self.x_nm, self.result_wavefunction[0][1], color='#555555', linewidth=1.5)


        # set limit axes
        lim_x1 = [min(self.x_nm), max(self.x_nm-2)]
        # lim_x1 = [-75, 75]
        # lim_x1 = [-70, 50]
        lim_y1 = [lim_E_min, lim_E_max]
        
        # lim_x1 = [self.x_nm[-20], self.x_nm[-1]+2]
        # lim_y1 = [450, 600]

        ax_graph.set(xlabel="Thickness (nm)", ylabel="Energy (meV)")
        ax_graph.set(xlim=lim_x1, ylim=lim_y1)

        return ax_graph

    def plot_oscillator_strength(self, ax_graph, lim_E_min, lim_E_max):
        # plot Oscillator Strength
        for i in range(0, len(self.result_oscstr)):
            ax_graph.plot(self.result_oscstr[i][0], self.result_oscstr[i][1], '.', color='red', markersize=3)

        # set axes limits
        lim_y2 = [lim_E_min-self.E0, lim_E_max-self.E0]
        ax_graph.set(ylim=lim_y2)
        ax_graph.set_xlabel('OscStr', color="red")
        ax_graph.set_ylabel('ΔE (meV)', fontdict=None, labelpad=0)
        ax_graph.set_ylabel('ΔE (meV)', fontdict=None, labelpad=-10)
        ax_graph.tick_params(axis='x', labelcolor="red")
        ax_graph.yaxis.tick_right()
        ax_graph.yaxis.set_label_position("right")

        return ax_graph

    def plot_oscillator_strength_horizontal(self, ax_graph, lim_E_min, lim_E_max):
        # plot Oscillator Strength
        for i in range(0, len(self.result_oscstr)):
            ax_graph.plot(self.result_oscstr[i][1], self.result_oscstr[i][0], '.', color='red', markersize=10)

        # set axes limits
        lim_x = [lim_E_min, lim_E_max]
        # print('osc_str', lim_x)
        ax_graph.set(xlim=lim_x)
        ax_graph.set_xlabel('ΔE (meV)')
        ax_graph.set_ylabel('Oscillator strength', color="red", fontdict=None, labelpad=1)
        ax_graph.tick_params(axis='y', labelcolor="red")
        ax_graph.yaxis.tick_left()
        ax_graph.yaxis.set_label_position("left")

        return ax_graph

    def plot_photocurrent(self, ax_graph, lim_E_min, lim_E_max):
        ax_graph.plot(self.result_pc[:], self.E_pc[:], color='blue', linewidth=1.0)
        
        ax_graph.plot(self.max_photocurrent, self.max_e_photocurrent, '.', color='blue', linewidth=1.0)
        ax_graph.plot(self.min_photocurrent, self.min_e_photocurrent, '.', color='blue', linewidth=1.0)

        # print("------------------")
        # print("max", self.max_e_photocurrent, self.max_photocurrent)
        # print("min", self.min_e_photocurrent, self.min_photocurrent)

        ax_graph.set_xlabel('PC (a.u)', color="blue", x=0.5, labelpad=14)
        ax_graph.tick_params(axis='x', labelcolor="blue", labelsize=8)
        ax_graph.xaxis.offsetText.set_fontsize(8)

        return ax_graph
    
    def plot_oscillator_strength_empty(self, ax_graph, lim_E_min, lim_E_max):
        # plot Oscillator Strength
        # for i in range(0, len(self.result_oscstr)):
            # ax_graph.plot(self.result_oscstr[i][0], self.result_oscstr[i][1], '.', color='red', markersize=3)

        # set axes limits
        lim_y2 = [lim_E_min-self.E0, lim_E_max-self.E0]
        ax_graph.set(ylim=lim_y2)
        # ax_graph.set_xlabel('OscStr', color="red")
        # ax_graph.set_ylabel('ΔE (meV)', fontdict=None, labelpad=0)
        # ax_graph.set_ylabel('ΔE (meV)', fontdict=None, labelpad=-10)
        # ax_graph.tick_params(axis='x', labelcolor="red")
        ax_graph.set_xticks([])
        ax_graph.yaxis.tick_right()
        ax_graph.yaxis.set_label_position("right")

        return ax_graph

    
    
    def plot_photocurrent_horizontal(self, ax_graph, lim_E_min, lim_E_max):
        lim_x = [lim_E_min, lim_E_max]
        # print('pc', lim_x)
        
        # para adicionar a pc 0 até a energia 0 e completar a linha abaixo da barreira
        self.E_pc = np.insert(self.E_pc, 0, 0)
        self.result_pc = np.insert(self.result_pc, 0, 0)
        
        ax_graph.set(xlim=lim_x)
        ax_graph.plot(self.E_pc[:], self.result_pc[:], color='blue', linewidth=2.0)
        
        ax_graph.plot(self.max_e_photocurrent, self.max_photocurrent, '.', color='blue', linewidth=5.0)
        ax_graph.plot(self.min_e_photocurrent, self.min_photocurrent, '.', color='blue', linewidth=5.0)

        ax_graph.set_ylabel('Photocurrent intensity (a.u)', color="blue", x=0.5, labelpad=14)
        ax_graph.tick_params(axis='y', labelcolor="blue", labelsize=8)
        ax_graph.yaxis.offsetText.set_fontsize(8)


        return ax_graph
    
    def plot_photocurrent_w(self, ax_graph, lim_E_min, lim_E_max):
        """
        graficar fotocorrente com a distribuição de cauchy
        """
        
        # função anterior
        ax_graph.plot(self.result_pc[:], self.E_pc[:], color='blue', linewidth=1.0)
      # print(len(self.result_pc), len(self.E_pc))
        with open('C:/CodesJer/GA_PD/temp_files/data.txt', 'w') as wf:
            for e, pc in zip(self.E_pc, self.result_pc):
                wf.write(f"{e}\t{pc}\n")
            

        ax_graph.plot(self.max_photocurrent, self.max_e_photocurrent, '.', color='blue', linewidth=1.0)
        ax_graph.plot(self.min_photocurrent, self.min_e_photocurrent, '.', color='blue', linewidth=1.0)
        ax_graph.set_xlabel('PC (a.u)', color="blue", x=0.5, labelpad=14)
        ax_graph.tick_params(axis='x', labelcolor="blue", labelsize=8)
        ax_graph.xaxis.offsetText.set_fontsize(8)

        # criamos um array de zeros
        pdf = np.array(self.E_pc) * 0
        for e, pc in zip(self.E_pc, self.result_pc):
            partial_pdf = self.cauchy_distribution(self.E_pc, e, 8.0)
            partial_pdf *= pc
            pdf += partial_pdf
            # ax_graph.plot(partial_pdf, self.E_pc, color='blue', linewidth=1.0)

        ax_graph.plot(pdf, self.E_pc[:], color='magenta', linewidth=1.0)
        return ax_graph
    
    @staticmethod
    def cauchy_distribution(x, x0, gamma):
        """
        Cauchy distribution (Lorentz distribution) probability density function (PDF).
        
        :param x: The value at which to evaluate the PDF.
        :param x0: Location parameter.
        :param gamma: Scale parameter.
        :return: The PDF value at x.
        """
        return 1.0 / (np.pi * gamma * (1 + ((x - x0) / gamma)**2))
    
    
    def plot_wave_function_ext(self, ax_graph, lim_E_min, lim_E_max):
        # to plot all wave functions
        for E, wf in self.result_wavefunction:
            ax_graph.plot(self.x_nm, wf, color='#555555', linewidth=0.5)
        
        # plot structure
        ax_graph.plot(self.x_nm, self.result_potential, color='#5555ff', linewidth=1.0)
        # to fill under the structure
        ax_graph.fill_between(self.x_nm,self.result_potential, color='#e2e2ff')
        ax_graph.fill_between(self.x_nm,-100, color='#e2e2ff')
        # plot WF max under the barrier
        ax_graph.plot(self.x_nm, self.result_wavefunction[self.max_oscstr_under_index+0][1], color='#ffaaaa', linewidth=1.5)
        # plot WF max over the barrier
        ax_graph.plot(self.x_nm, self.result_wavefunction[self.max_oscstr_above_index+0][1], color='#ff0000', linewidth=1.5)
        # plot WF E0
        ax_graph.plot(self.x_nm, self.result_wavefunction[0][1], color='#555555', linewidth=1.5)


        # set limit axes
        lim_x1 = [min(self.x_nm), max(self.x_nm)]
        # lim_x1 = [-75, 75]
        # lim_x1 = [-70, 50]
        lim_y1 = [lim_E_min, lim_E_max]

        ax_graph.set(xlabel="Thickness (nm)", ylabel="Energy (meV)")
        ax_graph.set(xlim=lim_x1, ylim=lim_y1)

        return ax_graph

    def plot_oscillator_strength_ext(self, ax_graph, lim_E_min, lim_E_max):
        # plot Oscillator Strength
        for i in range(0, len(self.result_oscstr)):
            ax_graph.plot(self.result_oscstr[i][0], self.result_oscstr[i][1], '.', color='red', markersize=3)

        # set axes limits
        lim_y2 = [lim_E_min-self.E0, lim_E_max-self.E0]
        ax_graph.set(ylim=lim_y2)
        ax_graph.set_xlabel('OscStr', color="red")
        ax_graph.set_ylabel('ΔE (meV)', fontdict=None, labelpad=0)
        ax_graph.set_ylabel('ΔE (meV)', fontdict=None, labelpad=-10)
        ax_graph.tick_params(axis='x', labelcolor="red")
        ax_graph.yaxis.tick_right()
        ax_graph.yaxis.set_label_position("right")

        return ax_graph

    def plot_photocurrent_ext(self, ax_graph, lim_E_min, lim_E_max):
        ax_graph.plot(self.result_pc[:], self.E_pc[:], color='blue', linewidth=1.0)
        
        # ax_graph.plot(self.max_abs_photocurrent, self.max_e_abs_photocurrent, '.', color='blue', linewidth=1.0)
        # ax_graph.plot(-self.min_abs_photocurrent, self.min_e_abs_photocurrent, '.', color='blue', linewidth=1.0)

        ax_graph.set_xlabel('PC (a.u)', color="blue", x=0.5, labelpad=14)
        ax_graph.tick_params(axis='x', labelcolor="blue", labelsize=8)
        ax_graph.xaxis.offsetText.set_fontsize(8)


        return ax_graph
    
    
    def process_results_osc(self):
        self.x_nm = self.x*1e+9
      # print('---', self.psi0[0][3])
        self.E0 = round(self.psi0[0][3]*1000/self.e_charge,6)
        
        # structure
        self.result_potential = self.v*1000/self.e_charge

        # for wave functions
        self.result_wavefunction = []
        for i, [ne, psi, vec_probability, EEEE] in enumerate(self.psi0):
            vec_probability = np.array(vec_probability)
            vec_probability = vec_probability/max(vec_probability)
            Energy = round(EEEE*1000/self.e_charge,6)
          # print(Energy)
            if i == 0:
                self.result_wavefunction.append([Energy, (vec_probability*50)+Energy])
            elif Energy > self.barreira*1000:
                self.result_wavefunction.append([Energy, (vec_probability*25)+Energy])
            else:
                self.result_wavefunction.append([Energy, (vec_probability*25)+Energy])

        

        # Oscillator Strength
        self.max_oscstr_under_barrier = 0
        self.max_e_oscstr_under_barrier = 0
        self.max_oscstr_above_barrier = 0
        self.max_e_oscstr_above_barrier = 0
        self.max_oscstr = 0
        self.max_e_oscstr = 0

        self.max_oscstr_under_index = 0
        self.max_oscstr_above_index = 0
        self.result_oscstr = []
        return None
    
    def plot_structure_osc(self):
        fig, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [16, 4]})
        plt.subplots_adjust(wspace = 0.0, top = 0.8)
        # plt.gcf().text(0.01, 0.01, "Fortran em Python")
        lim_E_min, lim_E_max = -50, 700
        # lim_E_min, lim_E_max = 492, 493
        # lim_E_min, lim_E_max = 490, 491
        # lim_E_min, lim_E_max = 480, 501

        # for pc, ele in zip(self.result_pc, self.E_pc):
        #   # print(ele, pc)

        ax1 = self.plot_wave_function(ax1, lim_E_min, lim_E_max)
        ax2 = self.plot_oscillator_strength(ax2, lim_E_min, lim_E_max)

        # set titles
        ax1.set_title("(a)", x=0.0, y=1.1, size=9)
        ax2.set_title('(b)', x=0.0, y=1.1, size=9)

        # set main title
        lw, lqw, lqb, de, rw, rqw, rqb = self.structure
        title_es = "Left || {:02.0f} - QW: {:04.1f} nm - QB: {:04.1f} nm".format(lw, lqw, lqb) + \
            "\nRight || {:02.0f} - QW: {:04.1f} nm - QB: {:04.1f} nm".format(rw, rqw, rqb) + \
            "\nMain QW: {:04.1f} nm\n".format(de)
        # "max1 PC: {:.2e} || PC energy: {:02.1f} (meV)\n".format(self.max_abs_photocurrent, self.max_e_abs_photocurrent) + \
        # "max2 PC: {:.2e} || PC energy: {:02.1f} (meV)\n".format(self.min_abs_photocurrent, self.min_e_abs_photocurrent) + \
        fig.suptitle(title_es, x=0.42, fontsize=10)

        # save image
        if "test_qqqq" in self.simulation_type:
            plt.savefig('temp_files/000' + self.simulation_type[-1] + '.png', dpi=150)
        elif self.simulation_type == "test":
            plt.savefig('temp_files/0001.png', dpi=150)
        elif self.simulation_type.startswith('test_'):
            name = self.simulation_type.replace('test_', '')
            plt.savefig('temp_files/{}.png'.format(name), dpi=150)
        elif self.simulation_type == "image_folder":
            plt.savefig(self.png_path, dpi=150)
        elif self.simulation_type == "image_to_map":
            plt.savefig(self.png_path, dpi=150)
        elif self.simulation_type == "develop":
            plt.savefig('temp_files/dev_python.png', dpi=150)
        elif self.simulation_type.startswith("gif_"):
            image_number = int(self.simulation_type.replace("gif_", ""))
            file_name = self.output_folder + "gifs/img/" + str(image_number).zfill(4) + ".png"
            plt.savefig(file_name, dpi=150)
        elif self.simulation_type == "save":
            pass
        
        self.plot_wf = ax1
        self.plot_os = ax2
        # clear plots
        plt.cla()               # clears an axis, i.e. the currently active axis in the current figure. It leaves the other axes untouched.
        plt.clf()               # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
        plt.close()             # closes a window, which will be the current window, if not specified otherwise. 
        plt.close('all') 

        del(fig)
        del(ax1)
        del(ax2)
