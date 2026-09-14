#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jose Ruiz
@modified_date: 2024-08-24
"""

####################################################################################################
# imports

import sys, os

# Obtén la ruta absoluta del archivo actual
current_file = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.append(project_path)



from enum import Enum

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



import numpy as np
import pandas as pd
import math, pickle
import logging

from dataclasses import dataclass, field
from typing import Optional

# Local application/library specific imports
from windows_executable.source.fortran_simulation.utils import set_structure_values
from windows_executable.source.fortran_simulation.SimulationOptions import SimulationOptions
from windows_executable.source.fortran_simulation.DataEnums import PlotType, ParametersSimulation
from windows_executable.source.fortran_simulation.logger import logger
from windows_executable.source.fortran_simulation.SampleData import SampleData
from conf import file_ed_exe



class FortranSimulator():
    def __init__(self, individual,
                 sim_options:SimulationOptions,
                 output_folder="temp_files/fortran_test_ds/dev/"):
        """_summary_
        LLQW: number of lateral quantum wells on left side
        LQW: left quantum well thickness
        LQB: left quantum barrier thickness
        MQW: main quantum well thickness
        RLQW: number of lateral quantum wells on right side
        RQW: right quantum well thickness
        RQB: right quantum barrier thickness
        Args:
            structure (list, optional): _description_. Defaults to [0,0,0, 2.5, 0,0,0].
            output_folder (str, optional): _description_. Defaults to "temp_files/fortran_test_ds/dev/".
            
        """
        
        self.individual = individual
        structure = set_structure_values(individual)
        self.structure = structure
        self.output_folder = output_folder
        self.sim_options = sim_options
        self.sample_data = SampleData()
        self.barrier_value = 503
        self.barrier_value_accepted = self.barrier_value + 10
        
        logger.info(f"Starting simulation structure: {structure}")
        
        
        self.f90_file = 'source/fortran_simulation/fortran_files/20240825_Superlattice_Eigensates_Photocurrent.f90'
        
        self.left_rep = structure[0]
        self.left_wells = structure[1]
        self.left_barriers = structure[2]
        
        self.defect = structure[3]
        
        self.right_rep = structure[4]
        self.right_wells = structure[5]
        self.right_barriers = structure[6]

        self.lr = structure[0]
        self.lw = structure[1]
        self.lb = structure[2]
        self.de = structure[3]
        self.rr = structure[4]
        self.rw = structure[5]
        self.rb = structure[6]
        
        self.LLQW_str = lr_str = "{:02.0f}".format(structure[0])
        self.LQW_str  = lw_str = "{:04.1f}".format(structure[1])
        self.LQB_str  = lb_str = "{:04.1f}".format(structure[2])
        self.MQW_str  = de_str = "{:04.1f}".format(structure[3])
        self.RLQW_str = rr_str = "{:02.0f}".format(structure[4])
        self.RQW_str  = rw_str = "{:04.1f}".format(structure[5])
        self.RQB_str  = rb_str = "{:04.1f}".format(structure[6])
        
        # definir nombres de carpetas y archivos
        self.file_id     = lr_str+"x"+lw_str+"_"+lb_str+"__"+de_str+"__"+rr_str+"x"+rw_str+"_"+rb_str
        self.title_img   = lr_str+"x"+lw_str+"_"+lb_str+"  "+de_str+"  "+rr_str+"x"+rw_str+"_"+rb_str
        self.pkl_path    = f"{output_folder}pkls/{self.file_id}.pkl"
        os.makedirs(f'{output_folder}pkls/', exist_ok=True)
        self.exist_plk = os.path.exists(self.pkl_path)
        
        self.simulation_folder = f'{output_folder}temp/{self.file_id}/'

        self.file_ed = self.simulation_folder + self.file_id + ".f90"
        self.file_ed_exe = self.simulation_folder + self.file_id + ".exe"

        # output_files 
        self.wavefunction_file = self.simulation_folder + "wavefunction_SL.txt"
        self.transmission_file = self.simulation_folder + "Transmission_SL.txt"
        self.photocurrent_file = self.simulation_folder + "Photocurrent_SL.txt"
        self.potencial_file = self.simulation_folder + "Potencial_SL.txt"
        self.oscstr_file = self.simulation_folder + "OscStr_SL.txt"
        self.oscstr_norm_file = self.simulation_folder + "OscStr_SL_norm.txt"
        self.energy_file = self.simulation_folder + "Energy_SL.txt"
        
        self.fimprograma_file = self.simulation_folder + "FimPrograma.txt"
        self.png_file = self.simulation_folder + "image.png"
        
        self.files_to_delete = [self.file_ed,
                                    self.file_ed_exe,
                                    self.wavefunction_file,
                                    self.transmission_file,
                                    self.photocurrent_file,
                                    self.potencial_file,
                                    self.oscstr_file,
                                    self.oscstr_norm_file,
                                    self.energy_file,
                                    self.fimprograma_file,
                                    self.png_file]
            
    def modify_f90_file(self):
        os.makedirs(self.simulation_folder, exist_ok=True)
        for file_to_delete in self.files_to_delete:
                if os.path.exists(file_to_delete):
                    os.remove(file_to_delete)
                    
        values_modify = [
            ['&&_&_LLQW_&_&&', self.LLQW_str],
            ['&&_&_LQW_&_&&', self.LQW_str],
            ['&&_&_LQB_&_&&', self.LQB_str],
            ['&&_&_MQW_&_&&', self.MQW_str],
            ['&&_&_RLQW_&_&&', self.RLQW_str],
            ['&&_&_RQW_&_&&', self.RQW_str],
            ['&&_&_RQB_&_&&', self.RQB_str],
            ['&&_&_FORTRAN_FOLDER_&_&&', self.simulation_folder],
            
        ]
        
        
        if not os.path.exists(self.file_ed):
            with open(self.f90_file, 'r', encoding="utf-8") as file:
                file_text = file.read()
            
            for _src, _dst in values_modify:
                file_text = file_text.replace(_src, _dst)
            
            with open(self.file_ed, 'w') as file:
                file.write(file_text)

    def run_with_inputs(self):
        os.makedirs(self.simulation_folder, exist_ok=True)
        
        self.LLQW_str_exe = "{:.0f}".format(self.structure[0])
        self.LQW_str_exe  = "{:.1f}d0".format(self.structure[1])
        self.LQB_str_exe  = "{:.1f}d0".format(self.structure[2])
        self.MQW_str_exe  = "{:.1f}d0".format(self.structure[3])
        self.RLQW_str_exe = "{:.0f}".format(self.structure[4])
        self.RQW_str_exe  = "{:.1f}d0".format(self.structure[5])
        self.RQB_str_exe  = "{:.1f}d0".format(self.structure[6])
        
        command_structure = f'{self.LLQW_str_exe} {self.LQW_str_exe} {self.LQB_str_exe} {self.MQW_str_exe} {self.RLQW_str_exe} {self.RQW_str_exe} {self.RQB_str_exe}'
        comand_to_run = f'{file_ed_exe} {command_structure} "{self.simulation_folder}"'
        
        os.system(comand_to_run)
        # aguarda até que exista o arquivo fimprograma
        while not os.path.exists(self.fimprograma_file):
            pass
    
    def compile_run(self):
        self.only_compile()
        self.only_run()

    def only_compile(self):
        # abrir msys2 e compilar o arquivo .f90 salvando como .exe
        self.comand_to_compile = f'gfortran {self.file_ed} -o {self.file_ed_exe}'
        os.system(self.comand_to_compile)
        # aguarda até que exista o arquivo .exe
        while not os.path.exists(self.file_ed_exe):
            pass
        
    def only_run(self):
        if not os.path.exists(self.fimprograma_file):
            self.comand_to_run = f'{self.file_ed_exe}'
            os.system(self.comand_to_run)
            # aguarda até que exista o arquivo fimprograma
            while not os.path.exists(self.fimprograma_file):
                pass
    
    def get_time(self):
        with open(self.fimprograma_file) as f:
            line = f.read()
        self.total_time = float(line.strip())
        
    def get_potencial(self):
        with open(self.potencial_file) as f:
            lines = f.readlines()
            self.x_potencial=[]
            self.y_potencial=[]
            for line in lines:
                self.x_potencial.append(float(line.split()[0]))
                self.y_potencial.append(float(line.split()[1]))
        self.x_potencial = np.array(self.x_potencial)
        self.y_potencial = np.array(self.y_potencial)

    def get_wavefunction(self):
        with open(self.wavefunction_file) as f:
            df_ = pd.read_csv(f, comment='#', header=None, sep=r'\s+')

            self.x_wavefunction=df_.loc[:,0].values

            self.wavefunctions = []
            self.autoenergias = []
            for i in range(len(df_.columns)):
                y_=df_.loc[:,i].values
                self.wavefunctions.append(y_)
                # primeiro valor da coluna é a energia
                self.autoenergias.append(y_[0])
        
        # desde o segundo para nao pegar o primeiro valor do X
        self.autoenergias = self.autoenergias[1:]
        self.E0 = self.autoenergias[0]
        
        # processar dados
        self.autoenergias = np.array(self.autoenergias)
        
        self.wavefunctions_norm = []
        for energy, wf in zip(self.autoenergias, self.wavefunctions[1:]):
            wf = np.array(wf)
            wf_norm = (wf - np.min(wf)) / (np.max(wf) - np.min(wf))
            if energy <= self.barrier_value:
                wf_norm = wf_norm * 50
            else:
                wf_norm = wf_norm * 25
                
            wf_norm = wf_norm + energy
            self.wavefunctions_norm.append(wf_norm)
        self.wavefunctions_norm = np.array(self.wavefunctions_norm)

    def get_oscstr(self):
        with open(self.oscstr_file) as f:
            lines = f.readlines()
            self.e_transition=[]
            self.e_oscstr=[]
            for line in lines:
                self.e_transition.append(float(line.split()[0]))
                self.e_oscstr.append(float(line.split()[1]))
    
        # para criar o dataframe
        self.e_oscstr_pd = pd.DataFrame(columns=["fundamental", "energy", "transition", "oscstr"])

        # Lista para acumular los DataFrame individuales
        dataframes = []

        # Iteración sobre los datos
        for i in range(len(self.e_oscstr)):
            # Crear un nuevo DataFrame con la fila a añadir
            newdata = pd.DataFrame({
                "fundamental": [self.autoenergias[0]],
                "energy": [self.autoenergias[0] + self.e_transition[i]],
                "transition": [self.e_transition[i]],
                "oscstr": [self.e_oscstr[i]]
            })

            # Eliminar columnas vacías de `newdata` si las hay
            newdata = newdata.dropna(axis=1, how='all')

            # Añadir el DataFrame a la lista
            dataframes.append(newdata)

        # Concatenar todos los DataFrame al final
        self.e_oscstr_pd = pd.concat(dataframes, ignore_index=True)
        
        # para limitar valores só acima da barrerira
        self.e_oscstr_pd["oscstr_f"] = np.where(self.e_oscstr_pd["energy"] > self.barrier_value_accepted, self.e_oscstr_pd["oscstr"],0)
        
        self.e_oscstr_pd["oscstr_f_all"] = self.e_oscstr_pd["oscstr"].copy()
        
        self.oscstr_e = self.e_oscstr_pd['transition'].values
        self.oscstr = self.e_oscstr_pd['oscstr'].values

        self.max_e_oscstr = self.e_oscstr_pd["oscstr_f"].max()
        self.max_e_oscstr_index = self.e_oscstr_pd["oscstr_f"].idxmax()
        self.max_e_transition = self.e_oscstr_pd["transition"][self.max_e_oscstr_index]
        
        self.max_e_oscstr_all = self.e_oscstr_pd["oscstr_f_all"].max()
        self.max_e_oscstr_all_index = self.e_oscstr_pd["oscstr_f_all"].idxmax()
        self.max_e_transition_all = self.e_oscstr_pd["transition"][self.max_e_oscstr_all_index]
 
    def get_photocurrent(self):
        if not self.exist_plk or self.sim_options.force_parser:
            with open(self.photocurrent_file) as f:
                lines = f.readlines()
                self.e_transition_pc=[]
                self.photocurrent=[]
                for line in lines:
                    self.e_transition_pc.append(float(line.split()[0]))
                    self.photocurrent.append(float(line.split()[1]))
                    
        normalize_photocurrent = True
        if normalize_photocurrent:
            # normalizar photocurrent
            # print(self.sim_options.reference_pc)
            reference_pc=1.551237337323166e-11
            self.photocurrent = [pc / reference_pc for pc in self.photocurrent]

            
        

        #remove nan from self.photocurrent
        # print(type(self.photocurrent[0]))
        while math.isnan(self.photocurrent[0]):
            del self.photocurrent[0]
            del self.e_transition_pc[0]
        
        self.max_photocurrent = max(self.photocurrent)
        self.max_photocurrent_index = self.photocurrent.index(self.max_photocurrent)
        self.max_e_photocurrent = self.e_transition_pc[self.max_photocurrent_index]

        self.min_photocurrent = min(self.photocurrent)
        self.min_photocurrent_index = self.photocurrent.index(self.min_photocurrent)
        self.min_e_photocurrent = self.e_transition_pc[self.min_photocurrent_index]

        if self.max_photocurrent > 0 and self.min_photocurrent < 0:
            self.abs_max_photocurrent = max(self.max_photocurrent, -self.min_photocurrent)
            self.abs_min_photocurrent = min(self.max_photocurrent, -self.min_photocurrent)
            if self.abs_max_photocurrent == self.max_photocurrent:
                self.abs_max_e_photocurrent = self.max_e_photocurrent
                self.abs_min_e_photocurrent = self.min_e_photocurrent
            elif self.abs_max_photocurrent == -self.min_photocurrent:
                self.abs_max_e_photocurrent = self.min_e_photocurrent
                self.abs_min_e_photocurrent = self.max_e_photocurrent
            else:
                print("erro calculando max e min da PC")
                sys.exit()
        else:
            if self.max_photocurrent > self.min_photocurrent:
                self.abs_max_photocurrent = self.max_photocurrent
                self.abs_max_e_photocurrent = self.max_e_photocurrent
            else:
                self.abs_max_photocurrent = self.min_photocurrent
                self.abs_max_e_photocurrent = self.min_e_photocurrent

        self.min_photocurrent_index = self.photocurrent.index(self.min_photocurrent)
        self.min_e_photocurrent = self.e_transition_pc[self.min_photocurrent_index]


        # max PC
        self.max_photocurrent = max(self.photocurrent)
        self.max_photocurrent_index = self.photocurrent.index(self.max_photocurrent)
        self.max_e_photocurrent = self.e_transition_pc[self.max_photocurrent_index]

        self.min_photocurrent = min(self.photocurrent)
        self.min_photocurrent_index = self.photocurrent.index(self.min_photocurrent)
        self.min_e_photocurrent = self.e_transition_pc[self.min_photocurrent_index]

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

    def simulate(self):
        if not os.path.exists(self.fimprograma_file):
            self.run_with_inputs()
        self.get_time()
        self.get_potencial()
        self.get_wavefunction()
        self.get_oscstr()
        self.get_photocurrent()
        sample_data = self.get_sample_data()
            
        return sample_data
        
    def get_sample_data(self):
        # inicializa
        sample_data = SampleData()
        sample_data.individual = self.individual
        sample_data.structure = self.structure
        sample_data.sample_id = self.file_id
        sample_data.pkl_path = self.pkl_path
        sample_data.title_img = self.title_img
        sample_data.sim_options = self.sim_options
        sample_data.total_time = self.total_time
        
        
        # structure
        sample_data.x_potencial = self.x_potencial
        sample_data.y_potencial = self.y_potencial
        
        # wavefunctions
        sample_data.autoenergias = self.autoenergias
        sample_data.wavefunctions = self.wavefunctions_norm
        sample_data.E0 = self.E0
        
        # oscillator strength
        sample_data.oscstr_e = self.oscstr_e
        sample_data.oscstr = self.oscstr
        sample_data.e_oscstr_pd = self.e_oscstr_pd
        
        sample_data.max_e_oscstr_index = self.max_e_oscstr_index
        sample_data.max_e_oscstr = self.max_e_oscstr
        sample_data.max_e_transition = self.max_e_transition
        
        # photocurrent
        sample_data.pc_e = self.e_transition_pc
        sample_data.pc = self.photocurrent
        
        sample_data.max_abs_photocurrent = self.max_abs_photocurrent
        sample_data.max_e_abs_photocurrent = self.max_e_abs_photocurrent
        sample_data.min_abs_photocurrent = self.min_abs_photocurrent
        sample_data.min_e_abs_photocurrent = self.min_e_abs_photocurrent
        
        
        sample_data.LLQW = self.lr
        sample_data.LQW = self.lw
        sample_data.LQB = self.lb
        sample_data.MQW = self.de
        sample_data.RLQW = self.rr
        sample_data.RQW = self.rw
        sample_data.RQB = self.rb
        
        sample_data.fitness_function()
        sample_data.result_dictionary()
        
        self.sample_data = sample_data
        return sample_data

    
if __name__ == '__main__':
    from conf import output_fortran_folder
    
    sim_options = SimulationOptions(force_parser=True,
                                    force_simulation=False,
                                    plot_type=PlotType.COMPLETE,
                                    reference_os=0.35883602926785424,
                                    reference_os_e=299.9442612296449,
                                    reference_pc=1.551237337323166e-11,
                                    reference_pc_e=309.6000193186107,
                                    plot_parameter=ParametersSimulation.OS_PC,
                                    lim_x_pc=(200, 400),
                                    # lim_x_structure=(-75, 75),
                                    plot_name='0000',
                                    )
    
    individuo=[[5, 20, 70, 25, 1, 20, 70], "test"]
    # individuo=[[4, 100, 170, 25, 1, 100, 170], "test"]
    structure = individuo[0]
    sample_data = FortranSimulator(structure, sim_options=sim_options, output_folder=f"{output_fortran_folder}").simulate()
    sample_data.plot()
    


