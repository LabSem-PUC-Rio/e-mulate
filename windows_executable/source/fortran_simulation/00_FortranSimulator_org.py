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
project_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.append(project_path)


import logging

from enum import Enum

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker



import numpy as np
import pandas as pd
import math, pickle
import random

import time, glob, shutil

from dataclasses import dataclass, field
from typing import Optional


# from ast import Assert
# from hashlib import new
# import os, sys, re
# from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
# from numpy import lib, size
# import datetime
# import imageio
# import tqdm
# from multiprocessing import Pool, cpu_count
# from pandas.core.frame import DataFrame
# from tqdm.contrib.concurrent import process_map  # or thread_map
# from matplotlib import cm


# Configuración básica del logging
    
logging.basicConfig(
    level=logging.INFO,  # Nivel de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='temp_files/QBMD_GA.log',  # Nombre del archivo de log
    filemode='w'  # Modo 'a' para añadir registros, 'w' para sobrescribir
)

logger = logging.getLogger('QBMD_GA')

class PlotType(Enum):
    COMPLETE = "complete"
    COMPARISON = "comparison"
    HORIZONTAL = "complete_horizontal"
    
class ParametersSimulation(Enum):
    OS = "OscillatorStrength"
    PC = "PhotoCurrent"
    OS_PC = "OscillatorStrength_PhotoCurrent"

@dataclass
class SimulationOptions:
    force_simulation: bool = False
    force_parser: bool = False
    plot_type: PlotType = PlotType.COMPLETE
    plot_parameter: ParametersSimulation = ParametersSimulation.OS_PC
    plot_name: str = "id"
    otimization_parameter: str = None
    lim_x_structure: tuple = None
    lim_x_pc: tuple = None
    include_os: bool = False
    include_pc: bool = False
    
    # reference
    reference_pc: float = None
    reference_pc_e: float = None
    reference_os: float = None
    reference_os_e: float = None
    
    
    def __post_init__(self):
        if self.plot_parameter in [ParametersSimulation.OS, ParametersSimulation.OS_PC]:
            self.include_os = True
        
        if self.plot_parameter in [ParametersSimulation.PC, ParametersSimulation.OS_PC]:
            self.include_pc = True
            
        if self.include_os and self.include_pc:
            self.otimization_parameter = 'os_pc'
        elif self.include_os and not self.include_pc:
            self.otimization_parameter = 'os'
        elif not self.include_os and self.include_pc:
            self.otimization_parameter = 'pc'
        else:
            raise('no valid otimization parameter')
    
@dataclass
class SampleData:
    sim_options: SimulationOptions = SimulationOptions()
    total_time: float = 0
    
    x_potencial: np.ndarray = field(default_factory=lambda: np.array([]))
    y_potencial: np.ndarray = field(default_factory=lambda: np.array([]))
    autoenergias: np.ndarray = field(default_factory=lambda: np.array([]))
    wavefunctions: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    e_oscstr_pd: pd.DataFrame = field(default_factory=pd.DataFrame)
    # max_e_oscstr_index: int = 0
    E0: float = 0
    
    oscstr_e: np.ndarray = field(default_factory=lambda: np.array([]))
    oscstr: np.ndarray = field(default_factory=lambda: np.array([]))
    max_e_oscstr_index: int = 0
    max_e_oscstr: float = 0
    max_e_transition: float = 0
    
    pc_e: np.ndarray = field(default_factory=lambda: np.array([]))
    pc: np.ndarray = field(default_factory=lambda: np.array([]))
    max_abs_photocurrent: float = 0
    max_e_abs_photocurrent: float = 0
    min_abs_photocurrent: float = 0
    min_e_abs_photocurrent: float = 0
    
    # strcuture
    LLQW: int = None
    LQW: float = None
    LQB: float = None
    MQW: float = None
    RLQW: int = None
    RQW: float = None
    RQB: float = None
    
    # paths
    sample_id: Optional[str] = None
    pkl_path: Optional[str] = None
    title_img: Optional[str] = None
    
    # optimization
    fitness: float = 0.0
    target: float = 300.0
    target_tolerance: float = 20.0
    
    def target_range(self):
        """Returns the acceptable range for the target."""
        return (self.target - self.target_tolerance, self.target + self.target_tolerance)
    
    def fitness_function(self, prints=False):
        self.fitness = 0
        
        def valid_energy(value):
            limit_min, limit_max = self.target_range()
            if value > limit_max or value < limit_min:
                return False
            return True

        #new fitness version, only PC
        
        if self.sim_options.otimization_parameter == "os":
            self.fitness = self.max_e_oscstr
            if not valid_energy(self.max_e_transition):
                self.fitness = 0
        elif self.sim_options.otimization_parameter == "pc":
            if self.max_abs_photocurrent > 0:
                self.fitness = (math.log10(abs(self.max_abs_photocurrent))*4) + 50
                if not valid_energy(self.max_e_abs_photocurrent):
                    self.fitness = 0
        elif self.sim_options.otimization_parameter == "os_pc":
            if self.max_abs_photocurrent > 0:
                fitness_pc = (math.log10(abs(self.max_abs_photocurrent))*4) + 50
            else:
                fitness_pc = 0
            
            fitness_oscstr = self.max_e_oscstr
            if prints:
                print(f"fitness_pc: {fitness_pc}")
                print(f"fitness_oscstr: {fitness_oscstr} * 20 : {fitness_oscstr * 20}")
            
            self.fitness = fitness_pc + (fitness_oscstr * 20)
            if prints:
                print(f"fitness: {self.fitness}")

            if not valid_energy(self.max_e_abs_photocurrent):
                self.fitness = 0
            if not valid_energy(self.max_e_transition):
                self.fitness = 0


        if prints:
            print("max1 PC: {:.2e} - E: {:02.1f} (meV)".format(self.max_abs_photocurrent, self.max_e_abs_photocurrent))
            print("max2 PC: {:.2e} - E: {:02.1f} (meV)".format(self.min_abs_photocurrent, self.min_e_abs_photocurrent))
            print("OscStr: {:.2f} - E: {:02.1f} (meV)".format(self.max_e_oscstr, self.max_e_transition))
            print("E0: {:02.1f} (meV)".format(self.E0))
            print("aptidao do individuo: ", round(self.fitness, 3))

    def plot(self):
        logger.info(f"Starting plot")
        logger.info(f"plot type: {self.sim_options.plot_type}")
        logger.info(f"plot parameter: {self.sim_options.plot_parameter}")
        if self.sim_options.plot_type == PlotType.COMPLETE:
            Plotter(self).plot_simulation()
        if self.sim_options.plot_type == PlotType.HORIZONTAL:
            Plotter(self).plot_horizontal_simulation()
        if self.sim_options.plot_type == PlotType.COMPARISON:
            Plotter(self).plot_comparison_simulation()



    def print_result(self):
        _pc = self.max_abs_photocurrent
        _pc_e = self.max_e_abs_photocurrent
        _os = self.max_e_oscstr
        _os_e = self.max_e_transition
        print(f"OS: {_os:8.4f}     || OS Energy: {_os_e:5.2f} (meV)")
        print(f"PC: {_pc:12.4e} || PC energy: {_pc_e:5.2f} (meV)")


class FortranSimulator():
    def __init__(self, structure=[0,0,0, 2.5, 0,0,0],
                 sim_options:SimulationOptions=SimulationOptions(),
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
        if not self.exist_plk:
            logger.info(f"Not exists pkl")
            self.modify_f90_file()
            self.compile_run()
            self.get_time()
            self.get_potencial()
            self.get_wavefunction()
            self.get_oscstr()
            self.get_photocurrent()
            sample_data = self.get_sample_data()
            
            with open(self.pkl_path, 'wb') as pickle_file:
                pickle.dump(sample_data, pickle_file)
        else:
            if self.sim_options.force_simulation:
                logger.info(f"Force simulation")
                self.modify_f90_file()
                self.compile_run()
            if self.sim_options.force_parser:
                logger.info(f"Force parser")
                self.get_time()
                self.get_potencial()
                self.get_wavefunction()
                self.get_oscstr()
                self.get_photocurrent()
                sample_data = self.get_sample_data()
                with open(self.pkl_path, 'wb') as pickle_file:
                    pickle.dump(sample_data, pickle_file)
            else:
                with open(self.pkl_path, 'rb') as pickle_file:
                    sample_data = pickle.load(pickle_file)
                
        return sample_data
        
    def get_sample_data(self):
        # inicializa
        sample_data = SampleData()
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
        
        self.sample_data = sample_data
        return sample_data

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

class Plotter:
    def __init__(self, sample_data:SampleData):
        self.sample_data = sample_data
    
    def plot_simulation(self):
        font_base = 12
        font_axis_number = font_base
        font_axis_text = font_base + 2
        font_subtitle = font_base + 2
        font_title = font_base + 4
        
        
        fig, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [8, 5]}, figsize=(10, 6))
        plt.subplots_adjust(wspace = 0.0, top = 0.84, left=0.08)
        lim_E_min, lim_E_max = -50, 700
        
        ax1 = Plotter.plot_structure(ax1,
                            self.sample_data.x_potencial,
                            self.sample_data.y_potencial,
                            self.sample_data.autoenergias,
                            self.sample_data.wavefunctions,
                            self.sample_data.max_e_oscstr_index,
                            lim_E_min,
                            lim_E_max)
        
        if self.sample_data.sim_options.include_os:
            ax2 = Plotter.plot_osc(ax_graph = ax2,
                        oscstr = self.sample_data.oscstr,
                        oscstr_e = self.sample_data.oscstr_e,
                        E0=self.sample_data.E0,
                        lim_E_min=lim_E_min,
                        lim_E_max=lim_E_max,
                        max_e_oscstr_index=self.sample_data.max_e_oscstr_index)
            axes_ = [ax1, ax2]
            
        if self.sample_data.sim_options.include_pc:
            ax3 = ax2.twiny()
            ax3 = Plotter.plot_pc(ax_graph = ax3,
                            pc = self.sample_data.pc,
                            pc_e = self.sample_data.pc_e,
                            E0=self.sample_data.E0,
                            lim_E_min=lim_E_min,
                            lim_E_max=lim_E_max,
                            max_abs_photocurrent = self.sample_data.max_abs_photocurrent,
                            max_e_abs_photocurrent = self.sample_data.max_e_abs_photocurrent,
                            min_abs_photocurrent = self.sample_data.min_abs_photocurrent,
                            min_e_abs_photocurrent = self.sample_data.min_e_abs_photocurrent)
            
            # para deixar ax2 e ax3 com o mesmo lugar no 0
            ax2_xlim = list(ax2.get_xlim())
            ax3_xlim = list(ax3.get_xlim())
            ax2_xlim[0] = (ax3_xlim[0] * ax2_xlim[1]) / ax3_xlim[1]
            ax2.set(xlim=ax2_xlim)
            ax3.set(xlim=ax3_xlim)
            axes_ = [ax1, ax2, ax3]
            
            if not self.sample_data.sim_options.include_os:
                # ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
                ax2.tick_params(axis='x', which='both', top=False, bottom=False, labeltop=False, labelbottom=False)
                ax2.set_ylabel('ΔE (meV)', fontdict=None, labelpad=0)
                ax2.yaxis.tick_right()
                ax2.yaxis.set_label_position("right")
                axes_ = [ax1, ax2]
            
        # set main title
        lw = self.sample_data.LLQW
        lqw = self.sample_data.LQW
        lqb = self.sample_data.LQB
        de = self.sample_data.MQW
        rw = self.sample_data.RLQW
        rqw = self.sample_data.RQW
        rqb = self.sample_data.RQB
        
        title_es = "      Left: {:2.0f} - QW: {:4.1f} nm - QB: {:4.1f} nm".format(lw, lqw, lqb)
        title_es += "\n     Right: {:2.0f} - QW: {:4.1f} nm - QB: {:4.1f} nm".format(rw, rqw, rqb)
        title_es += "\n  Central (Main) QW: {:4.1f} nm".format(de)
        if self.sample_data.sim_options.include_os:
            title_es += "\nOsc Str: {:9.2f} || OS energy: {:02.1f} (meV)".format(self.sample_data.max_e_oscstr, self.sample_data.max_e_transition)

        if self.sample_data.sim_options.include_pc:
            title_es += "\nmax  PC: {:9.2e} || PC energy: {:02.1f} (meV)".format(self.sample_data.max_abs_photocurrent, self.sample_data.max_e_abs_photocurrent)
            # title_es += "\nmin PC: {:.2e} || PC energy: {:02.1f} (meV)".format(sample_data.min_abs_photocurrent, sample_data.min_e_abs_photocurrent)

        fig.suptitle(title_es, x=0.1, fontsize=10, family='monospace', ha='left')
        
        for ax in axes_:
            for item in ([ax.xaxis.label, ax.yaxis.label]):
                item.set_fontsize(font_axis_text)

            for item in ([ax.title]):
                item.set_fontsize(font_subtitle)

            for item in ([ax.yaxis.offsetText]):
                item.set_fontsize(font_axis_number)

            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(font_axis_number)

            ax.title.set_fontsize(font_title)
            
            
        # set img name
        if self.sample_data.sim_options.plot_name == 'id':
            plt.savefig(f'temp_files/{self.sample_data.sample_id}.png', dpi=150)
        elif self.sample_data.sim_options.plot_name == None:
            pass
        else:
            plt.savefig(f'temp_files/{self.sample_data.sim_options.plot_name}.png', dpi=150)
            print(f'temp_files/{self.sample_data.sim_options.plot_name}.png')
        
        # clear plots
        plt.cla()               # clears an axis, i.e. the currently active axis in the current figure. It leaves the other axes untouched.
        plt.clf()               # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
        plt.close()             # closes a window, which will be the current window, if not specified otherwise. 
        plt.close('all') 
        del(fig)
    
    def plot_comparison_simulation(self):
        font_base = 12
        font_axis_number = font_base
        font_axis_text = font_base + 2
        font_subtitle = font_base + 2
        font_title = font_base + 4
        
        
        fig, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [8, 5]}, figsize=(10, 6))
        plt.subplots_adjust(wspace = 0.0, top = 0.84, left=0.08)
        lim_E_min, lim_E_max = -50, 700
        
        ax1 = Plotter.plot_structure(ax1,
                            self.sample_data.x_potencial,
                            self.sample_data.y_potencial,
                            self.sample_data.autoenergias,
                            self.sample_data.wavefunctions,
                            self.sample_data.max_e_oscstr_index,
                            lim_E_min,
                            lim_E_max)
        
        if self.sample_data.sim_options.include_os:
            ax2 = Plotter.plot_osc(ax_graph = ax2,
                        oscstr = self.sample_data.oscstr,
                        oscstr_e = self.sample_data.oscstr_e,
                        E0=self.sample_data.E0,
                        lim_E_min=lim_E_min,
                        lim_E_max=lim_E_max,
                        max_e_oscstr_index=self.sample_data.max_e_oscstr_index)
            axes_ = [ax1, ax2]
            
        if self.sample_data.sim_options.include_pc:
            ax3 = ax2.twiny()
            ax3 = Plotter.plot_pc(ax_graph = ax3,
                            pc = self.sample_data.pc,
                            pc_e = self.sample_data.pc_e,
                            E0=self.sample_data.E0,
                            lim_E_min=lim_E_min,
                            lim_E_max=lim_E_max,
                            max_abs_photocurrent = self.sample_data.max_abs_photocurrent,
                            max_e_abs_photocurrent = self.sample_data.max_e_abs_photocurrent,
                            min_abs_photocurrent = self.sample_data.min_abs_photocurrent,
                            min_e_abs_photocurrent = self.sample_data.min_e_abs_photocurrent)
            
            # para deixar ax2 e ax3 com o mesmo lugar no 0
            ax2_xlim = list(ax2.get_xlim())
            ax3_xlim = list(ax3.get_xlim())
            ax2_xlim[0] = (ax3_xlim[0] * ax2_xlim[1]) / ax3_xlim[1]
            ax2.set(xlim=ax2_xlim)
            ax3.set(xlim=ax3_xlim)
            axes_ = [ax1, ax2, ax3]
            
            if not self.sample_data.sim_options.include_os:
                # ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
                ax2.tick_params(axis='x', which='both', top=False, bottom=False, labeltop=False, labelbottom=False)
                ax2.set_ylabel('ΔE (meV)', fontdict=None, labelpad=0)
                ax2.yaxis.tick_right()
                ax2.yaxis.set_label_position("right")
                axes_ = [ax1, ax2]
            
        # set main title
        lw = self.sample_data.LLQW
        lqw = self.sample_data.LQW
        lqb = self.sample_data.LQB
        de = self.sample_data.MQW
        rw = self.sample_data.RLQW
        rqw = self.sample_data.RQW
        rqb = self.sample_data.RQB
        
        title_es = "      Left: {:2.0f} - QW: {:4.1f} nm - QB: {:4.1f} nm".format(lw, lqw, lqb)
        title_es += "\n     Right: {:2.0f} - QW: {:4.1f} nm - QB: {:4.1f} nm".format(rw, rqw, rqb)
        title_es += "\n  Central (Main) QW: {:4.1f} nm".format(de)
        if self.sample_data.sim_options.include_os:
            title_es += "\nOsc Str: {:9.2f} || OS energy: {:02.1f} (meV)".format(self.sample_data.max_e_oscstr, self.sample_data.max_e_transition)

        if self.sample_data.sim_options.include_pc:
            title_es += "\nmax  PC: {:9.2e} || PC energy: {:02.1f} (meV)".format(self.sample_data.max_abs_photocurrent, self.sample_data.max_e_abs_photocurrent)
            # title_es += "\nmin PC: {:.2e} || PC energy: {:02.1f} (meV)".format(sample_data.min_abs_photocurrent, sample_data.min_e_abs_photocurrent)

        fig.suptitle(title_es, x=0.1, fontsize=10, family='monospace', ha='left')
        
        
        
        if self.sample_data.sim_options.include_os or self.sample_data.sim_options.include_pc:
            sim_pc = self.sample_data.max_abs_photocurrent
            sim_os = self.sample_data.max_e_oscstr
            ref_pc = self.sample_data.sim_options.reference_pc
            ref_os = self.sample_data.sim_options.reference_os
            
            text_lines = []
            # text_lines.append((f"____________________________________", 'black'))
            text_lines.append((f"|       sim  |    ref  |  gain  |", 'black'))
            if self.sample_data.sim_options.include_os:
                gain_os = 100*(sim_os-ref_os)/ref_os
                text_lines.append((f"|OS:    {sim_os:04.3f}|    {ref_os:04.3f}|{gain_os:6.3f} %|", 'red'))
            if self.sample_data.sim_options.include_pc:
                gain_pc = 100*(sim_pc-ref_pc)/ref_pc
                text_lines.append((f"|PC: {sim_pc:.2e}| {ref_pc:.2e}|{gain_pc:6.3f} %|", 'blue'))
                
            # Posición inicial
            x, y = 0.0, -50.0

            # Tamaño de fuente
            fontsize = 10
            
            # Añadir la primera línea de texto para obtener el tamaño
            first_text = ax2.text(x, y, text_lines[0][0], color=text_lines[0][1], fontsize=fontsize, family='monospace', ha='left')

            # Obtener el tamaño de la primera línea de texto
            renderer = fig.canvas.get_renderer()
            bbox = first_text.get_window_extent(renderer=renderer)
            line_height = bbox.height * 2

            # Añadir cada línea de texto con el color correspondiente
            for i, (line, color) in enumerate(text_lines):
                ax2.text(x, y - i * line_height, line, color=color, fontsize=fontsize, family='monospace', ha='left')
                
        
        for ax in axes_:
            for item in ([ax.xaxis.label, ax.yaxis.label]):
                item.set_fontsize(font_axis_text)

            for item in ([ax.title]):
                item.set_fontsize(font_subtitle)

            for item in ([ax.yaxis.offsetText]):
                item.set_fontsize(font_axis_number)

            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(font_axis_number)

            ax.title.set_fontsize(font_title)
            
            
        # set img name
        if self.sample_data.sim_options.plot_name == 'id':
            plt.savefig(f'temp_files/{self.sample_data.sample_id}.png', dpi=150)
        elif self.sample_data.sim_options.plot_name == None:
            pass
        else:
            plt.savefig(f'temp_files/{self.sample_data.sim_options.plot_name}.png', dpi=150)
            print(f'temp_files/{self.sample_data.sim_options.plot_name}.png')
        
        # clear plots
        plt.cla()               # clears an axis, i.e. the currently active axis in the current figure. It leaves the other axes untouched.
        plt.clf()               # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
        plt.close()             # closes a window, which will be the current window, if not specified otherwise. 
        plt.close('all') 
        del(fig)
        
        
            
            
        
        
    def plot_horizontal_simulation(self):
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
        
        ax1 = Plotter.plot_structure(ax1,
                            self.sample_data.x_potencial,
                            self.sample_data.y_potencial,
                            self.sample_data.autoenergias,
                            self.sample_data.wavefunctions,
                            self.sample_data.max_e_oscstr_index,
                            lim_E_min,
                            lim_E_max)
        
        if self.sample_data.sim_options.plot_parameter == ParametersSimulation.OS:
            logger.info(f"plot {ParametersSimulation.OS}")
            ax2 = Plotter.plot_osc_horizontal(ax_graph = ax2,
                        oscstr = self.sample_data.oscstr,
                        oscstr_e = self.sample_data.oscstr_e,
                        E0=self.sample_data.E0,
                        lim_E_min=lim_E_min,
                        lim_E_max=lim_E_max,
                        max_e_oscstr_index=self.sample_data.max_e_oscstr_index)
            axes_ = [ax1, ax2]
        elif self.sample_data.sim_options.plot_parameter == ParametersSimulation.PC:
            logger.info(f"plot {ParametersSimulation.PC}")
            ax3 = ax2.twinx()
            ax3 = Plotter.plot_pc_horizontal(ax_graph = ax3,
                        pc = self.sample_data.pc,
                        pc_e = self.sample_data.pc_e,
                        E0=self.sample_data.E0,
                        lim_E_min=lim_E_min,
                        lim_E_max=lim_E_max,
                        max_abs_photocurrent = self.sample_data.max_abs_photocurrent,
                        max_e_abs_photocurrent = self.sample_data.max_e_abs_photocurrent,
                        min_abs_photocurrent = self.sample_data.min_abs_photocurrent,
                        min_e_abs_photocurrent = self.sample_data.min_e_abs_photocurrent)
            ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            axes_ = [ax1, ax2, ax3]
        elif self.sample_data.sim_options.plot_parameter == ParametersSimulation.OS_PC:
            logger.info(f"plot {ParametersSimulation.OS_PC}")
            ax2 = Plotter.plot_osc_horizontal(ax_graph = ax2,
                        oscstr = self.sample_data.oscstr,
                        oscstr_e = self.sample_data.oscstr_e,
                        E0=self.sample_data.E0,
                        lim_E_min=lim_E_min,
                        lim_E_max=lim_E_max,
                        max_e_oscstr_index=self.sample_data.max_e_oscstr_index)
            
            ax3 = ax2.twinx()
            
            ax3 = Plotter.plot_pc_horizontal(ax_graph = ax3,
                        pc = self.sample_data.pc,
                        pc_e = self.sample_data.pc_e,
                        E0=self.sample_data.E0,
                        lim_E_min=lim_E_min,
                        lim_E_max=lim_E_max,
                        max_abs_photocurrent = self.sample_data.max_abs_photocurrent,
                        max_e_abs_photocurrent = self.sample_data.max_e_abs_photocurrent,
                        min_abs_photocurrent = self.sample_data.min_abs_photocurrent,
                        min_e_abs_photocurrent = self.sample_data.min_e_abs_photocurrent)

            # para deixar ax2 e ax3 com o mesmo lugar no 0
            ax2_ylim = list(ax2.get_ylim())
            ax3_ylim = list(ax3.get_ylim())
            ax2_ylim[0] = (ax3_ylim[0] * ax2_ylim[1]) / ax3_ylim[1]
            ax2.set(ylim=ax2_ylim)
            ax3.set(ylim=ax3_ylim)
            axes_ = [ax1, ax2, ax3]
            

        # Se tem limites em x para o ax1 e ax2
        if self.sample_data.sim_options.lim_x_structure:
            ax1.set(xlim=self.sample_data.sim_options.lim_x_structure)
        if self.sample_data.sim_options.lim_x_pc:
            ax2.set(xlim=self.sample_data.sim_options.lim_x_pc)
            # só deixar ticks de 100 em 100 com minorticks na metade (50)
            ax2.set_xticks(list(range(self.sample_data.sim_options.lim_x_pc[0], self.sample_data.sim_options.lim_x_pc[1]+1, 100)))
            ax2.minorticks_on()
            ax2.set_xticks(list(range(self.sample_data.sim_options.lim_x_pc[0]+50, self.sample_data.sim_options.lim_x_pc[1]+1, 100)), minor=True)
            
        # Definir un formateador para los ticks del eje x
        def format_func(value, tick_number):
            return f'{value:.2f}'
        # Aplicar el formateador al eje x
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
            
        
        

        
        # apaga o primeiro tick se é menor que 0
        axe_ticks_y = ax2.get_yticks()
        for idx, axe_tick in enumerate(axe_ticks_y):
            if axe_tick < 0:
                ax2.yaxis.get_major_ticks()[idx].set_visible(False)

        # grid do ax2
        ax2.grid(axis='x', color='0.70')
        ax2.grid(axis='x', which="minor", color='0.85')
        ax2.grid(axis='y', color='0.70')

        
        # set titles
        ax1.set_title("(a)", x=0.05, y=1.02, size=font_subtitle)
        ax2.set_title('(b)', x=0.05, y=1.02, size=font_subtitle)


        for ax in axes_:
            for item in ([ax.xaxis.label, ax.yaxis.label]):
                item.set_fontsize(font_axis_text)

            for item in ([ax.title]):
                item.set_fontsize(font_subtitle)

            for item in ([ax.yaxis.offsetText]):
                item.set_fontsize(font_axis_number)

            for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(font_axis_number)

            ax.title.set_fontsize(font_title)

        # set img name
        if self.sample_data.sim_options.plot_name == 'id':
            plt.savefig(f'temp_files/{self.sample_data.sample_id}.png', dpi=150)
        elif self.sample_data.sim_options.plot_name == None:
            pass
        else:
            plt.savefig(f'temp_files/{self.sample_data.sim_options.plot_name}.png', dpi=150)
        
        # clear plots
        plt.cla()               # clears an axis, i.e. the currently active axis in the current figure. It leaves the other axes untouched.
        plt.clf()               # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
        plt.close()             # closes a window, which will be the current window, if not specified otherwise. 
        plt.close('all') 
        del(fig)

    @staticmethod
    def plot_structure(ax_graph, x_potencial, y_potencial, autoenergias, wavefunctions, max_e_oscstr_index, lim_E_min, lim_E_max):
        # to plot all wave functions
        for wf in wavefunctions:
            ax_graph.plot(x_potencial, wf, color='#555555', linewidth=0.5)
        
        # plot structure
        ax_graph.plot(x_potencial, y_potencial, color='#5555ff', linewidth=1.0)
        # to fill under the structure
        ax_graph.fill_between(x_potencial, y_potencial, color='#e2e2ff')
        ax_graph.fill_between(x_potencial, -100, color='#e2e2ff')
        
        # # plot WF max under the barrier
        # ax_graph.plot(self.x_nm, self.result_wavefunction[self.max_oscstr_under_index+0][1], color='#ffaaaa', linewidth=1.5)
        
        # plot WF max over the barrier
        max_e_oscstr_index
        ax_graph.plot(x_potencial, wavefunctions[max_e_oscstr_index+1], color='#ff0000', linewidth=1.5)
        # # print(self.result_wavefunction[self.max_oscstr_above_index-0][0] - self.E0)
        
        # plot WF E0
        ax_graph.plot(x_potencial, wavefunctions[0], color='#555555', linewidth=1.5)


        # set limit axes
        lim_x1 = [min(x_potencial), max(x_potencial)]
        lim_y1 = [lim_E_min, lim_E_max]
        
        ax_graph.set(xlabel="Thickness (nm)", ylabel="Energy (meV)")
        ax_graph.set(xlim=lim_x1, ylim=lim_y1)

        return ax_graph
        
    @staticmethod
    def plot_osc(ax_graph, oscstr: np.ndarray = np.array([]),
                oscstr_e: np.ndarray = np.array([]),
                E0=0, lim_E_min=-50,
                lim_E_max=700,
                max_e_oscstr_index=None):
        for _ose, _os in zip(oscstr_e, oscstr):
            ax_graph.plot(_os, _ose, '.', color='red', markersize=4)
        
        if max_e_oscstr_index:
            ax_graph.plot(oscstr[max_e_oscstr_index], oscstr_e[max_e_oscstr_index], '.', color='red', markersize=8)
            
        ax_graph.set_ylabel('ΔE (meV)', fontdict=None, labelpad=0)
        ax_graph.set_xlabel('Oscillator Strength', color="red")
        ax_graph.tick_params(axis='x', labelcolor="red")
        ax_graph.set_ylabel('ΔE (meV)')
        ax_graph.yaxis.tick_right()
        ax_graph.yaxis.set_label_position("right")
        ax_graph.tick_params(axis='x')
        ax_graph.tick_params(axis='y')

        ax_graph.set(ylim=[lim_E_min-E0, lim_E_max-E0])
        return ax_graph

    @staticmethod
    def plot_osc_horizontal(ax_graph, oscstr: np.ndarray = np.array([]),
                            oscstr_e: np.ndarray = np.array([]),
                            E0=0,
                            lim_E_min=-50,
                            lim_E_max=700,
                            max_e_oscstr_index=None):
        for _ose, _os in zip(oscstr_e, oscstr):
            ax_graph.plot(_ose, _os, '.', color='red', markersize=10)
            
        if max_e_oscstr_index:
            ax_graph.plot(oscstr_e[max_e_oscstr_index], oscstr[max_e_oscstr_index], '.', color='red', markersize=15)
            
        # ax_graph.set_title('(b)', x=0.0, y=1)
        ax_graph.set_xlabel('ΔE (meV)')
        ax_graph.set_ylabel('Oscillator strength', color="red", fontdict=None, labelpad=1)
        ax_graph.yaxis.set_label_position("left")
        ax_graph.tick_params(axis='y', labelcolor="red")
        ax_graph.yaxis.tick_left()
            
        return ax_graph

    @staticmethod
    def plot_pc(ax_graph, pc: np.ndarray = np.array([]),
                pc_e: np.ndarray = np.array([]),
                E0=0,
                lim_E_min=-50,
                lim_E_max=700,
                max_abs_photocurrent = None,
                max_e_abs_photocurrent = None,
                min_abs_photocurrent = None,
                min_e_abs_photocurrent = None):
        ax_graph.plot(pc, pc_e, color='blue', linewidth=1.0)
        ax_graph.set_xlabel('Photocurrent (a.u)', color="blue", x=0.5, labelpad=14)
        ax_graph.tick_params(axis='x', labelcolor="blue", labelsize=8)
        ax_graph.xaxis.offsetText.set_fontsize(8)

        ax_graph.set(ylim=[lim_E_min-E0, lim_E_max-E0])
        
        if max_abs_photocurrent and max_e_abs_photocurrent:
            ax_graph.plot(max_abs_photocurrent, max_e_abs_photocurrent, '.', color='blue', markersize=10)
        if min_abs_photocurrent and min_e_abs_photocurrent:
            ax_graph.plot(-min_abs_photocurrent, min_e_abs_photocurrent, '.', color='blue', markersize=10)
        
        return ax_graph

    @staticmethod
    def plot_pc_horizontal(ax_graph, pc: np.ndarray = np.array([]),
                            pc_e: np.ndarray = np.array([]),
                            E0=0,
                            lim_E_min=-50,
                            lim_E_max=700,
                            max_abs_photocurrent = None,
                            max_e_abs_photocurrent = None,
                            min_abs_photocurrent = None,
                            min_e_abs_photocurrent = None):
        
        ax_graph.plot(pc_e, pc, color='blue', linewidth=2.0)
        ax_graph.set_ylabel('Photocurrent intensity (a.u)', color="blue", x=0.5, labelpad=14)
        ax_graph.tick_params(axis='y', labelcolor="blue", labelsize=8)
        ax_graph.yaxis.offsetText.set_fontsize(8)

        ax_graph.set(xlim=[lim_E_min-E0, lim_E_max-E0])
        
        
        pc_e = np.insert(pc_e, 0, 0)
        pc = np.insert(pc, 0, 0)
        
        
        if max_abs_photocurrent and max_e_abs_photocurrent:
            ax_graph.plot(max_e_abs_photocurrent, max_abs_photocurrent, '.', color='blue', markersize=10)
        if min_abs_photocurrent and min_e_abs_photocurrent:
            ax_graph.plot(min_e_abs_photocurrent, -min_abs_photocurrent, '.', color='blue', markersize=10)

        
        return ax_graph


    
if __name__ == '__main__':
    from config import output_fortran_folder
    
    sim_options = SimulationOptions(force_parser=True,
                                    force_simulation=False,
                                    # plot_type=PlotType.COMPLETE,
                                    # plot_type=PlotType.HORIZONTAL,
                                    plot_type=PlotType.COMPARISON,
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
    structure = set_structure_values(individuo[0])
    sample_data = FortranSimulator(structure, sim_options=sim_options, output_folder=f"{output_fortran_folder}").simulate()
    sample_data.plot()
    
    

    
    
    
    
    
    
    
    
    
    # import math
    # import numpy as np
    # import matplotlib.pyplot as plt

    # class PhotocurrentFitnessCalculator:
    #     def __init__(self, max_abs_photocurrent):
    #         self.max_abs_photocurrent = max_abs_photocurrent

    #     def fitness_pc(self):
    #         return (math.log10(abs(self.max_abs_photocurrent))*4) + 50

    # # Rango de max_abs_photocurrent de 10^-14 a 10^-7
    # photocurrent_values = np.logspace(-14, -7, num=100)  # Genera 100 valores en el rango

    # # Calcular fitness_pc para cada valor de max_abs_photocurrent
    # fitness_values = []
    # for value in photocurrent_values:
    #     calculator = PhotocurrentFitnessCalculator(max_abs_photocurrent=value)
    #     fitness_values.append(calculator.fitness_pc())

    # # Graficar los resultados
    # plt.figure(figsize=(10, 6))
    # plt.plot(photocurrent_values, fitness_values, marker='o', linestyle='-', color='b')

    # plt.xscale('log')  # Escala logarítmica en el eje x
    # plt.xlabel('max_abs_photocurrent (log scale)')
    # plt.ylabel('fitness_pc')
    # plt.title('Fitness PC vs. Max Absolute Photocurrent')
    # plt.grid(True)
    # plt.savefig('temp_files/0000.png', dpi=250)