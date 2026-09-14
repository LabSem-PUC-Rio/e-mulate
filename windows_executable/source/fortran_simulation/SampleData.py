#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jose Ruiz
@modified_date: 2024-08-24
"""

import sys, os

# Obtén la ruta absoluta del archivo actual
current_file = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.append(project_path)

from enum import Enum
import numpy as np
import pandas as pd
import math
from dataclasses import dataclass, field
from typing import Optional, List

from source.fortran_simulation.logger import logger
from source.fortran_simulation.Plotter import Plotter
from source.fortran_simulation.DataEnums import PlotType, ParametersSimulation
from source.fortran_simulation.SimulationOptions import SimulationOptions
   
@dataclass
class SampleData:
    sim_options: SimulationOptions = field(default_factory=SimulationOptions)
    total_time: float = 0
    individual: List = field(default_factory=list)
    structure: List = field(default_factory=list)
    
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
            self.fitness = self.max_e_oscstr * 20
            if not valid_energy(self.max_e_transition):
                self.fitness = 0
        elif self.sim_options.otimization_parameter == "pc":
            if self.max_abs_photocurrent > 0:
                self.fitness = (math.log10(abs(self.max_abs_photocurrent))*4) + 50
                if not valid_energy(self.max_e_abs_photocurrent):
                    self.fitness = 0
        elif self.sim_options.otimization_parameter == "ospc":
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

    def result_dictionary(self):
        self.result_dictionary = {"individual" : self.individual,
                            "fitness" : self.fitness,
                            "structure" : self.structure,
                            "E0" : self.E0,
                            "pc1" : self.max_abs_photocurrent,
                            "pc1e" : self.max_e_abs_photocurrent,
                            "pc2" : self.min_abs_photocurrent,
                            "pc2e" : self.min_e_abs_photocurrent,
                            # "oscstr_above" : self.max_oscstr_above_barrier,
                            # "oscstre_above" : self.max_e_oscstr_above_barrier,
                            # "oscstr_under" : self.max_oscstr_under_barrier,
                            # "oscstre_under" : self.max_e_oscstr_under_barrier,
                            "oscstr" : self.max_e_oscstr,
                            "oscstre" : self.max_e_transition,
                            "time": self.total_time,
                            
                            
    "lw" : self.LLQW,
    "lqw" : self.LQW,
    "lqb" : self.LQB,
    "c" : self.MQW,
    "rw" : self.RLQW,
    "rqw" : self.RQW,
    "rqb" : self.RQB,
    "id" : self.sample_id,
    
    }
        

    def print_result(self):
        _pc = self.max_abs_photocurrent
        _pc_e = self.max_e_abs_photocurrent
        _os = self.max_e_oscstr
        _os_e = self.max_e_transition
        print(f"OS: {_os:8.4f}     || OS Energy: {_os_e:5.2f} (meV)")
        print(f"PC: {_pc:12.4e} || PC energy: {_pc_e:5.2f} (meV)")
