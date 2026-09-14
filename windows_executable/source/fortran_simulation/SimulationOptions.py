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
from typing import Optional

from source.fortran_simulation.DataEnums import PlotType, ParametersSimulation


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
    extra_info: dict = field(default_factory=dict)
    
    # reference
    reference_pc: float = None
    reference_pc_e: float = None
    reference_os: float = None
    reference_os_e: float = None
    
    
    def __post_init__(self):
        param_val = self.plot_parameter.value if isinstance(self.plot_parameter, Enum) else self.plot_parameter
        if param_val in [ParametersSimulation.OS.value, ParametersSimulation.OS_PC.value, ParametersSimulation.OS, ParametersSimulation.OS_PC]:
            self.include_os = True
        
        if param_val in [ParametersSimulation.PC.value, ParametersSimulation.OS_PC.value, ParametersSimulation.PC, ParametersSimulation.OS_PC]:
            self.include_pc = True
            
        if self.include_os and self.include_pc:
            self.otimization_parameter = 'ospc'
        elif self.include_os and not self.include_pc:
            self.otimization_parameter = 'os'
        elif not self.include_os and self.include_pc:
            self.otimization_parameter = 'pc'
        else:
            raise ValueError('no valid otimization parameter')

