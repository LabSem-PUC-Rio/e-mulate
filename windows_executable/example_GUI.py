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
project_path = os.path.dirname(os.path.dirname(current_file))
sys.path.append(project_path)

# Local application/library specific imports
from windows_executable.source.utils.utils_main_fortran import run_fortran_simulation #
from windows_executable.source.fortran_simulation.FortranSimulator import FortranSimulator
from windows_executable.source.fortran_simulation.SimulationOptions import SimulationOptions #
from windows_executable.source.fortran_simulation.utils import set_structure_values
from windows_executable.source.fortran_simulation.DataEnums import PlotType, ParametersSimulation





if __name__ == '__main__':
    
    structure = [5, 20, 70, 25, 1, 20, 70]
    
    sim_options = SimulationOptions(force_parser=True,
                                    force_simulation=False,
                                    )
    
    individuo=[structure, sim_options]
    fitness, result_dictionary = run_fortran_simulation(individuo)
    
    print("="*60)
    print(f"Simulación  completada con éxito.")
    print(f"Fitness: {fitness}")
    _structure = result_dictionary['structure']
    _pc = result_dictionary['pc1']
    _pc_e = result_dictionary['pc1e']
    _os = result_dictionary['oscstr']
    _os_e = result_dictionary['oscstre']
    print(f"Estructura: {_structure}")
    print(f"PC: {_pc}, PC_e: {_pc_e}")
    print(f"OS: {_os}, OS_e: {_os_e}")
    print("="*60)

    
