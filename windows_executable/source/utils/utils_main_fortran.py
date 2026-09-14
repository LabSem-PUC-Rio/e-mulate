#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jose Ruiz
@modified_date: 2024-08-24
"""

# Standard library imports
import os
import sys

# get file path
current_file = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
print("-------------------*****-------")
print(project_path)
sys.path.append(project_path)
from conf import output_fortran_folder


# Local application/library specific imports
from source.fortran_simulation.FortranSimulator import FortranSimulator
from source.fortran_simulation.SimulationOptions import SimulationOptions
from source.fortran_simulation.utils import set_structure_values
from source.fortran_simulation.DataEnums import PlotType, ParametersSimulation


def run_fortran_simulation(individual):
    sim_options = individual[1]
    
    sample_data = FortranSimulator(individual[0], sim_options=sim_options, output_folder=f"{output_fortran_folder}").simulate()
    
    return [sample_data.fitness, sample_data.result_dictionary]

def run_fortran_simulation_ga(individual):
    if individual[2]['optimization_parameter'] == 'os':
        _plot_parameter = ParametersSimulation.OS
    elif individual[2]['optimization_parameter'] == 'pc':
        _plot_parameter = ParametersSimulation.PC
    elif individual[2]['optimization_parameter'] == 'ospc':
        _plot_parameter = ParametersSimulation.OS_PC
    else:
        _plot_parameter = ParametersSimulation.OS_PC
    
    sim_options = SimulationOptions(force_parser=True,
                                        force_simulation=False,
                                        plot_type=PlotType.COMPLETE,
                                        plot_parameter=_plot_parameter,
                                        otimization_parameter=individual[2]['optimization_parameter'],
                                        )
    
    sample_data = FortranSimulator(individual[0], sim_options=sim_options, output_folder=f"{output_fortran_folder}").simulate()
    
    
    return [sample_data.fitness, sample_data.result_dictionary]


if __name__ == '__main__':
    print('test main simulation')
    
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
    
    individuo=[[5, 20, 70, 25, 1, 20, 70], sim_options]
    run_fortran_simulation(individuo)
    