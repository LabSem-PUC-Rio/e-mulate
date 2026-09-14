# Standard library imports
import os
import sys

# get file path
current_file = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.append(project_path)
import config

# Standard library imports
import datetime
import pickle
import time
import glob
import shutil

# to create the gif files
import imageio

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# multi processing
from multiprocessing import Pool, cpu_count
import tqdm

# Local application/library specific imports
from source.QBMD_simulator.QBMD_simulator import *

global filtered_pd_database

def create_csv(csv_pd_path):
    columns = ["id", "structure", \
                "lw", "lqw", "lqb", "c", "rw", "rqw", "rqb", \
                "fit", "E0", "pc1", "pc1e", "pc2", "pc2e",\
                "oscstr", "oscstre",
                "oscstr_under", "oscstre_under",
                "oscstr_above", "oscstre_above"]
    
    pd_database = pd.DataFrame(columns=columns)
    newdata = {"id": 0,
                "structure": 0,
                "lw": 0,
                "lqw": 0,
                "lqb": 0,
                "c": 0,
                "rw": 0,
                "rqw": 0,
                "rqb": 0,
                "fit": 0,
                "E0": 0,
                "pc1": 0,
                "pc1e": 0,
                "pc2": 0,
                "pc2e": 0,
                "oscstr": 0,
                "oscstre": 0,
                "oscstr_under": 0,
                "oscstre_under": 0,
                "oscstr_above": 0,
                "oscstre_above": 0
            }

    
    # pd_database = pd_database.append(newdata, ignore_index=True)
    pd_database.to_csv(csv_pd_path, index=False)

def run_simulation(individual):
    prints = True
    structure = set_structure_values(individual[0])
    options = individual[1]
    output_folder = options["output_folder"]
    base_folders = options["base_folders"]
    check_in_df = options["check_in_df"]
    simulation_type = options["simulation_type"]
    output_folder = options["output_folder"]

    lw, lqw, lqb, c, rw, rqw, rqb = structure
    id = set_id(structure)
    
    # starting the simulation...
    pwf={"pwf_E_initial" : 0,
        "pwf_E_final" : 700,
        "pwf_dE" : 0.1,
        "nn" : 100}
    ppc={"ppc_E_initial" : "auto",
        "ppc_E_final" : "auto",
        "ppc_dE" : 0.1}
    
    # pwf={"pwf_E_initial" : 250,
    #     "pwf_E_final" : 600,
    #     "pwf_dE" : 1.0,
    #     "nn" : 100}
    # ppc={"ppc_E_initial" : 290,
    #     "ppc_E_final" : 310,
    #     "ppc_dE" : 0.1}


    new_sim = QBMD_simulator(structure, simulation_type = simulation_type, pwf=pwf, ppc=ppc, output_folder=output_folder)
    if not new_sim.exist_pkl:
        if prints : print("iniciando")
        new_sim.calculate_structure_limits()
        new_sim.calculate_structure_steps()
        new_sim.create_structure_array()
        if prints : print("iniciando potencial")
        new_sim.potencial()
        if prints : print("iniciando Eigenstates_Transfer_Matrix")
        new_sim.Eigenstates_Transfer_Matrix_energy()
        new_sim.Eigenstates_Transfer_Matrix_wf()
        if prints : print("iniciando oscillator_strength")
        new_sim.oscillator_strength()
        if prints : print("iniciando pc")
        new_sim.pc()
        new_sim.save_pkl()
    else:
        infile = open(new_sim.pkl_path,'rb')
        new_sim = pickle.load(infile)
    new_sim.simulation_type = simulation_type
    # print(new_sim.E0)
    new_sim.process_results()
    new_sim.fitness_function(prints=False)
    new_sim.finish_dictionary(individual[0])
    if simulation_type == "woi":
        pass
    elif simulation_type.startswith('os_test'):
        new_sim.plot_structure_os()
    elif simulation_type.startswith('paper_test_paper_comparison'):
        new_sim.plot_structure_paper_comparison(options['plot'], options)
    elif simulation_type.startswith('paper_test_'):
        new_sim.plot_structure_paper(options['plot'], options)
    elif simulation_type.startswith('test_'):
        new_sim.plot_structure()
    else:
        new_sim.plot_structure()

    return [new_sim.fitness, new_sim.newdata]

def run_simulation_ga(simulations_config):
    output_folder = config.output_folder
    if type(simulations_config) is list:
        base_simulation = simulations_config[0]
        structure = set_structure_values(simulations_config[0])
        wave_function_config={"pwf_E_initial" : 0,
                            "pwf_E_final" : 700,
                            "pwf_dE" : 0.1,
                            "nn": 100}
        photocurrent_config={"ppc_E_initial" : "auto",
                            "ppc_E_final" : "auto",
                            "ppc_dE" : 0.1}
        ga_config = simulations_config[2]

    if type(simulations_config) is dict:
        base_simulation = simulations_config["simulation"]
        structure = set_structure_values(simulations_config["simulation"])
        if "wave_function_config" in simulations_config:
            wave_function_config = simulations_config["wave_function_config"]
        else:
            wave_function_config={"pwf_E_initial" : 0,
                                "pwf_E_final" : 700,
                                "pwf_dE" : 0.1,
                                "nn": 100}
        
        if "photocurrent_config" in simulations_config:
            photocurrent_config = simulations_config["photocurrent_config"]
        else:
            photocurrent_config={"ppc_E_initial" : "auto",
                                "ppc_E_final" : "auto",
                                "ppc_dE" : 0.1}
        if "output_folder" in simulations_config:
            output_folder= simulations_config["output_folder"]
    
    base_folders = output_folder+"sims/"
    check_in_df = False
    simulation_type = "woi"

    lw, lqw, lqb, c, rw, rqw, rqb = structure
    id = set_id(structure)
    
    columns = ["id", "structure", \
                "lw", "lqw", "lqb", "c", "rw", "rqw", "rqb", \
                "fit", "E0", "pc1", "pc1e", "pc2", "pc2e",\
                "oscstr", "oscstre"]
    
    csv_pd_path = output_folder + "database.csv"
    if not os.path.exists(csv_pd_path):
        create_csv(csv_pd_path)

    new_sim = QBMD_simulator(structure,
                            simulation_type = simulation_type,
                            pwf=wave_function_config,
                            ppc=photocurrent_config,
                            output_folder=output_folder)
    if not new_sim.exist_pkl:
        new_sim.calculate_structure_limits()
        new_sim.calculate_structure_steps()
        new_sim.create_structure_array()
        new_sim.potencial()
        new_sim.Eigenstates_Transfer_Matrix_energy()
        new_sim.Eigenstates_Transfer_Matrix_wf()
        new_sim.oscillator_strength()
        new_sim.pc()
        new_sim.save_pkl()
    else:
        infile = open(new_sim.pkl_path,'rb')
        new_sim = pickle.load(infile)
    new_sim.simulation_type = simulation_type
        
    new_sim.process_results()
    new_sim.fitness_function(prints=False,
                             optimization_parameter=ga_config['optimization_parameter'],
                             target=ga_config['target_energy'])
    new_sim.finish_dictionary(base_simulation)
    
    return [new_sim.fitness, new_sim.result_dictionary]

def create_gif(list_images, gif_path, delete_images=False):
    """
    function to create a gif animation
    list_images: list of images (complete path) to add to gif
    gif_path: complete path to save the gif animation
    delete_images: (boolean), to define if the list_images files will be deleted
        after the creation of the gif
    """
    with imageio.get_writer(gif_path, mode='I') as writer:
        for png_file in list_images:
            image = imageio.imread(png_file)
            writer.append_data(image)
    if delete_images:
        for png_file in list_images:
            os.remove(png_file)

def get_values_from_generation(max_geration_actual, df_results):
    x_max, y_max, x_mean, y_mean, pc = [], [], [], [], []
    for j in range(max_geration_actual):
        ger_idx = j + 1
        # filtra os valores para cada geracao
        geration = df_results[df_results['geracao']==ger_idx]
        
        # obtem os maximos e minimos de cada geracao
        idx_max = geration["aptidao"].idxmax(axis = 0)
        max_fitt = geration["aptidao"].max()
        best_ind_geration = df_results.iloc[idx_max]["individuo"]
        min_fitt = geration["aptidao"].min()
        mean_fitt = geration["aptidao"].mean()
        max_osc = geration["OscStr max"].max()
        max_pc = geration["PC max 1"].max()

        x_max.append(ger_idx)
        y_max.append(max_fitt)
        x_mean.append(ger_idx)
        y_mean.append(mean_fitt)
        pc.append(max_pc)
    return best_ind_geration, x_max, y_max, x_mean, y_mean, pc