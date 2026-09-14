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
import math
import re

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
from source.geneticalgorithm.GeneticAlgorithm import GeneticAlgorithm as GA
from source.utils.utils_main import *


#suppress warnings
import warnings
warnings.filterwarnings('ignore')

def get_max_fitness(df_results):
    max_gerations = df_results["generation"].max()
    for i in range(max_gerations):
        max_geration_actual = i + 1
        for j in range(max_geration_actual):
            ger_idx = j + 1
            geration = df_results[df_results['generation']==ger_idx]
            max_fitt = geration["fitness"].max()
    ylim_fitt = math.ceil(max_fitt)+1
    return ylim_fitt, max_gerations

def plot_ga(ax_graph, optim_values, max_gerations, ylim_fitt, x_max, y_max, x_mean, y_mean):
    ax_graph.set(xlabel='Generation', ylabel='Fitness')
    ax_graph.yaxis.tick_right()
    ax_graph.yaxis.set_label_position("right")
    ax_graph.set(xlim=[0, max_gerations])
    ax_graph.set(ylim=[0, ylim_fitt])
    if optim_values == '3':
        title_ga = "3 parameters to optimize: wells, barriers and defect thickness\nEqual left and right quantum wells and barriers thickness\n5 left wells and 1 right well"
    elif optim_values == '5':
        title_ga = "5 parameters to optimize:\nwells, barriers and defect nthickness\nAlso left and right quantum wells\n"
    elif optim_values == '7':
        title_ga = "7 parameters to optimize: defect thickness\n quantity of left and right wells\nleft quantum well and quantum barrier thickness\nright quantum well and quantum barrier thickness"
    ax_graph.set_title(title_ga)
    ax_graph.plot(x_max, y_max, color='g', marker='*', linestyle='--', label="Max", markersize=8)
    ax_graph.plot(x_mean, y_mean, color='b', marker='*', linestyle='--', label="Mean", markersize=8)
    ax_graph.legend(loc='lower right')
    
    return ax_graph

def plot_wf(ax_graph, new_sim, lim_x1, lim_y1):
    ax_graph.set(xlabel='Length (nm)', ylabel='Energy (meV)')

    # plot WF
    for E, wf in new_sim.result_wavefunction:
        ax_graph.plot(new_sim.x_nm, wf, color='#aaaaaa', linewidth=0.7)

    # plot structure
    ax_graph.plot(new_sim.x_nm, new_sim.result_potential, color='#5555ff', linewidth=0.8)
    # plot WF max under the barrier
    ax_graph.plot(new_sim.x_nm, new_sim.result_wavefunction[new_sim.max_oscstr_under_index+0][1], color='#ffaaaa', linewidth=1.5)
    # plot WF max over the barrier
    ax_graph.plot(new_sim.x_nm, new_sim.result_wavefunction[new_sim.max_oscstr_above_index+0][1], color='#ff0000', linewidth=1.5)
    # plot WF E0
    ax_graph.plot(new_sim.x_nm, new_sim.result_wavefunction[0][1], color='#555555', linewidth=1.5)

    ax_graph.set(xlabel="Length (nm)", ylabel="Energy (meV)")
    ax_graph.set(xlim=lim_x1, ylim=lim_y1)

    return ax_graph

def plot_os(ax_graph, new_sim, lim_y2):
    # plot Oscillator Strength
    for i in range(1, len(new_sim.result_oscstr)):
        ax_graph.plot(new_sim.result_oscstr[i][0], new_sim.result_oscstr[i][1], '.', color='red', markersize=3)

    ax_graph.set(ylim=lim_y2)
    ax_graph.set_title('(b)', x=0.0, y=1, size=9)
    ax_graph.set_xlabel('OscStr', color="red")
    ax_graph.set_ylabel('ΔE (meV)', fontdict=None, labelpad=0)
    ax_graph.tick_params(axis='x', labelcolor="red")
    ax_graph.yaxis.tick_right()
    ax_graph.yaxis.set_label_position("right")

    return ax_graph
    
def plot_pc(ax_graph, new_sim):
    ax_graph.plot(new_sim.result_pc, new_sim.E_pc, color='blue', linewidth=1.0)
    ax_graph.plot(new_sim.max_abs_photocurrent, new_sim.max_e_abs_photocurrent, '.', color='blue', linewidth=1.0)
    ax_graph.plot(-new_sim.min_abs_photocurrent, new_sim.min_e_abs_photocurrent, '.', color='blue', linewidth=1.0)

    ax_graph.set_xlabel('PC (a.u)', color="blue", x=0.5, labelpad=14)
    ax_graph.tick_params(axis='x', labelcolor="blue", labelsize=8)
    ax_graph.xaxis.offsetText.set_fontsize(8)
    return ax_graph

def get_max_and_min_of_generation(max_geration_actual, df_results):
    x_max, y_max, x_mean, y_mean = [], [], [], []
    for j in range(max_geration_actual):
        ger_idx = j + 1
        # filtra os valores para cada geracao
        geration = df_results[df_results['generation']==ger_idx]
        
        # obtem os maximos e minimos de cada geracao
        idx_max = geration["fitness"].idxmax(axis = 0)
        max_fitt = geration["fitness"].max()
        best_ind_geration = df_results.iloc[idx_max]["individual"]
        min_fitt = geration["fitness"].min()
        mean_fitt = geration["fitness"].mean()
        max_osc = geration["OscStr max"].max()

        x_max.append(ger_idx)
        y_max.append(max_fitt)
        x_mean.append(ger_idx)
        y_mean.append(mean_fitt)
    return best_ind_geration, x_max, y_max, x_mean, y_mean

def create_list_of_simulations_from_ga(df_results, max_gerations, gif_folder):
    best_individuals_generations = []
    for ger_idx in range(max_gerations):
        # pega os valores para cada geracao
        geration = df_results[df_results['generation']==ger_idx+1]
        
        # obtem o maximo de cada geracao
        idx_max = geration["fitness"].idxmax(axis = 0)
        max_fitt = geration["fitness"].max()
        best_ind_geration = df_results.iloc[idx_max]["individual"]

        # cria os dados da estrutura para cada geracao
        best_ind_geration = best_ind_geration.replace('[','').replace(']','').replace(',','').split()
        best_ind_geration = [int(value) for value in best_ind_geration]
        
        simulations_config = {"simulation": best_ind_geration,
                                "wave_function_config": {"pwf_E_initial" : 0,
                                                        "pwf_E_final" : 700,
                                                        "pwf_dE" : 0.1,
                                                        "nn": 100},
                                "photocurrent_config":{"ppc_E_initial" : "auto",
                                                        "ppc_E_final" : "auto",
                                                        "ppc_dE" : 0.1},
                                "output_folder":gif_folder,
        }  
        if simulations_config not in best_individuals_generations:
            best_individuals_generations.append(simulations_config)

    return best_individuals_generations, simulations_config

def run_parallel_simulations(best_individuals_generations):
    cores=cpu_count()
    print("cores: ", cores)
    with Pool(processes=cores) as p:
        with tqdm.tqdm(total=len(best_individuals_generations)) as pbar:
            for i, result in enumerate(p.map(run_simulation_ga, best_individuals_generations)):
                pbar.update()

def parameters_from_csv_path(line):
    re_csv_path = re.compile(r"parameters(?P<parameters>[0-9]*)_gen(?P<generations>[0-9]*)_ind(?P<population>[0-9]*)_mut(?P<mutation>[0-9]*\.[0-9]*)_cro(?P<crossover>[0-9]*\.[0-9]*)")
    if re_csv_path.search(line):
        iter_matches = re_csv_path.finditer(line)
        matches = re_csv_path.findall(line)
        for match_ in iter_matches:
            parameters = match_.group('parameters')
            generations = match_.group('generations')
            population = match_.group('population')
            mutation = match_.group('mutation')
            crossover = match_.group('crossover')
    
    return parameters, generations, population, mutation, crossover


def plot_ga2plots_csv(output_folder, file_csv):
    # take from csv name the number of parameters optimized
    print("starting plot...")

    optim_values, generations, population, mutation, crossover = parameters_from_csv_path(os.path.basename(file_csv))
    optim_name = os.path.basename(file_csv)[:-4]
    
    # load csv file and mgenerations
    df_results = pd.read_csv(file_csv)
    max_gerations = df_results["generation"].max()
    
    # set path and folder gif
    gif_folder = output_folder + "img_gif_ga/"+optim_name+"/"
    gif_images = gif_folder + "images/"
    gif_path   = output_folder + "img_gif_ga/"+optim_name+".gif"
    
    # os.makedirs(gif_folder,exist_ok=True)
    # shutil.rmtree(gif_folder)
    os.makedirs(gif_folder,exist_ok=True)
    
    # get max fitness
    ylim_fitt, max_gerations = get_max_fitness(df_results)
    
    # crear lista de simulaciones
    print("creating simulations list...")
    best_individuals_generations, simulations_config = create_list_of_simulations_from_ga(df_results, max_gerations, gif_folder)
    
    # simular em paralelo
    print("running simulations in parallel...")
    run_parallel_simulations(best_individuals_generations)

    ################### crear graficos
    print("generating plots...")
    for i in range(max_gerations):
        # cria o plot com 3 divisoes (estrutura, pc e oscstr, GA)
        fig, axes = plt.subplots(1, 4, figsize=(16, 8), gridspec_kw={'width_ratios': [6, 2, 2, 8]})
        fig.subplots_adjust(wspace = 0)
        
        # ajusta os valores dos graficos
        ax_wf = axes[0]
        ax_os = axes[1]
        ax_pc = ax_os.twiny()
        axes[2].remove()
        ax_ga = axes[3]

        # aumenta 1 no valor da geracao
        max_geration_actual = i + 1
        best_ind_geration, x_max, y_max, x_mean, y_mean = get_max_and_min_of_generation(max_geration_actual, df_results)
        
        ax_ga = plot_ga(ax_ga, optim_values, max_gerations, ylim_fitt, x_max, y_max, x_mean, y_mean)
        
        # cria os dados da estrutura para cada geracao
        best_ind_geration = best_ind_geration.replace('[','').replace(']','').replace(',','').split()
        best_ind_geration = [int(value) for value in best_ind_geration]
        real_structure = set_structure_values(best_ind_geration)


        lw, lqw, lqb, de, rw, rqw, rqb = real_structure
        wave_function_config = simulations_config["wave_function_config"]
        photocurrent_config = simulations_config["photocurrent_config"]

        simulation_type = "woi"
        new_sim = QBMD_simulator(real_structure,
                                simulation_type = simulation_type,
                                pwf=wave_function_config,
                                ppc=photocurrent_config,
                                output_folder=gif_folder)
        if not new_sim.exist_pkl:
            new_sim.calculate_structure_limits()
            new_sim.calculate_structure_steps()
            new_sim.create_structure_array()
            new_sim.potencial()
            new_sim.wavefunction_numerov()
            new_sim.oscillator_strength()
            new_sim.pc()
            new_sim.save_pkl()
        else:
            infile = open(new_sim.pkl_path,'rb')
            new_sim = pickle.load(infile)
        new_sim.simulation_type = simulation_type
            
        new_sim.process_results()
        new_sim.fitness_function(prints=False)
        
        ############################## plots
        lim_E_min, lim_E_max = -50, 700
        lim_x1 = [min(new_sim.x_nm), max(new_sim.x_nm)]
        lim_y1 = [lim_E_min, lim_E_max]
        lim_y2 = [lim_E_min-new_sim.E0, lim_E_max-new_sim.E0]

        ax_wf = plot_wf(ax_wf, new_sim, lim_x1, lim_y1)
        ax_os = plot_os(ax_os, new_sim, lim_y2)
        ax_pc = plot_pc(ax_pc, new_sim)
        
        # set main title
        title_es = "Left || {:02.0f} - QW: {:04.1f} nm - QB: {:04.1f} nm".format(lw, lqw, lqb) + \
            "\nRight || {:02.0f} - QW: {:04.1f} nm - QB: {:04.1f} nm".format(rw, rqw, rqb) + \
            "\nCentral (main) QW: {:04.1f} nm\n".format(de) + \
            "max1 PC: {:.2e} || PC energy: {:02.1f} (meV)\n".format(new_sim.max_abs_photocurrent, new_sim.max_e_abs_photocurrent) + \
            "max2 PC: {:.2e} || PC energy: {:02.1f} (meV)\n".format(new_sim.min_abs_photocurrent, new_sim.min_e_abs_photocurrent) + \
            "OscStr: {:.2f} - E: {:02.1f} (meV)".format(new_sim.max_oscstr_above_barrier, new_sim.max_e_oscstr_above_barrier)

        fig.suptitle(title_es, x=0.25, fontsize=8)

        # set titles
        ax_wf.set_title("(a)", x=0.0, y=1.1, size=9)
        ax_os.set_title('(b)', x=0.0, y=1.1, size=9)
        ax_os.yaxis.set_label_position("right")

        path_file = gif_images + str(int(max_geration_actual)).zfill(4) + ".png"
        fig.savefig(path_file, dpi=250, bbox_inches='tight')
        plt.cla()               # clears an axis, i.e. the currently active axis in the current figure. It leaves the other axes untouched.
        plt.clf()               # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
        plt.close()             # closes a window, which will be the current window, if not specified otherwise. 
        plt.close('all')
        


    if True:
        print("generating gif")
        png_files = glob.glob(gif_images+"*.png")
        png_files.sort()
        with imageio.get_writer(gif_path, mode='I') as writer:
            for filename in png_files:
                image = imageio.imread(filename)
                writer.append_data(image)
            writer.append_data(image)
            writer.append_data(image)


