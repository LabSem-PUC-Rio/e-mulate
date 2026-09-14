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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import numpy as np

from source.fortran_simulation.DataEnums import PlotType, ParametersSimulation

class Plotter:
    def __init__(self, sample_data):
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
            
            # para mostrar los de PC menos frecuentes
            ax3_x_ticks = ax3.get_xticks()
            while ax3_x_ticks[0] < 0:
                ax3_x_ticks = ax3_x_ticks[1:]
            ax3.set_xticks(ax3_x_ticks[::2])  # Cada segundo valor en el eje x
            
            # para deixar ax2 e ax3 com o mesmo lugar no 0
            ax2_xlim = list(ax2.get_xlim())
            ax3_xlim = list(ax3.get_xlim())
            ax2_xlim[0] = (ax3_xlim[0] * ax2_xlim[1]) / ax3_xlim[1]
            ax2.set(xlim=ax2_xlim)
            ax3.set(xlim=ax3_xlim)
            axes_ = [ax1, ax2, ax3]
            
            if not self.sample_data.sim_options.include_os:
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
                
        # Se tem limites em x para o ax1 e ax2
        if self.sample_data.sim_options.lim_x_structure:
            ax1.set(xlim=self.sample_data.sim_options.lim_x_structure)
            
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
        # self.sample_data.sim_options
        names_sub_plots = self.sample_data.sim_options.extra_info["names_sub_plots"]
        
        font_base = 20
        font_axis_number = font_base
        font_axis_text = font_base + 2
        font_subtitle = font_base + 2
        font_title = font_base + 4
        
        # fig, (ax1, ax2) = plt.subplots(1,2,
        #                                 gridspec_kw={'width_ratios': [8, 5]},
        #                                 figsize=(12, 6))
        
        fig, (ax1, ax2) = plt.subplots(2, 1,
                                    gridspec_kw={'height_ratios': [5, 5]},
                                    # figsize=(8, 9))
                                    figsize=(8, 9))
        
        plt.subplots_adjust(hspace = 0.30,
                            top = 0.95, bottom = 0.08,
                            left = 0.15, right = 0.82)
        
            
        lim_E_min, lim_E_max = -50, 600
        
        ax1 = Plotter.plot_structure(ax1,
                            self.sample_data.x_potencial,
                            self.sample_data.y_potencial,
                            self.sample_data.autoenergias,
                            self.sample_data.wavefunctions,
                            self.sample_data.max_e_oscstr_index,
                            lim_E_min,
                            lim_E_max)
        
        # Obtener los ticks actuales del eje x
        current_ticks = ax1.get_xticks()

        # Multiplicar los ticks actuales por 1.5
        new_ticks = current_ticks * 1.

        # Establecer los nuevos ticks en el eje x
        ax1.set_xticks(new_ticks)
        
        ax1.set_yticks([0, 250, 500])
        # ax1.set_yticks([0, 200, 400, 600])
        
        if self.sample_data.sim_options.plot_parameter == ParametersSimulation.OS:
            # logger.info(f"plot {ParametersSimulation.OS}")
            ax2 = Plotter.plot_osc_horizontal(ax_graph = ax2,
                        oscstr = self.sample_data.oscstr,
                        oscstr_e = self.sample_data.oscstr_e,
                        E0=self.sample_data.E0,
                        lim_E_min=lim_E_min,
                        lim_E_max=lim_E_max,
                        max_e_oscstr_index=self.sample_data.max_e_oscstr_index)
            axes_ = [ax1, ax2]
        elif self.sample_data.sim_options.plot_parameter == ParametersSimulation.PC:
            # logger.info(f"plot {ParametersSimulation.PC}")
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
            # logger.info(f"plot {ParametersSimulation.OS_PC}")
            ax2 = Plotter.plot_osc_horizontal(ax_graph = ax2,
                        oscstr = self.sample_data.oscstr,
                        oscstr_e = self.sample_data.oscstr_e,
                        E0=self.sample_data.E0,
                        lim_E_min=lim_E_min,
                        lim_E_max=lim_E_max,
                        max_e_oscstr_index=self.sample_data.max_e_oscstr_index)
            
            ax3 = ax2.twinx()
            
            if self.sample_data.sim_options.plot_name == 'paper_00_base':
                path_pc = "/home/joseruiz/codes/QBMD_GA/reference_data/2024_QBMD_GA_paper/files/sim_base/Photocurrent_P442_80K_0V.txt"
                path_pc_fit = "/home/joseruiz/codes/QBMD_GA/reference_data/2024_QBMD_GA_paper/files/sim_base/Photocurrent_SL_5x(1.70,6.70)-2.10-1x(1.70,6.70).txt"
                
                def leer_datos_txt(nombre_archivo):
                    x, y = [], []
                    with open(nombre_archivo, 'r') as f:
                        for linea in f:
                            # Ignorar líneas vacías
                            if not linea.strip():
                                continue
                            valores = linea.split()
                            if len(valores) >= 2:
                                x.append(float(valores[0]))
                                y.append(float(valores[1]))
                    return x, y
                
                
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
                
                
                x, y = leer_datos_txt(path_pc)
                ax3.plot(x, y, color="#c510dd", linestyle='--', label="Experimental PC")
                
                
                
                # Obtener handles y labels de ambos ejes
                # handles2, labels2 = ax2.get_legend_handles_labels()
                handles3, labels3 = ax3.get_legend_handles_labels()

                # Crear una sola leyenda combinada
                ax2.legend(
                    handles3,
                    labels3,
                    fontsize=font_base-4,
                    loc="best"
                )
                
                # x, y = leer_datos_txt(path_pc_fit)
                # ax3 = Plotter.plot_pc_horizontal(ax_graph = ax3,
                #         pc = y,
                #         pc_e = x,
                #         E0=self.sample_data.E0,
                #         lim_E_min=lim_E_min,
                #         lim_E_max=lim_E_max)
                # ax3.plot(x, y, color="#dd8410")
                
                
            else:
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
        ax1.set_title(names_sub_plots[0], x=0.05, y=1.02, size=font_subtitle)
        ax2.set_title(names_sub_plots[1], x=0.05, y=1.02, size=font_subtitle)


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

        
        
        
        #########################################################################
        # plots paper
        if self.sample_data.sim_options.plot_name == 'paper_00_base':
            print("-*-*-*-*-*-*-*-")
            
            arrow = True
            x_arrow = 20

            max_osc_transition_e = self.sample_data.oscstr_e[self.sample_data.max_e_oscstr_index]
            E0 = self.sample_data.E0
            print(f"E0: {E0}")
            print(f"max_osc_transition_e: {max_osc_transition_e}")
            max_osc_e = max_osc_transition_e + E0
            style = "Simple, tail_width=0.5, head_width=10, head_length=10"
            kw = dict(arrowstyle=style, color="r")
            arrow = patches.FancyArrowPatch((x_arrow, E0), (x_arrow, max_osc_e),
                                connectionstyle="arc3,rad=0.3", **kw)
            
            ax1.add_patch(arrow)

            e_str = int(round(max_osc_transition_e,0))
            ax1.text(x_arrow+9, ((max_osc_e + E0 + 30)/2), f'{e_str} meV',
                verticalalignment='center', horizontalalignment='left',
                color='r', fontsize=font_base-5, rotation=45)
            
            # horizontal arrow 
            style = "Simple, tail_width=4, head_width=15, head_length=15"
            kw = dict(arrowstyle=style, color="r")

            arrow = patches.FancyArrowPatch(
                (5, 570),
                (50, 570),
                connectionstyle="arc3,rad=0.0",
                **kw
            )

            ax1.add_patch(arrow)
            
            # ax1.text(0, 580, r"$e^{-}$",
            #     verticalalignment='center', horizontalalignment='left',
            #     color='r', fontsize=font_base, rotation=0)
            #####
            
            path_pc = "/home/joseruiz/codes/QBMD_GA/reference_data/2024_QBMD_GA_paper/files/sim_base/Photocurrent_P442_80K_0V.txt"
            path_pc_fit = "/home/joseruiz/codes/QBMD_GA/reference_data/2024_QBMD_GA_paper/files/sim_base/Photocurrent_SL_5x(1.70,6.70)-2.10-1x(1.70,6.70).txt"
            
            def leer_datos_txt(nombre_archivo):
                x, y = [], []
                with open(nombre_archivo, 'r') as f:
                    for linea in f:
                        # Ignorar líneas vacías
                        if not linea.strip():
                            continue
                        valores = linea.split()
                        if len(valores) >= 2:
                            x.append(float(valores[0]))
                            y.append(float(valores[1]))
                return x, y
            
            x, y = leer_datos_txt(path_pc)
            # ax3.plot(x, y, color='#0000ff')
            
            x, y = leer_datos_txt(path_pc_fit)
            # ax3.plot(x, y, color='#0000ff')
            
            
            
        #########################################################################
        # plots paper
        
        def cauchy_distribution(x, x0, gamma):
            """
            Cauchy distribution (Lorentz distribution) probability density function (PDF).
            
            :param x: The value at which to evaluate the PDF.
            :param x0: Location parameter.
            :param gamma: Scale parameter.
            :return: The PDF value at x.
            """
            return 1.0 / (np.pi * gamma * (1 + ((x - x0) / gamma)**2))
        
        if self.sample_data.sim_options.plot_name == 'paper_00_base_comparison_01':
            print("-*-* paper_00_base_comparison_01 -*-*-*-*-*-")
            
            
            ####################################
            # get values for simulation
            print()
            ax_graph = ax3
            pc = self.sample_data.pc
            pc_e = self.sample_data.pc_e
            E0=self.sample_data.E0
            max_abs_photocurrent = self.sample_data.max_abs_photocurrent
            max_e_abs_photocurrent = self.sample_data.max_e_abs_photocurrent
            pc_sim = {'e': np.array(pc_e),
                       'pc': np.array(pc)}
            
            
            
            # criamos um array de zeros
            pdf = np.array(pc_sim['pc']) * 0
            for e, pc in zip(pc_sim['e'], pc_sim['pc']):
                partial_pdf = cauchy_distribution(pc_sim['e'], e, (0.06*e))
                partial_pdf *= pc
                pdf += partial_pdf
            max_pdf = max(pdf)
            pdf = pdf/max_pdf
            
            pc_sim['pc_06'] = [x*max_abs_photocurrent for x in pdf]
            ax3.plot(pc_sim['e'], pc_sim['pc_06'], label="Lorentzian\nfitted\ncurve", color='#00aa00')
            
            
            
            # read measure
            with open('reference_data/2024_QBMD_GA_paper/files/real_Photocurrent_P458_77K_P0.0V.txt', 'r') as real_txt:
                lines_real = real_txt.readlines()
            
            pc_real = {'e': [],
                       'pc': []}
            for line in lines_real:
                _e, _pc = line.split()
                pc_real['e'].append(float(_e))
                pc_real['pc'].append(float(_pc))
                
            pc_real['pc'] = [x*max_abs_photocurrent / max(pc_real['pc']) for x in pc_real['pc']]
            ax3.plot(pc_real['e'], pc_real['pc'], label="Measured", color='#ff0000')
            ax3.legend(fontsize=font_base-9)
            
            


                
                
                
            
            
                
            
            
            
            
            
            
            
            
            
        
        # set img name
        if self.sample_data.sim_options.plot_name == 'id':
            plt.savefig(f'temp_files/{self.sample_data.sample_id}.png', dpi=150)
        elif self.sample_data.sim_options.plot_name == None:
            pass
        else:
            plt.savefig(f'temp_files/{self.sample_data.sim_options.plot_name}.png', dpi=300)
        
        # clear plots
        plt.cla()               # clears an axis, i.e. the currently active axis in the current figure. It leaves the other axes untouched.
        plt.clf()               # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
        plt.close()             # closes a window, which will be the current window, if not specified otherwise. 
        plt.close('all') 
        del(fig)
        
    def plot_horizontal_simulation_org(self):
        font_base = 20
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
        
            
        lim_E_min, lim_E_max = -50, 600
        
        ax1 = Plotter.plot_structure(ax1,
                            self.sample_data.x_potencial,
                            self.sample_data.y_potencial,
                            self.sample_data.autoenergias,
                            self.sample_data.wavefunctions,
                            self.sample_data.max_e_oscstr_index,
                            lim_E_min,
                            lim_E_max)
        
        # Obtener los ticks actuales del eje x
        current_ticks = ax1.get_xticks()

        # Multiplicar los ticks actuales por 1.5
        new_ticks = current_ticks * 1.

        # Establecer los nuevos ticks en el eje x
        ax1.set_xticks(new_ticks)
        
        ax1.set_yticks([0, 250, 500])
        # ax1.set_yticks([0, 200, 400, 600])
        
        if self.sample_data.sim_options.plot_parameter == ParametersSimulation.OS:
            # logger.info(f"plot {ParametersSimulation.OS}")
            ax2 = Plotter.plot_osc_horizontal(ax_graph = ax2,
                        oscstr = self.sample_data.oscstr,
                        oscstr_e = self.sample_data.oscstr_e,
                        E0=self.sample_data.E0,
                        lim_E_min=lim_E_min,
                        lim_E_max=lim_E_max,
                        max_e_oscstr_index=self.sample_data.max_e_oscstr_index)
            axes_ = [ax1, ax2]
        elif self.sample_data.sim_options.plot_parameter == ParametersSimulation.PC:
            # logger.info(f"plot {ParametersSimulation.PC}")
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
            # logger.info(f"plot {ParametersSimulation.OS_PC}")
            ax2 = Plotter.plot_osc_horizontal(ax_graph = ax2,
                        oscstr = self.sample_data.oscstr,
                        oscstr_e = self.sample_data.oscstr_e,
                        E0=self.sample_data.E0,
                        lim_E_min=lim_E_min,
                        lim_E_max=lim_E_max,
                        max_e_oscstr_index=self.sample_data.max_e_oscstr_index)
            
            ax3 = ax2.twinx()
            
            if self.sample_data.sim_options.plot_name == 'paper_00_base':
                path_pc = "/home/joseruiz/codes/QBMD_GA/reference_data/2024_QBMD_GA_paper/files/sim_base/Photocurrent_P442_80K_0V.txt"
                path_pc_fit = "/home/joseruiz/codes/QBMD_GA/reference_data/2024_QBMD_GA_paper/files/sim_base/Photocurrent_SL_5x(1.70,6.70)-2.10-1x(1.70,6.70).txt"
                
                def leer_datos_txt(nombre_archivo):
                    x, y = [], []
                    with open(nombre_archivo, 'r') as f:
                        for linea in f:
                            # Ignorar líneas vacías
                            if not linea.strip():
                                continue
                            valores = linea.split()
                            if len(valores) >= 2:
                                x.append(float(valores[0]))
                                y.append(float(valores[1]))
                    return x, y
                
                
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
                
                
                x, y = leer_datos_txt(path_pc)
                ax3.plot(x, y, color="#c510dd")
                
                
                
                
                
                # x, y = leer_datos_txt(path_pc_fit)
                # ax3 = Plotter.plot_pc_horizontal(ax_graph = ax3,
                #         pc = y,
                #         pc_e = x,
                #         E0=self.sample_data.E0,
                #         lim_E_min=lim_E_min,
                #         lim_E_max=lim_E_max)
                # ax3.plot(x, y, color="#dd8410")
                
                
            else:
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

        
        
        
        #########################################################################
        # plots paper
        if self.sample_data.sim_options.plot_name == 'paper_00_base':
            print("-*-*-*-*-*-*-*-")
            
            arrow = True
            x_arrow = 20

            max_osc_transition_e = self.sample_data.oscstr_e[self.sample_data.max_e_oscstr_index]
            E0 = self.sample_data.E0
            print(f"E0: {E0}")
            print(f"max_osc_transition_e: {max_osc_transition_e}")
            max_osc_e = max_osc_transition_e + E0
            style = "Simple, tail_width=0.5, head_width=10, head_length=10"
            kw = dict(arrowstyle=style, color="r")
            arrow = patches.FancyArrowPatch((x_arrow, E0), (x_arrow, max_osc_e),
                                connectionstyle="arc3,rad=0.3", **kw)
            
            ax1.add_patch(arrow)

            e_str = int(round(max_osc_transition_e,0))
            ax1.text(x_arrow+9, ((max_osc_e + E0 + 30)/2), f'{e_str} meV',
                verticalalignment='center', horizontalalignment='left',
                color='r', fontsize=font_base-5, rotation=45)
            
            path_pc = "/home/joseruiz/codes/QBMD_GA/reference_data/2024_QBMD_GA_paper/files/sim_base/Photocurrent_P442_80K_0V.txt"
            path_pc_fit = "/home/joseruiz/codes/QBMD_GA/reference_data/2024_QBMD_GA_paper/files/sim_base/Photocurrent_SL_5x(1.70,6.70)-2.10-1x(1.70,6.70).txt"
            
            def leer_datos_txt(nombre_archivo):
                x, y = [], []
                with open(nombre_archivo, 'r') as f:
                    for linea in f:
                        # Ignorar líneas vacías
                        if not linea.strip():
                            continue
                        valores = linea.split()
                        if len(valores) >= 2:
                            x.append(float(valores[0]))
                            y.append(float(valores[1]))
                return x, y
            
            x, y = leer_datos_txt(path_pc)
            # ax3.plot(x, y, color='#0000ff')
            
            x, y = leer_datos_txt(path_pc_fit)
            # ax3.plot(x, y, color='#0000ff')
            
            
            
        #########################################################################
        # plots paper
        
        def cauchy_distribution(x, x0, gamma):
            """
            Cauchy distribution (Lorentz distribution) probability density function (PDF).
            
            :param x: The value at which to evaluate the PDF.
            :param x0: Location parameter.
            :param gamma: Scale parameter.
            :return: The PDF value at x.
            """
            return 1.0 / (np.pi * gamma * (1 + ((x - x0) / gamma)**2))
        
        if self.sample_data.sim_options.plot_name == 'paper_00_base_comparison_01':
            print("-*-* paper_00_base_comparison_01 -*-*-*-*-*-")
            
            
            ####################################
            # get values for simulation
            print()
            ax_graph = ax3
            pc = self.sample_data.pc
            pc_e = self.sample_data.pc_e
            E0=self.sample_data.E0
            max_abs_photocurrent = self.sample_data.max_abs_photocurrent
            max_e_abs_photocurrent = self.sample_data.max_e_abs_photocurrent
            pc_sim = {'e': np.array(pc_e),
                       'pc': np.array(pc)}
            
            
            
            # criamos um array de zeros
            pdf = np.array(pc_sim['pc']) * 0
            for e, pc in zip(pc_sim['e'], pc_sim['pc']):
                partial_pdf = cauchy_distribution(pc_sim['e'], e, (0.06*e))
                partial_pdf *= pc
                pdf += partial_pdf
            max_pdf = max(pdf)
            pdf = pdf/max_pdf
            
            pc_sim['pc_06'] = [x*max_abs_photocurrent for x in pdf]
            ax3.plot(pc_sim['e'], pc_sim['pc_06'], label="Lorentzian\nfitted\ncurve", color='#00aa00')
            
            
            
            # read measure
            with open('reference_data/2024_QBMD_GA_paper/files/real_Photocurrent_P458_77K_P0.0V.txt', 'r') as real_txt:
                lines_real = real_txt.readlines()
            
            pc_real = {'e': [],
                       'pc': []}
            for line in lines_real:
                _e, _pc = line.split()
                pc_real['e'].append(float(_e))
                pc_real['pc'].append(float(_pc))
                
            pc_real['pc'] = [x*max_abs_photocurrent / max(pc_real['pc']) for x in pc_real['pc']]
            ax3.plot(pc_real['e'], pc_real['pc'], label="Measured", color='#ff0000')
            ax3.legend(fontsize=font_base-9)
            
            


                
                
                
            
            
                
            
            
            
            
            
            
            
            
            
        
        # set img name
        if self.sample_data.sim_options.plot_name == 'id':
            plt.savefig(f'temp_files/{self.sample_data.sample_id}.png', dpi=150)
        elif self.sample_data.sim_options.plot_name == None:
            pass
        else:
            plt.savefig(f'temp_files/{self.sample_data.sim_options.plot_name}.png', dpi=250)
        
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
        ax_graph.plot(x_potencial, wavefunctions[max_e_oscstr_index+1], color='#ff0000', linewidth=2.5)
        # # print(self.result_wavefunction[self.max_oscstr_above_index-0][0] - self.E0)
        
        # plot WF E0
        # ax_graph.plot(x_potencial, wavefunctions[0], color='#555555', linewidth=1.5)
        ax_graph.plot(x_potencial, wavefunctions[0], color='#222222', linewidth=2.5)


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
            ax_graph.plot(oscstr_e[max_e_oscstr_index], oscstr[max_e_oscstr_index], '.', color='red', markersize=15, label="Oscillator strength")
            
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
        
        # ax_graph.plot(pc_e, pc, color='blue', linewidth=2.0, label="Simulated PC")
        ax_graph.plot(pc_e, pc, color='blue', linewidth=2.0)
        ax_graph.set_ylabel('Photocurrent intensity (a.u)', color="blue", x=0.5, labelpad=14)
        ax_graph.set_ylabel('Normalized Photocurrent', color="blue", x=0.5, labelpad=14)
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


