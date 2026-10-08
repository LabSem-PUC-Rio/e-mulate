#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEAP Genetic Algorithm implementation for semiconductor superlattice optimization
integrated with photocurrent Fortran simulation engine.

Features:
- Optimization of 3, 5, or 7 parameters
- DEAP genetic operators (crossover, mutation, tournament selection, elitism)
- Integration with photocurrent_sim_windows Fortran simulation and automatic caching
- Multiple optimization objectives (Target Peak Energy, Photocurrent + Oscillator Strength, PC only, OS only)
- Full CSV logging with start timestamp format: YYYYMMDD_HHMMSS_otim_X_parametros.csv
- Resume optimization capability from an existing CSV
- Live callbacks for GUI progress and dynamic plotting
"""

import os
import sys
import time
import math
import ast
import random
import datetime
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import deap
from deap import base, creator, tools
import concurrent.futures
import threading


def parse_seeds_text(text: str, n_params: int, bounds: Optional[List[Tuple[int, int]]] = None) -> List[List[int]]:
    """
    Parses seed structures from string or python literal formats.
    Accepts:
    - Lists of lists: [[20, 70, 25], [30, 80, 20]]
    - Single list: [20, 70, 25]
    - Multiple lines:
        20, 70, 25
        30, 80, 20
    """
    seeds = []
    if not text:
        return seeds
    text = text.strip()
    try:
        val = ast.literal_eval(text)
        if isinstance(val, list):
            if val and isinstance(val[0], list):
                for item in val:
                    if len(item) == n_params:
                        seeds.append([int(round(x)) for x in item])
            elif len(val) == n_params:
                seeds.append([int(round(x)) for x in val])
    except Exception:
        pass

    if not seeds:
        for line in text.splitlines():
            line = line.strip().replace("[", "").replace("]", "").replace(",", " ")
            parts = [p for p in line.split() if p]
            if len(parts) == n_params:
                try:
                    seeds.append([int(round(float(p))) for p in parts])
                except ValueError:
                    pass

    # Ensure all seeds satisfy physical boundary and defect constraints
    active_bounds = bounds or DEFAULT_BOUNDS.get(n_params, None)
    if active_bounds:
        for s in seeds:
            clamp_individual(s, active_bounds)

    return seeds

# Ensure root directory and windows_executable are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
windows_exec_dir = os.path.dirname(os.path.dirname(current_dir))

for p in [project_dir, windows_exec_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from conf import output_fortran_folder
from windows_executable.source.fortran_simulation.utils import set_structure_values
from windows_executable.source.fortran_simulation.SimulationOptions import SimulationOptions
from windows_executable.source.fortran_simulation.FortranSimulator import FortranSimulator


# ---------------------------------------------------------------------------
# Default gene boundaries for 3, 5, and 7 parameter optimization
# Note: thicknesses are represented in tenths of nm (e.g. 20 -> 2.0 nm)
# ---------------------------------------------------------------------------
DEFAULT_BOUNDS = {
    3: [
        (10, 80),   # qw: well thickness (1.0 to 8.0 nm)
        (20, 150),  # qb: barrier thickness (2.0 to 15.0 nm)
        (10, 80),   # defe: central well thickness (1.0 to 8.0 nm)
    ],
    5: [
        (1, 8),     # w_l: left wells count
        (10, 80),   # qw: well thickness (1.0 to 8.0 nm)
        (20, 150),  # qb: barrier thickness (2.0 to 15.0 nm)
        (10, 80),   # defe: central well thickness (1.0 to 8.0 nm)
        (1, 8),     # w_r: right wells count
    ],
    7: [
        (1, 8),     # w_l: left wells count
        (10, 80),   # qw_l: left well thickness (1.0 to 8.0 nm)
        (20, 150),  # qb_l: left barrier thickness (2.0 to 15.0 nm)
        (10, 80),   # defe: central well thickness (1.0 to 8.0 nm)
        (1, 8),     # w_r: right wells count
        (10, 80),   # qw_r: right well thickness (1.0 to 8.0 nm)
        (20, 150),  # qb_r: right barrier thickness (2.0 to 15.0 nm)
    ],
}

CSV_COLUMNS = [
    "generation", "individual", "fitness", "structure",
    "OscStr max", "OscStr energy (meV)",
    "PC max 1", "PC max 1 energy (meV)",
    "PC max 2", "PC max 2 energy (meV)",
    "origin", "parent1", "parent2",
    "left wells", "qw left (nm)", "qb left (nm)",
    "main qw (nm)",
    "right wells", "qw right (nm)", "qb right (nm)",
    "python structure", "time"
]


# DEAP classes setup
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def clamp_individual(ind: list, bounds: List[Tuple[int, int]]):
    """
    Clamps each gene of an individual within its min/max boundaries and enforces the physical constraint:
    MQW (central defect well) must always be greater than or equal to LQW and RQW by 0.5 nm or more (+5 units).
    """
    # 1. Clamping within parameter ranges
    for i, (b_min, b_max) in enumerate(bounds):
        if ind[i] < b_min:
            ind[i] = b_min
        elif ind[i] > b_max:
            ind[i] = b_max

    # 2. Identify well indices based on chromosome length
    n = len(ind)
    if n == 3:
        lqw_idx, rqw_idx, mqw_idx = 0, 0, 2
    elif n == 5:
        lqw_idx, rqw_idx, mqw_idx = 1, 1, 3
    elif n == 7:
        lqw_idx, rqw_idx, mqw_idx = 1, 5, 3
    else:
        return

    mqw_max = bounds[mqw_idx][1]

    # Upper cap on outer wells so that MQW can satisfy the +5 constraint without exceeding mqw_max
    max_allowed_qw = mqw_max - 5
    if ind[lqw_idx] > max_allowed_qw:
        ind[lqw_idx] = max_allowed_qw
    if ind[rqw_idx] > max_allowed_qw:
        ind[rqw_idx] = max_allowed_qw

    # Strictly enforce MQW >= max(LQW, RQW) + 5
    required_mqw = max(ind[lqw_idx], ind[rqw_idx]) + 5
    if ind[mqw_idx] < required_mqw:
        ind[mqw_idx] = required_mqw


def compute_fitness(
    sample_data,
    objective: str = "ospc",
    opt_pc: bool = True,
    opt_os: bool = True,
    limit_energy: bool = True,
    target_energy: float = 300.0,
    target_margin: float = 20.0,
) -> float:
    """
    Computes fitness scalar based on Fortran simulation results and selected objectives.

    Features:
    - Modular optimization of Photocurrent (PC), Oscillator Strength (OS), or both.
    - Optional energy range filtering [target_energy - target_margin, target_energy + target_margin].
    - Continuous proportional penalty: individuals just outside the boundary (e.g., 320.01 meV)
      are NOT discarded; the penalty smoothly scales with distance from the boundary.
    """
    # Backwards compatibility if objective string was passed
    if objective == "pc":
        opt_pc = True
        opt_os = False
    elif objective == "os":
        opt_pc = False
        opt_os = True
    elif objective == "target_energy":
        opt_pc = True
        opt_os = True
        limit_energy = True

    # Fallback if both unselected
    if not opt_pc and not opt_os:
        opt_pc = True
        opt_os = True

    pc_max = sample_data.max_abs_photocurrent
    pc_e = sample_data.max_e_abs_photocurrent
    os_max = sample_data.max_e_oscstr
    os_e = sample_data.max_e_transition

    # Base photocurrent fitness term (logarithmic scaling)
    if pc_max > 0:
        fit_pc = (math.log10(abs(pc_max)) * 4.0) + 50.0
    else:
        fit_pc = 0.0

    # Base oscillator strength fitness term
    fit_os = os_max * 20.0

    base_fit = 0.0
    if opt_pc:
        base_fit += max(0.0, fit_pc)
    if opt_os:
        base_fit += max(0.0, fit_os)

    # Defense-in-depth: Penalize any structure where MQW is not >= max(LQW, RQW) + 0.5 nm
    lqw = getattr(sample_data, "LQW", 0.0)
    rqw = getattr(sample_data, "RQW", 0.0)
    mqw = getattr(sample_data, "MQW", 0.0)
    if mqw > 0 and (lqw > 0 or rqw > 0):
        if mqw < max(lqw, rqw) + 0.499:
            base_fit *= 0.01

    # If energy limitation is disabled, return pure base fitness
    if not limit_energy:
        return float(max(0.1, base_fit))

    # Reference peak energy: photocurrent peak if PC is optimized and simulated, else transition peak
    ref_energy = pc_e if (opt_pc and pc_max > 0) else os_e
    dist = abs(ref_energy - target_energy)
    dist_outside = max(0.0, dist - target_margin)

    if dist_outside <= 0.0:
        # Inside the range [target_energy - target_margin, target_energy + target_margin]
        # Full base fitness + bonus for proximity to central target energy
        margin_safe = max(target_margin, 1e-3)
        prox_bonus = 25.0 * (1.0 - (dist / margin_safe))
        fitness = base_fit + prox_bonus
    else:
        # Outside the range: smooth proportional penalty scaling continuously with distance.
        # Candidates just outside the border (e.g. 320.01 meV) are NOT discarded; penalty is minimal.
        # For larger distances, penalty increases gracefully while keeping a non-zero gradient.
        decay_scale = max(10.0, target_margin * 0.75)
        penalty_factor = 1.0 / (1.0 + (dist_outside / decay_scale) ** 2.0)
        fitness = base_fit * penalty_factor

    return float(max(0.1, fitness))


class DeapGAOptimizer:
    """
    Genetic Algorithm optimizer leveraging DEAP for quantum heterostructure design.
    """

    def __init__(
        self,
        n_params: int = 3,
        pop_size: int = 30,
        generations: int = 30,
        elitism_count: int = 2,
        cx_prob: float = 0.8,
        mut_prob: float = 0.15,
        cx_type: str = "uniform",
        objective: str = "ospc",
        opt_pc: bool = True,
        opt_os: bool = True,
        limit_energy: bool = True,
        target_energy: float = 300.0,
        target_margin: float = 20.0,
        csv_path: Optional[str] = None,
        resume: bool = False,
        output_dir: Optional[str] = None,
        bounds: Optional[List[Tuple[int, int]]] = None,
        seeds: Optional[List[List[int]]] = None,
        parallel: bool = False,
        n_workers: int = 1,
        is_cancelled: Optional[Callable[[], bool]] = None,
        on_individual_done: Optional[Callable] = None,
        on_generation_done: Optional[Callable] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ):
        self.n_params = int(n_params)
        self.pop_size = int(pop_size)
        self.generations = int(generations)
        self.elitism_count = int(elitism_count)
        self.cx_prob = float(cx_prob)
        self.mut_prob = float(mut_prob)
        self.cx_type = cx_type.lower()
        self.objective = objective
        self.opt_pc = bool(opt_pc)
        self.opt_os = bool(opt_os)
        self.limit_energy = bool(limit_energy)
        self.target_energy = float(target_energy)
        self.target_margin = float(target_margin)
        self.csv_path = csv_path
        import conf
        current_out = getattr(conf, "output_fortran_folder", output_fortran_folder)
        self.output_dir = output_dir or os.path.join(current_out, "optimizations")
        self.bounds = bounds or DEFAULT_BOUNDS.get(self.n_params, DEFAULT_BOUNDS[3])
        self.seeds = seeds or []
        self.parallel = bool(parallel)
        self.n_workers = max(1, int(n_workers))
        self.is_cancelled = is_cancelled or (lambda: False)
        self.on_individual_done = on_individual_done
        self.on_generation_done = on_generation_done
        self.on_log = on_log or (lambda msg: print(msg))

        # In-memory evaluation cache to avoid duplicate simulations
        self._simulation_cache: Dict[Tuple, Tuple[float, dict]] = {}
        self._cache_lock = threading.Lock()

        # Set up CSV path and determine initial generation
        self.start_generation = 1
        self.resumed_population = []
        self._setup_csv_and_resume()

        # Set up DEAP toolbox
        self._setup_toolbox()

    def _log(self, msg: str):
        if self.on_log:
            self.on_log(msg)

    def _setup_csv_and_resume(self):
        """Prepares CSV file and loads previous population if resuming."""
        os.makedirs(self.output_dir, exist_ok=True)

        if self.resume and self.csv_path and os.path.exists(self.csv_path):
            self._log(f"Continuando otimização a partir de: {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            if not df.empty and "generation" in df.columns:
                last_gen = int(df["generation"].max())
                self.start_generation = last_gen + 1
                self._log(f"Última geração detectada no CSV: {last_gen}. Retomando na geração {self.start_generation}.")

                # Extract population from last generation
                last_gen_df = df[df["generation"] == last_gen]
                for _, row in last_gen_df.iterrows():
                    try:
                        ind_raw = ast.literal_eval(str(row["individual"]))
                        fit_val = float(row["fitness"])
                        ind = creator.Individual(ind_raw)
                        ind.fitness.values = (fit_val,)
                        self.resumed_population.append(ind)
                    except Exception as e:
                        self._log(f"Aviso ao ler indivíduo do CSV: {e}")

                if self.resumed_population:
                    self.pop_size = len(self.resumed_population)
                    self._log(f"População restaurada com sucesso: {self.pop_size} indivíduos.")
        else:
            # Create a brand new CSV with start timestamp AAAAMMDD_HHMMSS_otim_X_parametros.csv
            now = datetime.datetime.now()
            csv_filename = f"{now.strftime('%Y%m%d_%H%M%S')}_otim_{self.n_params}_parametros.csv"
            self.csv_path = os.path.join(self.output_dir, csv_filename)
            self._log(f"Iniciando nova otimização. Arquivo CSV: {self.csv_path}")

            # Initialize CSV file with headers
            df_init = pd.DataFrame(columns=CSV_COLUMNS)
            df_init.to_csv(self.csv_path, index=False, header=True)

    def _setup_toolbox(self):
        """Configures DEAP toolbox operators."""
        self.toolbox = base.Toolbox()

        # Attribute generator for each gene based on its specific boundaries
        for i, (b_min, b_max) in enumerate(self.bounds):
            self.toolbox.register(f"attr_{i}", random.randint, b_min, b_max)

        def create_valid_individual():
            ind = creator.Individual([random.randint(b[0], b[1]) for b in self.bounds])
            clamp_individual(ind, self.bounds)
            return ind

        self.toolbox.register("individual", create_valid_individual)

        # Population generator
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Crossover operator
        if "one" in self.cx_type:
            self.toolbox.register("mate", tools.cxOnePoint)
        elif "two" in self.cx_type:
            self.toolbox.register("mate", tools.cxTwoPoint)
        else:
            self.toolbox.register("mate", tools.cxUniform, indpb=0.5)

        # Mutation operator: integer bounded mutation
        lows = [b[0] for b in self.bounds]
        ups = [b[1] for b in self.bounds]
        self.toolbox.register("mutate", tools.mutUniformInt, low=lows, up=ups, indpb=0.25)

        # Selection operator
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate_individual(self, ind: list) -> Tuple[float, dict]:
        """
        Evaluates an individual using FortranSimulator with caching.
        Returns (fitness, result_dict).
        """
        ind_key = tuple(ind)
        with self._cache_lock:
            if ind_key in self._simulation_cache:
                return self._simulation_cache[ind_key]

        clamp_individual(ind, self.bounds)

        sim_options = SimulationOptions(
            force_parser=False,
            force_simulation=False,
        )

        # FortranSimulator automatically checks if temp/{file_id}/FimPrograma.txt exists.
        # If it already exists, it skips Fortran execution and parses outputs from disk.
        t0 = time.time()
        import conf
        current_out = getattr(conf, "output_fortran_folder", output_fortran_folder)
        simulator = FortranSimulator(ind, sim_options=sim_options, output_folder=current_out)
        sample_data = simulator.simulate()
        elapsed = time.time() - t0

        fitness = compute_fitness(
            sample_data,
            objective=self.objective,
            opt_pc=self.opt_pc,
            opt_os=self.opt_os,
            limit_energy=self.limit_energy,
            target_energy=self.target_energy,
            target_margin=self.target_margin,
        )

        res_dict = {
            "sample_id": sample_data.sample_id,
            "structure": sample_data.structure,
            "pc1": sample_data.max_abs_photocurrent,
            "pc1e": sample_data.max_e_abs_photocurrent,
            "pc2": sample_data.min_abs_photocurrent,
            "pc2e": sample_data.min_e_abs_photocurrent,
            "oscstr": sample_data.max_e_oscstr,
            "oscstre": sample_data.max_e_transition,
            "time": elapsed,
            "lw": sample_data.LLQW,
            "lqw": sample_data.LQW,
            "lqb": sample_data.LQB,
            "c": sample_data.MQW,
            "rw": sample_data.RLQW,
            "rqw": sample_data.RQW,
            "rqb": sample_data.RQB,
        }

        with self._cache_lock:
            self._simulation_cache[ind_key] = (fitness, res_dict)
        return fitness, res_dict

    def _evaluate_batch(self, individuals: list):
        """
        Evaluates a list of individuals sequentially or in parallel using ThreadPoolExecutor.
        Deduplicates uncached items before parallel execution to avoid redundant work.
        Updates ind.fitness.values and invokes on_individual_done callback.
        """
        if not individuals:
            return

        # 1. Identify unique individuals that need Fortran evaluation
        unique_to_eval = {}
        for ind in individuals:
            key = tuple(ind)
            with self._cache_lock:
                cached = key in self._simulation_cache
            if not cached and key not in unique_to_eval:
                unique_to_eval[key] = ind

        # 2. Run simulation in parallel or sequentially for uncached individuals
        if unique_to_eval:
            if self.parallel and self.n_workers > 1:
                n_threads = min(self.n_workers, len(unique_to_eval))
                self._log(
                    f"Executando {len(unique_to_eval)} simulações em paralelo "
                    f"usando {n_threads} núcleos/threads..."
                )
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
                    futures = {
                        executor.submit(self.evaluate_individual, ind): ind
                        for ind in unique_to_eval.values()
                    }
                    for fut in concurrent.futures.as_completed(futures):
                        if self.is_cancelled():
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                        try:
                            fut.result()
                        except Exception as e:
                            self._log(f"Erro em thread de simulação: {e}")
            else:
                for ind in unique_to_eval.values():
                    if self.is_cancelled():
                        break
                    self.evaluate_individual(ind)

        # 3. Assign fitness from cache and report progress
        total = len(individuals)
        for i, ind in enumerate(individuals):
            if self.is_cancelled():
                break
            fit, _ = self.evaluate_individual(ind)
            ind.fitness.values = (fit,)
            if self.on_individual_done:
                self.on_individual_done(i + 1, total, list(ind), fit)

    def _append_generation_to_csv(self, gen_num: int, population: list, meta_info: List[dict]):
        """Appends all evaluated individuals of a generation to the optimization CSV."""
        rows = []
        for ind, meta in zip(population, meta_info):
            res = meta.get("res", {})
            origin = meta.get("origin", "unknown")
            parent1 = meta.get("parent1", "")
            parent2 = meta.get("parent2", "")

            row = {
                "generation": gen_num,
                "individual": str(list(ind)),
                "fitness": round(ind.fitness.values[0], 4),
                "structure": str(res.get("structure", [])),
                "OscStr max": round(res.get("oscstr", 0.0), 6),
                "OscStr energy (meV)": round(res.get("oscstre", 0.0), 2),
                "PC max 1": f"{res.get('pc1', 0.0):.4e}",
                "PC max 1 energy (meV)": round(res.get("pc1e", 0.0), 2),
                "PC max 2": f"{res.get('pc2', 0.0):.4e}",
                "PC max 2 energy (meV)": round(res.get("pc2e", 0.0), 2),
                "origin": origin,
                "parent1": str(parent1) if parent1 else "",
                "parent2": str(parent2) if parent2 else "",
                "left wells": res.get("lw", 0),
                "qw left (nm)": res.get("lqw", 0.0),
                "qb left (nm)": res.get("lqb", 0.0),
                "main qw (nm)": res.get("c", 0.0),
                "right wells": res.get("rw", 0),
                "qw right (nm)": res.get("rqw", 0.0),
                "qb right (nm)": res.get("rqb", 0.0),
                "python structure": str(res.get("structure", [])),
                "time": round(res.get("time", 0.0), 3),
            }
            rows.append(row)

        df_new = pd.DataFrame(rows, columns=CSV_COLUMNS)
        df_new.to_csv(self.csv_path, mode="a", index=False, header=False)

    def run(self) -> Tuple[list, float, str]:
        """
        Executes the genetic algorithm loop.
        Returns: (best_individual, best_fitness, csv_path)
        """
        target_info = f" | Faixa de Energia: {self.target_energy:.1f} ± {self.target_margin:.1f} meV" if self.limit_energy else " | Energia Livre (Qualquer pico)"
        opt_targets = []
        if self.opt_pc:
            opt_targets.append("PC")
        if self.opt_os:
            opt_targets.append("OS")
        self._log(f"Iniciando loop do GA (DEAP) para {self.generations} gerações [Objetivos: {', '.join(opt_targets)}{target_info}]...")

        # Initialize or restore population
        if self.resumed_population:
            population = [creator.Individual(list(ind)) for ind in self.resumed_population]
            for orig_ind, p_ind in zip(self.resumed_population, population):
                if orig_ind.fitness.valid:
                    p_ind.fitness.values = orig_ind.fitness.values
                p_ind._origin = "resumed"
        else:
            population = []
            if self.seeds:
                self._log(f"Injetando {len(self.seeds)} semente(s) na população inicial...")
                for seed in self.seeds:
                    ind = creator.Individual(list(seed))
                    clamp_individual(ind, self.bounds)
                    ind._origin = "seed"
                    population.append(ind)

            if len(population) < self.pop_size:
                needed = self.pop_size - len(population)
                for _ in range(needed):
                    ind = self.toolbox.individual()
                    ind._origin = "init"
                    population.append(ind)
            elif len(population) > self.pop_size:
                self.pop_size = len(population)
                self._log(f"Tamanho da população ajustado para {self.pop_size} indivíduos devido às sementes.")

        target_end_gen = self.start_generation + self.generations - 1

        # Evaluate initial population
        self._evaluate_batch(population)
        if self.is_cancelled():
            self._log("Otimização cancelada pelo usuário.")
            return [], 0.0, self.csv_path

        init_meta = []
        for ind in population:
            _, res = self.evaluate_individual(ind)
            init_meta.append({
                "origin": getattr(ind, "_origin", "init"),
                "parent1": "",
                "parent2": "",
                "res": res,
            })

        # If starting new, write generation 1 to CSV
        if not self.resumed_population:
            self._append_generation_to_csv(self.start_generation, population, init_meta)

            fits = [ind.fitness.values[0] for ind in population]
            best_ind = tools.selBest(population, 1)[0]
            best_fit = max(fits)
            mean_fit = float(np.mean(fits))

            _, best_res = self.evaluate_individual(best_ind)
            pc_max = best_res.get("pc1", 0.0)
            pc_e = best_res.get("pc1e", 0.0)
            os_max = best_res.get("oscstr", 0.0)
            os_e = best_res.get("oscstre", 0.0)

            self._log(
                f"Geração {self.start_generation} concluída: Melhor Fitness = {best_fit:.4f}, Média = {mean_fit:.4f} | "
                f"PC Máx = {pc_max:.4e} ({pc_e:.1f} meV) | OS Máx = {os_max:.4f} ({os_e:.1f} meV)"
            )

            if self.on_generation_done:
                self.on_generation_done(
                    self.start_generation,
                    target_end_gen,
                    list(best_ind),
                    best_fit,
                    mean_fit,
                    self.csv_path,
                    best_res
                )

        current_gen = self.start_generation if not self.resumed_population else self.start_generation - 1

        # Generational loop
        while current_gen < target_end_gen:
            if self.is_cancelled():
                self._log("Otimização cancelada pelo usuário.")
                break

            current_gen += 1
            self._log(f"--- Geração {current_gen}/{target_end_gen} ---")

            # 1. Elitism: preserve best individuals without modification
            elites = []
            if self.elitism_count > 0:
                elites = list(map(self.toolbox.clone, tools.selBest(population, self.elitism_count)))
                for elite in elites:
                    elite._origin = "elit"
                    elite._parents = ("", "")

            # 2. Select remaining individuals for reproduction
            offspring_count = self.pop_size - len(elites)
            offspring = self.toolbox.select(population, offspring_count)
            offspring = list(map(self.toolbox.clone, offspring))

            # 3. Apply Crossover
            for i in range(1, len(offspring), 2):
                if random.random() < self.cx_prob:
                    p1_copy = list(offspring[i - 1])
                    p2_copy = list(offspring[i])
                    self.toolbox.mate(offspring[i - 1], offspring[i])
                    clamp_individual(offspring[i - 1], self.bounds)
                    clamp_individual(offspring[i], self.bounds)
                    del offspring[i - 1].fitness.values
                    del offspring[i].fitness.values
                    offspring[i - 1]._parents = (p1_copy, p2_copy)
                    offspring[i]._parents = (p1_copy, p2_copy)

            # 4. Apply Mutation
            for mutant in offspring:
                if random.random() < self.mut_prob:
                    self.toolbox.mutate(mutant)
                    clamp_individual(mutant, self.bounds)
                    if mutant.fitness.valid:
                        del mutant.fitness.values
                    mutant._is_mutant = True

            # Set origins for offspring
            for ind in offspring:
                parents = getattr(ind, "_parents", ("", ""))
                is_mut = getattr(ind, "_is_mutant", False)
                origin = "mut" if is_mut else ("cross" if parents[0] else "select")
                ind._origin = origin
                ind._parents = parents

            # 5. Evaluate offspring
            self._evaluate_batch(offspring)
            if self.is_cancelled():
                self._log("Otimização cancelada durante avaliação da geração.")
                break

            # 6. Reassemble population: elites + offspring
            population[:] = elites + offspring

            # 7. Collect metadata and append to CSV
            gen_meta = []
            for ind in population:
                _, res = self.evaluate_individual(ind)
                parents = getattr(ind, "_parents", ("", ""))
                origin = getattr(ind, "_origin", "unknown")
                gen_meta.append({
                    "origin": origin,
                    "parent1": parents[0] if origin != "elit" else "",
                    "parent2": parents[1] if origin != "elit" else "",
                    "res": res,
                })

            self._append_generation_to_csv(current_gen, population, gen_meta)

            # 8. Compute generation statistics and notify callback
            fits = [ind.fitness.values[0] for ind in population]
            best_ind = tools.selBest(population, 1)[0]
            best_fit = max(fits)
            mean_fit = float(np.mean(fits))

            _, best_res = self.evaluate_individual(best_ind)
            pc_max = best_res.get("pc1", 0.0)
            pc_e = best_res.get("pc1e", 0.0)
            os_max = best_res.get("oscstr", 0.0)
            os_e = best_res.get("oscstre", 0.0)

            self._log(
                f"Geração {current_gen} concluída: Melhor Fitness = {best_fit:.4f}, Média = {mean_fit:.4f} | "
                f"PC Máx = {pc_max:.4e} ({pc_e:.1f} meV) | OS Máx = {os_max:.4f} ({os_e:.1f} meV)"
            )

            if self.on_generation_done:
                self.on_generation_done(
                    current_gen,
                    target_end_gen,
                    list(best_ind),
                    best_fit,
                    mean_fit,
                    self.csv_path,
                    best_res
                )

        best_ind = tools.selBest(population, 1)[0]
        best_fit = best_ind.fitness.values[0]
        _, best_res = self.evaluate_individual(best_ind)
        pc_max = best_res.get("pc1", 0.0)
        pc_e = best_res.get("pc1e", 0.0)
        os_max = best_res.get("oscstr", 0.0)
        os_e = best_res.get("oscstre", 0.0)

        self.best_individual = list(best_ind)
        self.best_fitness = best_fit
        self.best_result = best_res

        self._log(
            f"Otimização finalizada! Melhor indivíduo: {list(best_ind)} com Fitness = {best_fit:.4f} | "
            f"PC Máx = {pc_max:.4e} ({pc_e:.1f} meV) | OS Máx = {os_max:.4f} ({os_e:.1f} meV)"
        )
        return list(best_ind), best_fit, self.csv_path


if __name__ == "__main__":
    # Test quick run
    print("Testando DeapGAOptimizer...")
    opt = DeapGAOptimizer(
        n_params=3,
        pop_size=4,
        generations=2,
        objective="target_energy",
        target_energy=300.0,
        target_margin=20.0
    )
    best_ind, best_fit, csv_path = opt.run()
    print("CSV gerado:", csv_path)
    print("Best ind:", best_ind, "Fitness:", best_fit)
