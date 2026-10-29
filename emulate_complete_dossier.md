# e-mulate: Quantum Heterostructure Simulator & Evolutionary Bandgap Engineering Platform

**Lead Developer & Researcher:** Dr. Jose Ruiz  
**Principal Investigator:** Prof. Daniel Penello  
**Institution:** LabSem - Laboratório de Semicondutores, Departamento de Física, PUC-Rio  
**Timeline:** 2022 – 2026  
**License:** Academic / Open Research  
**Tech Stack:** Python 3.10+, Fortran 90, PyQt5, DEAP (Genetic Algorithms), NumPy, SciPy, Pandas, Matplotlib  

---

## 1. Executive Summary

**e-mulate** is an integrated computational physics and machine-assisted engineering platform designed to model, simulate, and inversely design quantum semiconductor heterostructures (coupled quantum wells and defective finite superlattices). It is primarily targeted at middle- and far-infrared optoelectronic devices, including:
- **QWIPs** (Quantum Well Infrared Photodetectors)
- **QCDs** (Quantum Cascade Detectors)
- **QCLs** (Quantum Cascade Lasers)
- Third-generation intermediate-band and quantum-well solar cells.

The software unifies rigorous quantum mechanics numerical solvers with an ultra-fast compiled Fortran 90 core and a bio-inspired multi-objective Genetic Algorithm (DEAP), presented through a modern PyQt5 interface with interactive bi-directional Matplotlib plots and Excel-style filterable database tables.

---

## 2. Theoretical Physics & Numerical Methods

### 2.1 1D Time-Independent Effective-Mass Schrödinger Equation
The electronic states confined in the conduction band along the epitaxial growth direction $z$ are governed by:

$$-\frac{\hbar^2}{2} \frac{d}{dz}\left[\frac{1}{m^*(z)}\frac{d\psi(z)}{dz}\right] + V(z)\psi(z) = E\psi(z)$$

Where:
- $m^*(z)$ is the position-dependent conduction band effective mass of electrons (e.g. $0.067\,m_0$ in GaAs vs. $0.067 + 0.083x\,m_0$ in $\text{Al}_x\text{Ga}_{1-x}\text{As}$).
- $V(z)$ is the step-like potential profile formed by the conduction band offset $\Delta E_c$ at each hetero-interface.

### 2.2 BenDaniel-Duke Boundary Conditions
To preserve probability density and probability flux continuity across abrupt hetero-interfaces at $z = z_0$:

$$\psi_A(z_0) = \psi_B(z_0) \quad \text{and} \quad \left.\frac{1}{m_A^*} \frac{d\psi_A}{dz}\right|_{z_0} = \left.\frac{1}{m_B^*} \frac{d\psi_B}{dz}\right|_{z_0}$$

### 2.3 4th-Order Numerov Method
For arbitrary continuous or piece-wise constant potential profiles, the 1D Schrödinger equation is solved using Numerov's discrete formulation with $\mathcal{O}(h^6)$ local truncation error:

$$y_{n+1} = \frac{2\left(1 - \frac{5}{12}h^2 k_n^2\right)y_n - \left(1 + \frac{1}{12}h^2 k_{n-1}^2\right)y_{n-1}}{1 + \frac{1}{12}h^2 k_{n+1}^2}$$

Yielding accurate bound-state eigenenergies $E_n$ and normalized envelope wavefunctions $\psi_n(z)$.

### 2.4 Transfer Matrix Method (TMM) for Transmission Probability
For unbound and quasi-bound resonant states subjected to external electric fields, the wavefunction in each layer $j$ is represented as right- and left-traveling waves:

$$\begin{pmatrix} A_{j+1} \\ B_{j+1} \end{pmatrix} = M_j \begin{pmatrix} A_j \\ B_j \end{pmatrix}, \quad M = \prod_{j=1}^{N} M_j$$

The transmission coefficient is determined by:

$$T(E) = \frac{k_R\, m_L^*}{k_L\, m_R^*} \frac{1}{|M_{11}|^2}$$

Peaks in $T(E)$ identify tunneling resonance energies and their full-width at half-maximum (FWHM) $\Gamma$ relates directly to state lifetime $\tau = \hbar / \Gamma$.

### 2.5 Intersubband Optical Transition & Oscillator Strength
Under transverse magnetic (TM) optical polarization, the electric dipole transition rate from ground state $\psi_i$ to excited state $\psi_f$ is characterized by the dimensionless oscillator strength:

$$f_{if} = \frac{2 m_0 (E_f - E_i)}{\hbar^2} \left| \int \psi_i^*(z)\, z\, \psi_f(z)\, dz \right|^2$$

### 2.6 Absolute Photocurrent Modeling
The spectral photocurrent response $I_{PC}(h\nu)$ is evaluated in true physical units of **Amperes (A)** without arbitrary relative references:

$$I_{PC}(h\nu) = q \cdot \Phi_0 \cdot \sum_{i,f} \alpha_{if}(h\nu) \cdot p_{\text{tunnel}}(E_f, F) \cdot \eta_{\text{collection}}$$

Integrating optical absorption cross-section, field-assisted tunneling escape probability $p_{\text{tunnel}}$, and miniband transport efficiency across the superlattice collector.

---

## 3. Superlattice Geometry & Free Parameters

The physical structure is parameterized by **7 geometric degrees of freedom**:

| Parameter | Full Name | Type / Unit | Typical Range | Physical Function |
| :--- | :--- | :--- | :--- | :--- |
| **LLQW** | Left Lateral QWs Count | Integer | 1 to 10 | Number of periodic quantum wells in the left injection block |
| **LQW** | Left Quantum Well Width | Float (nm) | 1.0 to 5.0 nm | Width of each quantum well in the left periodic block |
| **LQB** | Left Quantum Barrier Width | Float (nm) | 3.0 to 10.0 nm | Width of each barrier in the left block |
| **MQW** | Main Quantum Well (Defect) | Float (nm) | 2.0 to 6.0 nm | Defect well hosting the localized ground state |
| **RLQW** | Right Lateral QWs Count | Integer | 1 to 5 | Number of periodic quantum wells in the right collector block |
| **RQW** | Right Quantum Well Width | Float (nm) | 1.0 to 5.0 nm | Width of each quantum well in the right periodic block |
| **RQB** | Right Quantum Barrier Width | Float (nm) | 3.0 to 10.0 nm | Width of each barrier in the right block |

### Mandatory Physical Constraint:
$$MQW \ge \max(LQW, RQW) + 0.5\,\text{nm}$$
*Physics Reason:* The central well must be substantially wider than adjacent periodic wells to lower the ground state energy below the superlattice miniband, localizing electron population in the defect state for selective optical excitation.

---

## 4. Software Architecture & Core Modules

```
                    ┌──────────────────────────────────────────────┐
                    │            e-mulate GUI (PyQt5)              │
                    │  Modern Theme, Responsive Splitters, Status  │
                    └──────┬──────────────┬──────────────┬─────────┘
                           │              │              │
             ┌─────────────▼────┐  ┌──────▼──────┐  ┌────▼─────────────┐
             │ Fortran Simulator│  │  DEAP GA    │  │ Parameter Sweep  │
             │ Individual Run   │  │ Optimizer   │  │ (Automate Grid)  │
             └─────────────┬────┘  └──────┬──────┘  └────┬─────────────┘
                           │              │              │
                           └──────────────┼──────────────┘
                                          │
                    ┌─────────────────────▼────────────────────────┐
                    │      Subprocess Execution (No Shell Quote)   │
                    │  photocurrent_sim_windows.exe / MinGW Win64  │
                    └─────────────────────┬────────────────────────┘
                                          │
                    ┌─────────────────────▼────────────────────────┐
                    │ Output Parsing & Multi-Tier Caching Engine   │
                    │   8 Raw Output TXT files + samples_cache.json │
                    └─────────────────────┬────────────────────────┘
                                          │
                    ┌─────────────────────▼────────────────────────┐
                    │  Interactive Matplotlib Visualizations        │
                    │  - Band Potential & Wavefunctions            │
                    │  - Oscillator Strength Spectrum              │
                    │  - Absolute Photocurrent (A)                 │
                    │  - 2D Cross-Filtering Scatter Plot           │
                    └──────────────────────────────────────────────┘
```

### Module Breakdown:
1. **Fortran Sim (Run Fortran):**
   - High-performance compiled simulator generating potential profile, wavefunctions, oscillator strengths, and absolute photocurrent spectra in ~3 seconds.
   - Intelligent disk cache detection: if `FimPrograma.txt` exists, data parses instantly without re-running.
2. **Genetic Algorithm (GA / DEAP):**
   - Multi-objective inverse bandgap optimization (maximize PC, maximize OS, or targeted combination).
   - Targeted resonance window (default: $300.0 \pm 20.0\,\text{meV}$) with smooth continuous distance penalty.
   - Multi-core CPU parallelization via `ThreadPoolExecutor` and real-time generation fitness plotting.
3. **Automate (Grid Sweep):**
   - Multi-parameter combinatorial sweep across arbitrary step sizes.
   - Prior estimation of total simulation count and runtime.
   - Multithreaded execution with safe cancellation and direct CSV export.
4. **Database & Excel-Style Column Filters:**
   - Pre-computed database with 216+ structures.
   - Persistent JSON caching (`samples_cache.json`) loading 216 samples in < 50 ms.
   - Dedicated Excel-like column header menus: sorting (A-Z/Z-A), numeric comparison (`>`, `<`, `==`, range), and unique value checkboxes.
   - Interactive 2D scatter cross-filtering: clicking any data point on the scatter plot immediately highlights its row in the table and updates all 3 physical band plots.
5. **Cross-Platform Portability:**
   - Centralized `resolve_simulation_paths` automatically handles Windows (`\`) and Unix (`/`) separators with or without `temp` subfolders.
   - Settings menu to change database paths and executables on the fly.

---

## 5. Case Study & Benchmark Performance

**Reference Sample:** `05x01.0_03.7__02.1__01x01.0_03.7` & `05x02.0_07.0__02.5__01x02.0_07.0`
- **Max Photocurrent:** $1.5512 \times 10^{-11}\,\text{A}$ (peak at $309.6\,\text{meV}$ / $\lambda \approx 4.005\,\mu\text{m}$)
- **Max Oscillator Strength:** $0.3588$ (transition at $299.9\,\text{meV}$)
- **Single Simulation Runtime:** $\sim 3.8\,\text{seconds}$
- **Database Table Load Time (216 samples):** $< 0.05\,\text{seconds}$ via JSON cache.
