"""
Simulador Modular de Super-redes Quânticas em Python (Transfer Matrix & Photocurrent)
Arquivo: superlattice_solver.py

Tradução fiel e otimizada da sub-rotina Fortran modular:
20240825_Superlattice_Eigensates_Photocurrent_modular.f90

Física simulada:
1. Perfil espacial de potencial V(x) e massa efetiva m(x) dependente de energia (não-parabolicidade).
2. Autoenergias e funções de onda ligadas (matriz de transferência e método de bissecção).
3. Forças de oscilador intersubbanda considerando correções de não-parabolicidade.
4. Espectro de fotocorrente fotovoltaica e transmissão no contínuo.
"""

from dataclasses import dataclass
import os
import sys
import time
from typing import Optional, Tuple, List
import numpy as np


# ==============================================================================
# 1. CONSTANTES FÍSICAS E PARÂMETROS DO MODELO (simulation_constants)
# ==============================================================================
PI: float = 3.14159265359
ME: float = 9.10938356e-31         # Massa do elétron livre [kg]
HBAR: float = 1.054571817e-34      # Constante reduzida de Planck [J*s]
E_CHARGE: float = 1.602176634e-19  # Carga elementar [C]
NM: float = 1.0e-9                 # Escala de nanômetros [m]
IC: complex = 1j                   # Unidade imaginária

# Parâmetros dos materiais para a heteroestrutura InGaAs / InAlAs
EO_INGAAS: float = 14.3            # Constante dielétrica do InGaAs
EO_INALAS: float = 12.7            # Constante dielétrica do InAlAs
M_INGAAS: float = 0.0436 * ME      # Massa efetiva no InGaAs [kg]
M_INALAS: float = 0.0836 * ME      # Massa efetiva no InAlAs [kg]
VO: float = 0.503 * E_CHARGE       # Altura da barreira de potencial [J]
BARREIRA_EV: float = 0.503         # Altura da barreira em [eV]

# Parâmetros de não-parabolicidade (Nelson et al. PhysRevB.50.8663)
NONPARABW: float = 1.3e-18
EW: float = HBAR * HBAR / (2.0 * M_INGAAS * NONPARABW)   # Gap do InGaAs [J]
EB: float = EW * M_INALAS / M_INGAAS                     # Gap do InAlAs [J]

A0_INGAAS: float = 0.53 * EO_INGAAS / M_INGAAS
A0_INALAS: float = 0.53 * EO_INALAS / M_INALAS

# Parâmetros numéricos do algoritmo de bissecção
TOL: float = E_CHARGE * 5.0e-4 * 1.0e-12
MAX_INTERACTIONS: int = 1000


# ==============================================================================
# 2. CONFIGURAÇÃO GEOMÉTRICA (superlattice_config)
# ==============================================================================
@dataclass
class SuperlatticeConfig:
    """Configuração dimensional e estrutural da super-rede quântica."""
    num_qb_left: int       # Número de períodos poço/barreira no espelho de Bragg esquerdo
    L_qw_left: float       # Largura do poço quântico esquerdo [m]
    L_qb_left: float       # Largura da barreira quântica esquerda [m]
    L_cqw: float           # Largura do poço quântico central (defeito) [m]
    num_qb_right: int      # Número de períodos barreira/poço no espelho de Bragg direito
    L_qw_right: float      # Largura do poço quântico direito [m]
    L_qb_right: float      # Largura da barreira quântica direita [m]

    @classmethod
    def from_nm(cls, num_qb_left: int, L_qw_left_nm: float, L_qb_left_nm: float,
                L_cqw_nm: float, num_qb_right: int, L_qw_right_nm: float,
                L_qb_right_nm: float) -> "SuperlatticeConfig":
        """Instancia configuração convertendo dimensões de nm para metros [m]."""
        return cls(
            num_qb_left=int(num_qb_left),
            L_qw_left=float(L_qw_left_nm) * NM,
            L_qb_left=float(L_qb_left_nm) * NM,
            L_cqw=float(L_cqw_nm) * NM,
            num_qb_right=int(num_qb_right),
            L_qw_right=float(L_qw_right_nm) * NM,
            L_qb_right=float(L_qb_right_nm) * NM,
        )


# ==============================================================================
# 3. UTILIDADES MATEMÁTICAS E NUMÉRICAS (math_utils)
# ==============================================================================
def trap(d: np.ndarray, ni: int, nf: int, dx: float) -> complex:
    """
    Integração numérica pela regra do trapézio composta para arranjos complexos.
    Replica com exatidão a fórmula utilizada no Fortran (índices 0-based correspondentes).
    
    No Fortran (1-based):
      trap = sum_{j = ni + 2}^{nf - 1} d(j)*dx + 0.5 * (d(ni) + d(nf)) * dx
    Em Python (0-based):
      ni_0 = ni, nf_0 = nf
      j vai de ni + 2 até nf - 2 (inclusive)
    """
    val = np.sum(d[ni + 2 : nf - 1]) * dx
    val += 0.5 * (d[ni] + d[nf - 1]) * dx
    return complex(val)


def calculate_derivative_real(y: np.ndarray, dx: float) -> np.ndarray:
    """
    Cálculo de derivadas espaciais para arranjos reais com diferenças finitas centradas
    nos pontos interiores e diferenças avançada/atrasada nos extremos.
    """
    n = len(y)
    dy = np.empty(n, dtype=np.float64)
    dy[0] = (y[1] - y[0]) / dx
    dy[-1] = (y[-1] - y[-2]) / dx
    dy[1:-1] = (y[2:] - y[:-2]) / (2.0 * dx)
    return dy


def calculate_derivative_complex(y: np.ndarray, dx: float) -> np.ndarray:
    """
    Cálculo de derivadas espaciais para funções de onda complexas com diferenças finitas.
    """
    n = len(y)
    dy = np.empty(n, dtype=np.complex128)
    dy[0] = (y[1] - y[0]) / dx
    dy[-1] = (y[-1] - y[-2]) / dx
    dy[1:-1] = (y[2:] - y[:-2]) / (2.0 * dx)
    return dy


# ==============================================================================
# 4. GEOMETRIA, MALHA E PERFIS DE MATERIAIS (superlattice_structure)
# ==============================================================================
def compute_grid_geometry(cfg: SuperlatticeConfig, dx: float) -> Tuple[float, float, int, np.ndarray]:
    """
    Calcula os limites do domínio espacial L_left e L_right, o número de nós n,
    e constrói a malha de coordenadas espaciais x(j) [m].
    Garante número de nós idêntico ao gerado pelo compilador Fortran.
    """
    L_bm_left = -float(cfg.num_qb_left) * (cfg.L_qb_left + cfg.L_qw_left)
    L_bm_right = float(cfg.num_qb_right) * (cfg.L_qb_right + cfg.L_qw_right)

    if -L_bm_left >= L_bm_right:
        L0_left = -50.0 * NM
        L_left = L_bm_left + L0_left
        L0_right = -L_left - cfg.L_cqw - L_bm_right
        L_right = cfg.L_cqw + L_bm_right + L0_right + cfg.L_cqw
    else:
        L0_right = 50.0 * NM
        L_right = cfg.L_cqw + L_bm_right + L0_right + cfg.L_cqw
        L0_left = -L_right - L_bm_left
        L_left = L_bm_left + L0_left

    # O código Fortran original usa int((L_right - L_left)/dx) + 2 onde dx tem precisão float32.
    # Usar int(round((L_right - L_left)/dx)) + 1 garante que x[n-1] == L_right com precisão exata.
    n = int(round((L_right - L_left) / dx)) + 1

    x = L_left + np.arange(n, dtype=np.float64) * dx
    return L_left, L_right, n, x


def build_potential(cfg: SuperlatticeConfig, x: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Constrói o perfil de potencial V(x) e determina o índice do nó de partição x_cut (0-based).
    Condições de contorno de poço infinito nas extremidades: v[0] = v[-1] = 10000 eV.
    """
    n = len(x)
    v = np.zeros(n, dtype=np.float64)

    L_bm_left_len = float(cfg.num_qb_left) * (cfg.L_qb_left + cfg.L_qw_left)
    L0_left_len = L_bm_left_len
    L0 = L_bm_left_len

    # 1. Barreira externa esquerda
    j = 0
    while j < n and x[j] <= -L0:
        v[j] = VO
        j += 1

    # 2. Super-rede à esquerda do poço central (poço / barreira)
    k = 1
    n_qb_left = 0
    i = j
    while i < n:
        L0 = L0_left_len - float(n_qb_left) * (cfg.L_qb_left + cfg.L_qw_left)
        if x[i] >= -L0 and x[i] <= -L0 + cfg.L_qw_left:
            v[i] = 0.0
        elif x[i] >= -L0 + cfg.L_qw_left and x[i] <= -L0 + cfg.L_qw_left + cfg.L_qb_left:
            v[i] = VO
        else:
            v[i] = 0.0
            n_qb_left += 1
            k += 1
        if k > cfg.num_qb_left:
            break
        i += 1

    x_cut = i  # Índice de junção 0-based

    # 3. Poço quântico central (defeito)
    j = i
    while j < n and x[j] >= 0.0 and x[j] <= cfg.L_cqw:
        v[j] = 0.0
        j += 1

    # 4. Super-rede à direita do poço central (barreira / poço)
    k = 1
    n_qb_right = 0
    i = j
    while i < n:
        L0 = cfg.L_cqw + float(n_qb_right) * (cfg.L_qw_right + cfg.L_qb_right)
        if x[i] >= L0 and x[i] <= L0 + cfg.L_qb_right:
            v[i] = VO
        elif x[i] >= L0 + cfg.L_qb_right and x[i] <= L0 + cfg.L_qb_right + cfg.L_qw_right:
            v[i] = 0.0
        else:
            v[i] = VO
            n_qb_right += 1
            k += 1
        if k > cfg.num_qb_right:
            break
        i += 1

    # 5. Barreira externa direita
    for j_idx in range(i, n):
        v[j_idx] = VO

    # Condições de contorno de potencial infinito nas extremidades
    v[0] = 10000.0 * E_CHARGE
    v[-1] = 10000.0 * E_CHARGE

    return v, x_cut


def build_effective_mass(cfg: SuperlatticeConfig, x: np.ndarray, m_qw: float, m_qb: float) -> np.ndarray:
    """
    Constrói o perfil espacial de massa efetiva m(x) ao longo da heteroestrutura.
    """
    n = len(x)
    m = np.empty(n, dtype=np.float64)

    L_bm_left_len = float(cfg.num_qb_left) * (cfg.L_qb_left + cfg.L_qw_left)
    L0_left_len = L_bm_left_len
    L0 = L_bm_left_len

    # 1. Barreira externa esquerda
    j = 0
    while j < n and x[j] <= -L0:
        m[j] = m_qb
        j += 1

    # 2. Super-rede esquerda
    k = 1
    n_qb_left = 0
    i = j
    while i < n:
        L0 = L0_left_len - float(n_qb_left) * (cfg.L_qb_left + cfg.L_qw_left)
        if x[i] >= -L0 and x[i] <= -L0 + cfg.L_qw_left:
            m[i] = m_qw
        elif x[i] >= -L0 + cfg.L_qw_left and x[i] <= -L0 + cfg.L_qw_left + cfg.L_qb_left:
            m[i] = m_qb
        else:
            m[i] = m_qw
            n_qb_left += 1
            k += 1
        if k > cfg.num_qb_left:
            break
        i += 1

    # 3. Poço quântico central
    j = i
    while j < n and x[j] >= 0.0 and x[j] <= cfg.L_cqw:
        m[j] = m_qw
        j += 1

    # 4. Super-rede direita
    k = 1
    n_qb_right = 0
    i = j
    while i < n:
        L0 = cfg.L_cqw + float(n_qb_right) * (cfg.L_qw_right + cfg.L_qb_right)
        if x[i] >= L0 and x[i] <= L0 + cfg.L_qb_right:
            m[i] = m_qb
        elif x[i] >= L0 + cfg.L_qb_right and x[i] <= L0 + cfg.L_qb_right + cfg.L_qw_right:
            m[i] = m_qw
        else:
            m[i] = m_qb
            n_qb_right += 1
            k += 1
        if k > cfg.num_qb_right:
            break
        i += 1

    # 5. Barreira externa direita
    for j_idx in range(i, n):
        m[j_idx] = m_qb

    return m


def get_nonparabolic_mass(Ener: float, E0: float = 0.0, is_photocurrent: bool = False) -> Tuple[float, float]:
    """
    Calcula as massas efetivas nos poços e barreiras dependentes da energia
    segundo o modelo de não-parabolicidade de Nelson et al.
    """
    if is_photocurrent:
        m_qw = M_INGAAS * (1.0 + Ener / EW)
        m_qb = M_INALAS * (1.0 + (E0 + Ener - VO) / EB)
    else:
        m_qw = M_INGAAS * (1.0 + Ener / EW)
        m_qb = M_INALAS * (1.0 + (Ener - VO) / EB)
    return m_qw, m_qb


# ==============================================================================
# 5. MATRIZ DE TRANSFERÊNCIA E AUTOESTADOS (transfer_matrix_solver)
# ==============================================================================
def compute_transfer_matrix(Ener: float, dx: float, v: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Calcula a matriz de transferência 2x2 complexa total para uma determinada energia Ener [J].
    Implementa as condições de contorno de BenDaniel-Duke em cada interface.
    Vetorizado para alto desempenho.
    """
    n = len(v)
    diff = Ener - v
    is_pos = diff > 0.0

    ki = np.empty(n, dtype=np.complex128)
    ki[is_pos] = np.sqrt(2.0 * m[is_pos] * diff[is_pos] / (HBAR * HBAR))
    ki[~is_pos] = IC * np.sqrt(2.0 * m[~is_pos] * (-diff[~is_pos]) / (HBAR * HBAR))

    dx_half = dx / 2.0

    # Pré-cálculo vetorizado das exponenciais de propagação e matrizes de descontinuidade
    p_all = np.exp(IC * ki * dx_half)
    p_all_inv = 1.0 / p_all

    beta_all = (ki[:-1] * m[1:]) / (ki[1:] * m[:-1])
    d1_all = 0.5 * (1.0 + beta_all)
    d2_all = 0.5 * (1.0 - beta_all)

    # Elementos das matrizes Ti(i) = P_final @ Dn @ P_inicial
    ti_00 = p_all[1:] * d1_all * p_all[:-1]
    ti_01 = p_all[1:] * d2_all * p_all_inv[:-1]
    ti_10 = p_all_inv[1:] * d2_all * p_all[:-1]
    ti_11 = p_all_inv[1:] * d1_all * p_all_inv[:-1]

    # Multiplicação acumulativa T = Ti @ T0 (T0 inicial = identidade)
    t00 = 1.0 + 0j
    t01 = 0j
    t10 = 0j
    t11 = 1.0 + 0j

    for i in range(n - 1):
        c00 = ti_00[i]
        c01 = ti_01[i]
        c10 = ti_10[i]
        c11 = ti_11[i]

        new_00 = c00 * t00 + c01 * t10
        new_01 = c00 * t01 + c01 * t11
        new_10 = c10 * t00 + c11 * t10
        new_11 = c10 * t01 + c11 * t11

        t00 = new_00
        t01 = new_01
        t10 = new_10
        t11 = new_11

    return np.array([[t00, t01], [t10, t11]], dtype=np.complex128)


def compute_wavefunction_split(n: int, x_cut: int, dx: float, k: int, Ener: float,
                               v: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Cálculo da função de onda de um autoestado por junção bidirecional (split).
    Propaga da esquerda (0 -> x_cut) e da direita (n-1 -> x_cut),
    ajusta a paridade (-1)^k, faz a continuidade e normaliza integral(|psi|^2 dx) = 1.
    """
    diff = Ener - v
    is_pos = diff > 0.0

    ki = np.empty(n, dtype=np.complex128)
    ki[is_pos] = np.sqrt(2.0 * m[is_pos] * diff[is_pos] / (HBAR * HBAR))
    ki[~is_pos] = IC * np.sqrt(2.0 * m[~is_pos] * (-diff[~is_pos]) / (HBAR * HBAR))

    dx_half = dx / 2.0

    # 1. Integração da esquerda para a direita (0 -> x_cut)
    WaveFunc_L = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.complex128)
    psi_L = np.zeros(n, dtype=np.complex128)

    for i in range(x_cut + 1):
        ki_curr = ki[i]
        ki_next = ki[i + 1]
        beta = (ki_curr * m[i + 1]) / (ki_next * m[i])

        p_f1 = np.exp(IC * ki_next * dx_half)
        p_f2 = np.exp(-IC * ki_next * dx_half)
        p_i1 = np.exp(IC * ki_curr * dx_half)
        p_i2 = np.exp(-IC * ki_curr * dx_half)

        d1 = 0.5 * (1.0 + beta)
        d2 = 0.5 * (1.0 - beta)

        ti_00 = p_f1 * d1 * p_i1
        ti_01 = p_f1 * d2 * p_i2
        ti_10 = p_f2 * d2 * p_i1
        ti_11 = p_f2 * d1 * p_i2

        wf_00 = ti_00 * WaveFunc_L[0, 0] + ti_01 * WaveFunc_L[1, 0]
        wf_10 = ti_10 * WaveFunc_L[0, 0] + ti_11 * WaveFunc_L[1, 0]
        WaveFunc_L[0, 0] = wf_00
        WaveFunc_L[1, 0] = wf_10

        psi_L[i] = WaveFunc_L[0, 0] + WaveFunc_L[1, 0]

    # 2. Integração da direita para a esquerda (n-1 -> x_cut com -dx_half)
    WaveFunc_R = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.complex128)
    psi_R = np.zeros(n, dtype=np.complex128)

    for i in range(n - 1, x_cut - 1, -1):
        ki_curr = ki[i]
        ki_prev = ki[i - 1]
        beta = (ki_curr * m[i - 1]) / (ki_prev * m[i])

        p_f1 = np.exp(IC * ki_prev * (-dx_half))
        p_f2 = np.exp(-IC * ki_prev * (-dx_half))
        p_i1 = np.exp(IC * ki_curr * (-dx_half))
        p_i2 = np.exp(-IC * ki_curr * (-dx_half))

        d1 = 0.5 * (1.0 + beta)
        d2 = 0.5 * (1.0 - beta)

        ti_00 = p_f1 * d1 * p_i1
        ti_01 = p_f1 * d2 * p_i2
        ti_10 = p_f2 * d2 * p_i1
        ti_11 = p_f2 * d1 * p_i2

        wf_00 = ti_00 * WaveFunc_R[0, 0] + ti_01 * WaveFunc_R[1, 0]
        wf_10 = ti_10 * WaveFunc_R[0, 0] + ti_11 * WaveFunc_R[1, 0]
        WaveFunc_R[0, 0] = wf_00
        WaveFunc_R[1, 0] = wf_10

        psi_R[i] = WaveFunc_R[0, 0] + WaveFunc_R[1, 0]

    # 3. Ajuste de paridade e junção contínua em x_cut
    parity = ((-1.0) ** k)
    psi_R[x_cut:n] = parity * psi_R[x_cut:n]

    scale = psi_R[x_cut] / psi_L[x_cut]
    psi_L[:x_cut + 1] = psi_L[:x_cut + 1] * scale

    psi = np.empty(n, dtype=np.complex128)
    psi[:x_cut] = psi_L[:x_cut]
    psi[x_cut:] = psi_R[x_cut:]

    # 4. Normalização: integral(|psi|^2 dx) = 1
    prod = np.conj(psi) * psi
    soma = trap(prod, 0, n, dx)
    psi = psi / np.sqrt(soma.real)

    return psi


def find_eigenstates(cfg: SuperlatticeConfig, x: np.ndarray, v: np.ndarray, dx: float,
                     x_cut: int, Ei: float = 0.0, Ef: float = 700.0, dE: float = 0.1,
                     max_states: int = 100, energy_file: Optional[str] = None
                     ) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Busca de autoenergias ligadas por bissecção em T(2,2) e cálculo das funções de onda psi0.
    Retorna (num_found, energy, psi0):
      num_found: Quantidade de autoestados encontrados
      energy:    Vetor 1D com energias em Joules [J] (tamanho num_found)
      psi0:      Matriz 2D de funções de onda complexas normalizadas psi0(n, num_found)
    """
    n = len(x)
    En = int((Ef - Ei) / dE)

    found_energies: List[float] = []

    # Estado inicial em jn = 1
    Ener_old = (Ei + 1.0 * dE) * E_CHARGE * 1.0e-3
    Ener = Ener_old
    m_qw, m_qb = get_nonparabolic_mass(Ener, 0.0, False)
    m = build_effective_mass(cfg, x, m_qw, m_qb)
    T = compute_transfer_matrix(Ener, dx, v, m)
    T22_old = T[1, 1].real

    file_handle = None
    if energy_file:
        file_handle = open(energy_file, "w")

    try:
        # Varredura em energia procurando cruzamentos por zero em T(2,2)
        for jn in range(2, En + 1):
            Ener_new = (Ei + float(jn) * dE) * E_CHARGE * 1.0e-3
            Ener = Ener_new
            m_qw, m_qb = get_nonparabolic_mass(Ener, 0.0, False)
            m = build_effective_mass(cfg, x, m_qw, m_qb)
            T = compute_transfer_matrix(Ener, dx, v, m)
            T22_new = T[1, 1].real

            mult2 = T22_old * T22_new

            # Cruzamento por zero detectado -> Refinamento por Bissecção
            if mult2 < 0.0:
                interaction = 0
                e_left = Ener_old
                e_right = Ener_new

                while abs(e_right - e_left) > TOL and interaction <= MAX_INTERACTIONS:
                    e_mid = (e_left + e_right) / 2.0
                    m_qw_mid, m_qb_mid = get_nonparabolic_mass(e_mid, 0.0, False)
                    m_mid = build_effective_mass(cfg, x, m_qw_mid, m_qb_mid)
                    T_mid = compute_transfer_matrix(e_mid, dx, v, m_mid)
                    T22_mid = T_mid[1, 1].real

                    if (T22_old * T22_mid) < 0.0:
                        e_right = e_mid
                    else:
                        e_left = e_mid

                    interaction += 1

                if len(found_energies) < max_states:
                    found_energies.append(e_mid)
                    if file_handle:
                        file_handle.write(f"   {1.0e3 * e_mid / E_CHARGE:22.14f}\n")

            # Preparar próximo passo da varredura
            Ener = (Ei + float(jn) * dE) * E_CHARGE * 1.0e-3
            m_qw, m_qb = get_nonparabolic_mass(Ener, 0.0, False)
            m = build_effective_mass(cfg, x, m_qw, m_qb)
            T = compute_transfer_matrix(Ener, dx, v, m)
            T22_old = T[1, 1].real
            Ener_old = Ener
    finally:
        if file_handle:
            file_handle.close()

    num_found = len(found_energies)
    energy_arr = np.array(found_energies, dtype=np.float64)

    # Cálculo das funções de onda split para cada autoenergia
    psi0 = np.empty((n, num_found), dtype=np.complex128)
    for idx, E_val in enumerate(energy_arr):
        m_qw, m_qb = get_nonparabolic_mass(E_val, 0.0, False)
        m_state = build_effective_mass(cfg, x, m_qw, m_qb)
        psi0[:, idx] = compute_wavefunction_split(n, x_cut, dx, idx + 1, E_val, v, m_state)

    return num_found, energy_arr, psi0


# ==============================================================================
# 6. PROPRIEDADES ÓPTICAS (optical_properties)
# ==============================================================================
def compute_oscillator_strengths(cfg: SuperlatticeConfig, x: np.ndarray, dx: float,
                                 num_states: int, psi0: np.ndarray, energy: np.ndarray,
                                 osc_file: Optional[str] = None
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula as forças de oscilador intersubbanda f_{1 -> i} considerando correções
    de não-parabolicidade (Nelson et al. Phys. Rev. B 50, 8663, 1994).
    Retorna (dE_meV, f_osc):
      dE_meV: Vetor de diferenças de energia E_i - E_1 em meV para i >= 2
      f_osc:  Forças de oscilador adimensionais correspondentes
    """
    n = len(x)
    f_osc = np.zeros(num_states, dtype=np.float64)

    # Massa efetiva e sua derivada espacial para o estado fundamental
    m_qw_1, m_qb_1 = get_nonparabolic_mass(energy[0], 0.0, False)
    meff_ground = build_effective_mass(cfg, x, m_qw_1, m_qb_1)
    mass_deri = calculate_derivative_real(meff_ground, dx)

    psi_1_conj = np.conj(psi0[:, 0])

    file_handle = None
    if osc_file:
        file_handle = open(osc_file, "w")

    try:
        for i in range(1, num_states):
            m_qw_i, m_qb_i = get_nonparabolic_mass(energy[i], 0.0, False)
            meff_state = build_effective_mass(cfg, x, m_qw_i, m_qb_i)

            psi0_deri = calculate_derivative_complex(psi0[:, i], dx)

            # Elemento de matriz com correção de não-parabolicidade:
            prod = (- psi_1_conj * psi0[:, i] * mass_deri / (meff_ground * meff_ground)
                    + psi_1_conj * psi0_deri / meff_ground
                    + psi_1_conj * psi0_deri / meff_state)

            soma = trap(prod, 0, n, dx)
            dE = energy[i] - energy[0]

            f_val = m_qw_i * (HBAR * HBAR) * (abs(soma) ** 2) / (2.0 * dE)
            f_osc[i] = f_val

            if file_handle:
                file_handle.write(f"   {1.0e3 * dE / E_CHARGE:22.14f}        {f_val:22.14e}\n")
    finally:
        if file_handle:
            file_handle.close()

    dE_arr = (energy[1:num_states] - energy[0]) * 1.0e3 / E_CHARGE
    return dE_arr, f_osc[1:num_states]


def compute_continuum_photocurrent(cfg: SuperlatticeConfig, x: np.ndarray, v: np.ndarray,
                                   dx: float, psi_ground: np.ndarray, E0: float,
                                   Ei: float = 0.0, Ef: float = 500.0, dE: float = 0.1,
                                   Fd: float = 1.0e5 * E_CHARGE,
                                   pc_file: Optional[str] = None,
                                   trans_file: Optional[str] = None
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Espectro de fotocorrente fotovoltaica no contínuo e coeficiente de transmissão
    (Vinter et al. Phys. Rev. B 60, R13993, 1999).
    Retorna (energies_meV, photocurrent, transmission).
    """
    n = len(x)
    # No Fortran original (inputs.f90 / modular.f90), o passo dE = 0.1 é um literal sem 'd0',
    # portanto compilado como REAL*4 (float32 = 0.10000000149011612).
    # Emulamos exatamente esta precisão para garantir coincidência 1:1 absoluta com o binário.
    if abs(dE - 0.1) < 1e-12:
        dE_val = float(np.float32(0.1))
        conv_factor = float(np.float32(1e-3))
    else:
        dE_val = float(dE)
        conv_factor = 1.0e-3

    En = int((Ef - Ei) / dE_val)

    energies_meV = np.empty(En, dtype=np.float64)
    photocurrent = np.empty(En, dtype=np.float64)
    transmission = np.empty(En, dtype=np.float64)

    fh_pc = open(pc_file, "w") if pc_file else None
    fh_tr = open(trans_file, "w") if trans_file else None

    dx_val = dx
    hbar_sq = HBAR * HBAR

    try:
        for jn in range(1, En + 1):
            Ener_J = (Ei + float(jn) * dE_val) * conv_factor * E_CHARGE
            Ener_meV = (Ener_J / E_CHARGE) * 1.0e3
            Ener = Ener_J
            energies_meV[jn - 1] = Ener_meV

            # Correção de não-parabolicidade no contínuo
            m_qw, m_qb = get_nonparabolic_mass(Ener, E0, True)
            m = build_effective_mass(cfg, x, m_qw, m_qb)

            T = np.eye(2, dtype=np.complex128)
            T0 = np.eye(2, dtype=np.complex128)
            F = np.zeros(2, dtype=np.complex128)

            aux1 = (2.0 * m * (Ener + E0 - v)) / hbar_sq
            ki = np.sqrt(aux1.astype(np.complex128))

            # Pré-cálculo vetorizado das exponenciais de propagação e coeficientes
            exp_ki_curr_xi = np.exp(IC * ki[:-1] * x[:-1])
            exp_ki_next_xi = np.exp(IC * ki[1:] * x[:-1])
            exp_ki_curr_xnext = np.exp(IC * ki[:-1] * x[1:])

            inv_ki_curr_xi = 1.0 / exp_ki_curr_xi
            inv_ki_next_xi = 1.0 / exp_ki_next_xi
            inv_ki_curr_xnext = 1.0 / exp_ki_curr_xnext

            beta_arr = (ki[:-1] * m[1:]) / (ki[1:] * m[:-1])
            d1_arr = 0.5 * (1.0 + beta_arr)
            d2_arr = 0.5 * (1.0 - beta_arr)

            cte_arr = (m[:-1] * Fd) / (2.0 * IC * hbar_sq * ki[:-1])
            term1 = exp_ki_curr_xi * x[:-1] * psi_ground[:-1]
            term2 = exp_ki_curr_xnext * x[1:] * psi_ground[1:]
            term3 = inv_ki_curr_xi * x[:-1] * psi_ground[:-1]
            term4 = inv_ki_curr_xnext * x[1:] * psi_ground[1:]

            beta_plus_arr = -cte_arr * (term1 - term2) * dx_val
            beta_minus_arr = cte_arr * (term3 - term4) * dx_val

            # Propagação regressiva do extremo direito n-2 até 0 (no Fortran: n-1 até 1)
            for i in range(n - 2, -1, -1):
                P_inicial_00 = inv_ki_next_xi[i]
                P_inicial_11 = exp_ki_next_xi[i]

                P_final_00 = exp_ki_curr_xi[i]
                P_final_11 = inv_ki_curr_xi[i]

                d1 = d1_arr[i]
                d2 = d2_arr[i]

                # aux = P_inicial @ Dn
                aux_00 = P_inicial_00 * d1
                aux_01 = P_inicial_00 * d2
                aux_10 = P_inicial_11 * d2
                aux_11 = P_inicial_11 * d1

                # Ti = aux @ P_final
                ti_00 = aux_00 * P_final_00
                ti_01 = aux_01 * P_final_11
                ti_10 = aux_10 * P_final_00
                ti_11 = aux_11 * P_final_11

                # T = T0 @ Ti
                t_00 = T0[0, 0] * ti_00 + T0[0, 1] * ti_10
                t_01 = T0[0, 0] * ti_01 + T0[0, 1] * ti_11
                t_10 = T0[1, 0] * ti_00 + T0[1, 1] * ti_10
                t_11 = T0[1, 0] * ti_01 + T0[1, 1] * ti_11

                T0[0, 0] = t_00
                T0[0, 1] = t_01
                T0[1, 0] = t_10
                T0[1, 1] = t_11

                # Interação elétron-fóton na interface i
                b_plus = beta_plus_arr[i]
                b_minus = beta_minus_arr[i]

                F[0] += (t_00 * b_minus + t_01 * b_plus)
                F[1] += (t_10 * b_minus + t_11 * b_plus)

            # No regime sub-barreira (Ener + E0 <= Vo), o elétron excitado não possui energia
            # suficiente para escapar para o contínuo sobre as barreiras externas (V(1) = V(n) = Vo).
            # As ondas nas extremidades são puramente evanescentes (ki imaginário puro) e a fotocorrente
            # de escape para os contatos é fisicamente nula (0.0).
            # Evita também a amplificação numérica espúria de resíduos de ponto flutuante por |T(2,2)| ~ 10^-20.
            if Ener_J + E0 <= VO or abs(T0[1, 1]) < 1e-18:
                pc_val = 0.0
            else:
                ctecur_left = (HBAR * ki[0].real) / m[0]
                ctecur_right = (HBAR * ki[-1].real) / m[-1]

                ratio = -F[1] / T0[1, 1]
                ratio_sq = ratio.real * ratio.real + ratio.imag * ratio.imag
                Je = ctecur_left * Fd * ratio_sq

                term_d = F[0] - F[1] * T0[0, 1] / T0[1, 1]
                term_d_sq = term_d.real * term_d.real + term_d.imag * term_d.imag
                Jd = ctecur_right * Fd * term_d_sq

                pc_val = Jd - Je

            tr_val = 1.0 / np.abs(T0[0, 0])

            photocurrent[jn - 1] = pc_val
            transmission[jn - 1] = tr_val

            if fh_pc:
                # Gravação em notação científica idêntica à saída do Fortran write(8,*)
                fh_pc.write(f"  {Ener_meV:20.14f}        {pc_val:22.16e}\n")
            if fh_tr:
                fh_tr.write(f"  {Ener_meV:20.14f}        {tr_val:22.16e}\n")
    finally:
        if fh_pc:
            fh_pc.close()
        if fh_tr:
            fh_tr.close()

    return energies_meV, photocurrent, transmission


# ==============================================================================
# 7. ORQUESTRADOR GERAL DA SIMULAÇÃO (run_superlattice_simulation)
# ==============================================================================
@dataclass
class SimulationResults:
    """Resultados completos da simulação da super-rede quântica."""
    x: np.ndarray                      # Coordenadas espaciais [m]
    v: np.ndarray                      # Perfil de potencial [J]
    energies: np.ndarray               # Autoenergias dos estados ligados [J]
    psi0: np.ndarray                   # Funções de onda dos estados ligados (n, num_states)
    f_osc: np.ndarray                  # Forças de oscilador dos estados excitados
    photon_energies_meV: np.ndarray    # Energias dos fótons para fotocorrente [meV]
    photocurrent: np.ndarray           # Espectro de fotocorrente [A/m^2]
    transmission: np.ndarray           # Coeficiente de transmissão no contínuo
    elapsed_time: float                # Tempo total de execução [s]


def run_superlattice_simulation(
    cfg: SuperlatticeConfig,
    sample_folder: Optional[str] = None,
    dx: float = 0.1 * NM,
    nn: int = 100,
    Fd: float = 1.0e5 * E_CHARGE,
    Ei_bound: float = 0.0,
    Ef_bound: float = 700.0,
    dE_bound: float = 0.1,
    Ei_pc: float = 0.0,
    Ef_pc: float = 500.0,
    dE_pc: float = 0.1,
) -> SimulationResults:
    """
    Executa o fluxo completo do simulador quântico de forma idêntica ao código Fortran.
    Salva os 8 arquivos padrão se sample_folder for especificado:
      - wavefunction_SL.txt
      - Energy_SL.txt
      - Transmission_SL.txt
      - Photocurrent_SL.txt
      - Potencial_SL.txt
      - OscStr_SL.txt
      - OscStr_SL_norm.txt
      - FimPrograma.txt
    """
    start_time = time.time()

    if sample_folder:
        os.makedirs(sample_folder, exist_ok=True)
        wf_path = os.path.join(sample_folder, "wavefunction_SL.txt")
        energy_path = os.path.join(sample_folder, "Energy_SL.txt")
        trans_path = os.path.join(sample_folder, "Transmission_SL.txt")
        pc_path = os.path.join(sample_folder, "Photocurrent_SL.txt")
        pot_path = os.path.join(sample_folder, "Potencial_SL.txt")
        osc_path = os.path.join(sample_folder, "OscStr_SL.txt")
        osc_norm_path = os.path.join(sample_folder, "OscStr_SL_norm.txt")
        fim_path = os.path.join(sample_folder, "FimPrograma.txt")
    else:
        wf_path = energy_path = trans_path = pc_path = pot_path = osc_path = osc_norm_path = fim_path = None

    # 1. Construção da geometria e malha
    L_left, L_right, n, x = compute_grid_geometry(cfg, dx)

    # 2. Perfil de potencial com barreiras infinitas nas pontas para busca de estados ligados
    v, x_cut = build_potential(cfg, x)

    # 3. Busca de autoestados ligados por bissecção
    num_found, energies, psi0 = find_eigenstates(
        cfg=cfg,
        x=x,
        v=v,
        dx=dx,
        x_cut=x_cut,
        Ei=Ei_bound,
        Ef=Ef_bound,
        dE=dE_bound,
        max_states=nn,
        energy_file=energy_path,
    )

    # 4. Gravação das funções de onda no formato esperado pela GUI de e-mulate
    if wf_path:
        with open(wf_path, "w") as f:
            for i in range(n):
                line_parts = [f"  {x[i] / NM:18.15f}"]
                for j in range(num_found):
                    val = (np.real(psi0[i, j] ** 2) + np.imag(psi0[i, j] ** 2)) * 0.4e-6 + 1.0e3 * energies[j] / E_CHARGE
                    line_parts.append(f"  {val:20.14f}")
                f.write(" ".join(line_parts) + "\n")

    # 5. Forças de oscilador intersubbanda
    dE_arr, f_osc = compute_oscillator_strengths(
        cfg=cfg,
        x=x,
        dx=dx,
        num_states=num_found,
        psi0=psi0,
        energy=energies,
        osc_file=osc_path,
    )

    # Arquivo OscStr_SL_norm.txt (aberto e fechado vazio no Fortran original)
    if osc_norm_path:
        with open(osc_norm_path, "w"):
            pass

    # 6. Preparação do potencial para o contínuo e escrita de Potencial_SL.txt
    v[0] = VO
    v[-1] = VO
    if pot_path:
        with open(pot_path, "w") as f:
            for j in range(n):
                f.write(f"  {x[j] / NM:22.17f}        {1.0e3 * v[j] / E_CHARGE:22.14f}\n")

    # 7. Espectro de fotocorrente e transmissão no contínuo
    if num_found > 0:
        psi_ground = psi0[:, 0]
        E0 = energies[0]
    else:
        psi_ground = np.zeros(n, dtype=np.complex128)
        E0 = 0.0

    photon_energies_meV, photocurrent, transmission = compute_continuum_photocurrent(
        cfg=cfg,
        x=x,
        v=v,
        dx=dx,
        psi_ground=psi_ground,
        E0=E0,
        Ei=Ei_pc,
        Ef=Ef_pc,
        dE=dE_pc,
        Fd=Fd,
        pc_file=pc_path,
        trans_file=trans_path,
    )

    elapsed = time.time() - start_time
    if fim_path:
        with open(fim_path, "w") as f:
            f.write(f"   {elapsed:20.14f}\n")

    return SimulationResults(
        x=x,
        v=v,
        energies=energies,
        psi0=psi0,
        f_osc=f_osc,
        photon_energies_meV=photon_energies_meV,
        photocurrent=photocurrent,
        transmission=transmission,
        elapsed_time=elapsed,
    )


# ==============================================================================
# 8. INTERFACE DE LINHA DE COMANDO (compatível com a chamada do executável Fortran)
# ==============================================================================
def parse_fortran_float(val_str: str) -> float:
    """Converte strings como '2.5d0', '2.5D0' ou '2.5' para float."""
    cleaned = val_str.lower().replace("d", "e").strip().strip('"').strip("'")
    return float(cleaned)


def main():
    """
    Linha de comando aceitando os 8 parâmetros na mesma ordem do executável Fortran:
      1: RW (num_qb_left)
      2: RQWt (L_qw_left em nm)
      3: RQBt (L_qb_left em nm)
      4: MQWt (L_cqw em nm)
      5: LW (num_qb_right)
      6: LQWt (L_qw_right em nm)
      7: LQBt (L_qb_right em nm)
      8: out_folder
    """
    if len(sys.argv) < 9:
        print("Uso: python superlattice_solver.py RW RQWt RQBt MQWt LW LQWt LQBt out_folder")
        sys.exit(1)

    num_qb_left = int(sys.argv[1])
    L_qw_left = parse_fortran_float(sys.argv[2])
    L_qb_left = parse_fortran_float(sys.argv[3])
    L_cqw = parse_fortran_float(sys.argv[4])
    num_qb_right = int(sys.argv[5])
    L_qw_right = parse_fortran_float(sys.argv[6])
    L_qb_right = parse_fortran_float(sys.argv[7])
    sample_folder = sys.argv[8].strip().strip('"').strip("'")

    cfg = SuperlatticeConfig.from_nm(
        num_qb_left=num_qb_left,
        L_qw_left_nm=L_qw_left,
        L_qb_left_nm=L_qb_left,
        L_cqw_nm=L_cqw,
        num_qb_right=num_qb_right,
        L_qw_right_nm=L_qw_right,
        L_qb_right_nm=L_qb_right,
    )

    results = run_superlattice_simulation(cfg, sample_folder=sample_folder)
    print(f"Simulação concluída com sucesso em {results.elapsed_time:.2f} s.")
    print(f"Autoestados localizados: {len(results.energies)}")


if __name__ == "__main__":
    main()
