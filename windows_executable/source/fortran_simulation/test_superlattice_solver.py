"""
Conjunto de Testes Unitários para o Simulador de Super-redes Quânticas
Arquivo: test_superlattice_solver.py

Testa individualmente cada função e sub-rotina traduzida de Fortran para Python:
1. Constantes físicas e do modelo.
2. Utilidades matemáticas (trap, derivadas real e complexa).
3. Geometria da malha, perfil de potencial e perfil de massa efetiva.
4. Não-parabolicidade da massa efetiva.
5. Solucionador da matriz de transferência (propriedades físicas e determinantes).
6. Funções de onda por junção bidirecional (continuidade e normalização unitária).
7. Localização de autoenergias ligadas por bissecção (assert_allclose contra referência).
8. Forças de oscilador intersubbanda.
9. Fotocorrente e transmissão no contínuo.
10. Execução de ponta a ponta gerando todos os arquivos de saída.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
from numpy.testing import assert_allclose

# Adiciona a raiz do projeto ao sys.path se necessário
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from windows_executable.source.fortran_simulation.superlattice_solver import (
    PI, ME, HBAR, E_CHARGE, NM, IC, VO, EW, EB, M_INGAAS, M_INALAS,
    SuperlatticeConfig,
    trap,
    calculate_derivative_real,
    calculate_derivative_complex,
    compute_grid_geometry,
    build_potential,
    build_effective_mass,
    get_nonparabolic_mass,
    compute_transfer_matrix,
    compute_wavefunction_split,
    find_eigenstates,
    compute_oscillator_strengths,
    compute_continuum_photocurrent,
    run_superlattice_simulation,
)


class TestSuperlatticeSolver(unittest.TestCase):
    """Testes unitários detalhados para o solucionador quântico modular."""

    def setUp(self):
        """Configuração de referência padrão para uma super-rede simétrica."""
        self.cfg = SuperlatticeConfig.from_nm(
            num_qb_left=2,
            L_qw_left_nm=3.0,
            L_qb_left_nm=4.0,
            L_cqw_nm=5.0,
            num_qb_right=2,
            L_qw_right_nm=3.0,
            L_qb_right_nm=4.0,
        )
        self.dx = 0.1 * NM

    # --------------------------------------------------------------------------
    # 1. Constantes físicas
    # --------------------------------------------------------------------------
    def test_physical_constants(self):
        """Valida que as constantes físicas e energias de gap obedecem às definições físicas."""
        self.assertAlmostEqual(PI, 3.14159265359, places=9)
        self.assertAlmostEqual(E_CHARGE, 1.602176634e-19, places=25)
        self.assertAlmostEqual(VO / E_CHARGE, 0.503, places=6)
        self.assertTrue(EW > 0.0)
        self.assertTrue(EB > EW)
        self.assertAlmostEqual(EB / EW, M_INALAS / M_INGAAS, places=6)

    # --------------------------------------------------------------------------
    # 2. Utilidades matemáticas
    # --------------------------------------------------------------------------
    def test_trap_rule_analytical(self):
        """
        Testa a integração por trapézio em função constante e linear.
        trap(d, 0, n, dx) = sum_{j=2}^{n-2} d(j)*dx + 0.5*(d(0) + d(n-1))*dx
        """
        n = 100
        dx = 0.01
        # Para uma constante c, sum de n-3 termos + 0.5*(c+c) = (n-2)*c*dx
        c = 3.5 + 2.0j
        arr = np.full(n, c, dtype=np.complex128)
        val = trap(arr, 0, n, dx)
        expected = (n - 2) * c * dx
        assert_allclose(val, expected, rtol=1e-12, atol=1e-12)

    def test_calculate_derivative_real_sin(self):
        """Testa diferenciação real de sin(k x) comparando com k cos(k x)."""
        n = 500
        x = np.linspace(0.0, 2.0 * np.pi, n)
        dx = x[1] - x[0]
        k = 2.0
        y = np.sin(k * x)
        dy_num = calculate_derivative_real(y, dx)
        dy_ana = k * np.cos(k * x)
        # O erro das diferenças finitas nos pontos interiores é O(dx^2)
        assert_allclose(dy_num[2:-2], dy_ana[2:-2], rtol=1e-3, atol=1e-3)

    def test_calculate_derivative_complex_exp(self):
        """Testa diferenciação complexa de exp(i k x) comparando com i k exp(i k x)."""
        n = 500
        x = np.linspace(0.0, 1.0, n)
        dx = x[1] - x[0]
        k = 3.0
        y = np.exp(1j * k * x)
        dy_num = calculate_derivative_complex(y, dx)
        dy_ana = 1j * k * y
        assert_allclose(dy_num[2:-2], dy_ana[2:-2], rtol=1e-3, atol=1e-3)

    # --------------------------------------------------------------------------
    # 3. Geometria da malha e perfis
    # --------------------------------------------------------------------------
    def test_compute_grid_geometry(self):
        """Verifica os limites espaciais L_left, L_right e o vetor de malha x."""
        L_left, L_right, n, x = compute_grid_geometry(self.cfg, self.dx)
        self.assertAlmostEqual(L_left / NM, -64.0, places=4)
        self.assertAlmostEqual(L_right / NM, 69.0, places=4)
        self.assertEqual(len(x), n)
        self.assertAlmostEqual(x[0] / NM, -64.0, places=4)
        self.assertAlmostEqual(x[-1] / NM, 69.0, places=4)
        # Espaçamento uniforme
        diffs = np.diff(x)
        assert_allclose(diffs, self.dx, rtol=1e-6)

    def test_build_potential_boundaries(self):
        """Valida que o potencial possui barreiras infinitas nas pontas e patamares corretos."""
        _, _, n, x = compute_grid_geometry(self.cfg, self.dx)
        v, x_cut = build_potential(self.cfg, x)

        self.assertEqual(len(v), n)
        # Pontas com 10000 eV
        self.assertAlmostEqual(v[0], 10000.0 * E_CHARGE, places=4)
        self.assertAlmostEqual(v[-1], 10000.0 * E_CHARGE, places=4)

        # x_cut deve apontar para o início do poço central (x ~ 0 nm)
        self.assertTrue(0 < x_cut < n)
        self.assertTrue(-0.5 * NM <= x[x_cut] <= 0.5 * NM)

        # Região do poço central (0 < x < L_cqw) deve ter potencial zero
        mask_central = (x > 0.5 * NM) & (x < (self.cfg.L_cqw - 0.5 * NM))
        self.assertTrue(np.all(v[mask_central] == 0.0))

    def test_build_effective_mass(self):
        """Verifica se as massas nos poços e barreiras correspondem aos valores passados."""
        _, _, n, x = compute_grid_geometry(self.cfg, self.dx)
        m = build_effective_mass(self.cfg, x, M_INGAAS, M_INALAS)
        self.assertEqual(len(m), n)
        # Poço central
        mask_central = (x > 0.5 * NM) & (x < (self.cfg.L_cqw - 0.5 * NM))
        assert_allclose(m[mask_central], M_INGAAS, rtol=1e-10)
        # Barreira externa esquerda
        self.assertAlmostEqual(m[10], M_INALAS, places=25)

    def test_nonparabolic_mass(self):
        """Testa o cálculo da correção não-parabólica da massa efetiva."""
        Ener = 100.0 * 1e-3 * E_CHARGE  # 100 meV
        m_qw, m_qb = get_nonparabolic_mass(Ener, E0=0.0, is_photocurrent=False)
        self.assertTrue(m_qw > M_INGAAS)  # Massa aumenta com a energia
        self.assertTrue(m_qb < M_INALAS)  # Para Ener < Vo, Ener - Vo < 0

    # --------------------------------------------------------------------------
    # 4. Matriz de transferência
    # --------------------------------------------------------------------------
    def test_transfer_matrix_determinant(self):
        """
        Em qualquer meio estratificado conservativo (conservação de fluxo de probabilidade),
        o determinante da matriz de transferência de BenDaniel-Duke deve ter módulo unitário:
        det(T) = k_final * m_initial / (k_initial * m_final).
        Para uma barreira simétrica com mesmas condições nas pontas, det(T) = 1.
        """
        L_left, L_right, n, x = compute_grid_geometry(self.cfg, self.dx)
        v, _ = build_potential(self.cfg, x)
        # Remove barreiras infinitas para testar determinante físico padrão
        v_phys = v.copy()
        v_phys[0] = VO
        v_phys[-1] = VO

        Ener = 250.0 * 1e-3 * E_CHARGE
        m_qw, m_qb = get_nonparabolic_mass(Ener, 0.0, False)
        m = build_effective_mass(self.cfg, x, m_qw, m_qb)

        T = compute_transfer_matrix(Ener, self.dx, v_phys, m)
        det_T = np.linalg.det(T)
        self.assertAlmostEqual(abs(det_T), 1.0, places=4)

    # --------------------------------------------------------------------------
    # 5. Funções de onda e normalização
    # --------------------------------------------------------------------------
    def test_wavefunction_normalization(self):
        """Valida que a função de onda calculada pela sub-rotina split é normalizada a 1."""
        _, _, n, x = compute_grid_geometry(self.cfg, self.dx)
        v, x_cut = build_potential(self.cfg, x)

        Ener = 123.238 * 1e-3 * E_CHARGE
        m_qw, m_qb = get_nonparabolic_mass(Ener, 0.0, False)
        m = build_effective_mass(self.cfg, x, m_qw, m_qb)

        psi = compute_wavefunction_split(n, x_cut, self.dx, k=1, Ener=Ener, v=v, m=m)
        prod = np.conj(psi) * psi
        integral = trap(prod, 0, n, self.dx)

        self.assertAlmostEqual(integral.real, 1.0, places=6)
        self.assertAlmostEqual(integral.imag, 0.0, places=8)

    # --------------------------------------------------------------------------
    # 6. Autoestados e bissecção
    # --------------------------------------------------------------------------
    def test_find_eigenstates_bound_states(self):
        """
        Testa a busca de autoenergias ligadas por bissecção até 220 meV.
        Garante que encontra os primeiros estados ligados e compara com as referências.
        """
        _, _, n, x = compute_grid_geometry(self.cfg, self.dx)
        v, x_cut = build_potential(self.cfg, x)

        num_found, energies, psi0 = find_eigenstates(
            cfg=self.cfg,
            x=x,
            v=v,
            dx=self.dx,
            x_cut=x_cut,
            Ei=0.0,
            Ef=220.0,
            dE=0.1,
            max_states=10,
        )

        self.assertGreaterEqual(num_found, 4)
        energies_meV = energies * 1.0e3 / E_CHARGE

        # Referências obtidas com o executável Fortran original:
        ref_energies = [123.23809484, 202.50244310, 202.91307984, 215.65507403]
        assert_allclose(energies_meV[:4], ref_energies, rtol=1e-5, atol=1e-4)

    # --------------------------------------------------------------------------
    # 7. Forças de oscilador
    # --------------------------------------------------------------------------
    def test_oscillator_strengths_positive(self):
        """Verifica que as forças de oscilador calculadas são positivas e consistentes."""
        _, _, n, x = compute_grid_geometry(self.cfg, self.dx)
        v, x_cut = build_potential(self.cfg, x)

        num_found, energies, psi0 = find_eigenstates(
            cfg=self.cfg,
            x=x,
            v=v,
            dx=self.dx,
            x_cut=x_cut,
            Ei=0.0,
            Ef=220.0,
            dE=0.1,
            max_states=5,
        )

        dE_arr, f_osc = compute_oscillator_strengths(self.cfg, x, self.dx, num_found, psi0, energies)
        self.assertEqual(len(f_osc), num_found - 1)
        # As forças de oscilador devem ser não-negativas
        self.assertTrue(np.all(f_osc >= 0.0))
        # Referência do primeiro estado excitado f_osc[0] ~ 0.0321
        self.assertAlmostEqual(f_osc[0], 0.03211, places=3)

    # --------------------------------------------------------------------------
    # 8. Fotocorrente e transmissão no contínuo
    # --------------------------------------------------------------------------
    def test_continuum_photocurrent(self):
        """Valida que o cálculo de transmissão e fotocorrente executa sem divergências."""
        _, _, n, x = compute_grid_geometry(self.cfg, self.dx)
        v, _ = build_potential(self.cfg, x)
        v[0] = VO
        v[-1] = VO

        psi_ground = np.ones(n, dtype=np.complex128) / np.sqrt(n * self.dx)
        E0 = 123.2 * 1e-3 * E_CHARGE

        energies_meV, pc, tr = compute_continuum_photocurrent(
            cfg=self.cfg,
            x=x,
            v=v,
            dx=self.dx,
            psi_ground=psi_ground,
            E0=E0,
            Ei=0.0,
            Ef=10.0,
            dE=1.0,
            Fd=1.0e5 * E_CHARGE,
        )

        self.assertEqual(len(energies_meV), 10)
        self.assertEqual(len(pc), 10)
        self.assertEqual(len(tr), 10)
        # Transmissão deve ser estritamente positiva e menor ou igual a 1
        self.assertTrue(np.all(tr > 0.0))
        self.assertTrue(np.all(tr <= 1.0))

    # --------------------------------------------------------------------------
    # 9. Teste de ponta a ponta (arquivos gerados)
    # --------------------------------------------------------------------------
    def test_end_to_end_simulation_files(self):
        """Executa simulação completa em diretório temporário e verifica geração dos 8 arquivos."""
        with tempfile.TemporaryDirectory() as tmpdir:
            res = run_superlattice_simulation(
                cfg=self.cfg,
                sample_folder=tmpdir,
                dx=self.dx,
                nn=5,
                Ei_bound=0.0,
                Ef_bound=205.0,
                dE_bound=0.2,
                Ei_pc=0.0,
                Ef_pc=5.0,
                dE_pc=1.0,
            )

            expected_files = [
                "wavefunction_SL.txt",
                "Energy_SL.txt",
                "Transmission_SL.txt",
                "Photocurrent_SL.txt",
                "Potencial_SL.txt",
                "OscStr_SL.txt",
                "OscStr_SL_norm.txt",
                "FimPrograma.txt",
            ]

            for fname in expected_files:
                fpath = os.path.join(tmpdir, fname)
                self.assertTrue(os.path.exists(fpath), f"Arquivo {fname} não foi criado.")
                # Arquivos normais não devem ser vazios (exceto OscStr_SL_norm.txt)
                if fname != "OscStr_SL_norm.txt":
                    self.assertGreater(os.path.getsize(fpath), 0, f"Arquivo {fname} está vazio.")

            self.assertGreaterEqual(len(res.energies), 1)
            self.assertTrue(res.elapsed_time > 0.0)


if __name__ == "__main__":
    unittest.main()
