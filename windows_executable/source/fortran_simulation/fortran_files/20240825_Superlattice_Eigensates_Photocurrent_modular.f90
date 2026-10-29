!==============================================================================
! SIMULADOR MODULAR DE SUPER-REDES QUÂNTICAS (TRANSFER MATRIX & PHOTOCURRENT)
!
! Arquivo: 20240825_Superlattice_Eigensates_Photocurrent_modular.f90
!
! DESCRIÇÃO:
!   Cálculo autoconsistente de:
!   1. Perfil espacial de potencial V(x) e massa efetiva m(x).
!   2. Autoenergias e funções de onda ligadas (matriz de transferência e bissecção).
!   3. Forças de oscilador intersubbanda considerando efeitos de não-parabolicidade.
!   4. Espectro de fotocorrente fotovoltaica no contínuo (propagação coerente
!      e interação elétron-fóton).
!
! OBJETIVOS DESTA REFATORAÇÃO:
!   - Modularização estrita por responsabilidades físicas e numéricas.
!   - Eliminação do estado global oculto (sem variáveis compartilhadas em módulos).
!   - Parâmetros de entrada/saída explícitos (intent(in), intent(out), intent(inout)).
!   - Preservação exata da física, fórmulas e formatos de saída do original.
!   - Documentação clara em português para facilitar a futura tradução para Python / C++ / Julia.
!==============================================================================

!------------------------------------------------------------------------------
! MÓDULO 1: CONSTANTES FÍSICAS E PARÂMETROS DO MODELO
!------------------------------------------------------------------------------
module simulation_constants
    implicit none

    ! Definição padrão de precisão dupla (double precision)
    integer, parameter :: dp = selected_real_kind(15, 307)

    ! Constantes físicas fundamentais
    real(dp), parameter :: pi       = 3.14159265359_dp
    real(dp), parameter :: me       = 9.10938356E-31_dp   ! Massa do elétron livre [kg]
    real(dp), parameter :: hbar     = 1.054571817E-34_dp  ! Constante reduzida de Planck [J*s]
    real(dp), parameter :: e_charge = 1.602176634E-19_dp  ! Carga elementar [C]
    real(dp), parameter :: nm       = 1.0E-9_dp           ! Escala de nanômetros [m]
    complex(dp), parameter :: ic   = (0.0_dp, 1.0_dp)    ! Unidade imaginária

    ! Parâmetros dos materiais para a heteroestrutura InGaAs / InAlAs
    real(dp), parameter :: eo_InGaAs = 14.3_dp            ! Constante dielétrica do InGaAs
    real(dp), parameter :: eo_InAlAs = 12.7_dp            ! Constante dielétrica do InAlAs
    real(dp), parameter :: m_InGaAs  = 0.0436_dp * me     ! Massa efetiva no InGaAs [kg]
    real(dp), parameter :: m_InAlAs  = 0.0836_dp * me     ! Massa efetiva no InAlAs [kg]
    real(dp), parameter :: Vo        = 0.503_dp * e_charge! Altura da barreira de potencial [J]
    real(dp), parameter :: barreira  = 0.503_dp           ! Altura da barreira em [eV]

    ! Parâmetros de não-parabolicidade da massa efetiva (Nelson et al. PhysRevB.50.8663)
    real(dp), parameter :: Nonparabw = 1.3E-18_dp
    real(dp), parameter :: Ew        = hbar * hbar / (2.0_dp * m_InGaAs * Nonparabw) ! Gap do InGaAs [J]
    real(dp), parameter :: Eb        = Ew * m_InAlAs / m_InGaAs                      ! Gap do InAlAs [J]

    real(dp), parameter :: a0_InGaAs = 0.53_dp * eo_InGaAs / m_InGaAs
    real(dp), parameter :: a0_InAlAs = 0.53_dp * eo_InAlAs / m_InAlAs

    ! Parâmetros numéricos do algoritmo de bissecção
    real(dp), parameter :: tol              = e_charge * 5.0E-4_dp * 1.0E-12_dp
    integer, parameter  :: max_interactions = 1000

end module simulation_constants


!------------------------------------------------------------------------------
! MÓDULO 2: UTILIDADES MATEMÁTICAS E NUMÉRICAS
!------------------------------------------------------------------------------
module math_utils
    use simulation_constants, only: dp
    implicit none

contains

    !> Integração numérica pela regra do trapézio composta para arranjos complexos
    !> Parâmetros de entrada:
    !>   d:  Arranjo unidimensional complexo a ser integrado
    !>   ni: Índice inicial de integração
    !>   nf: Índice final de integração
    !>   dx: Passo espacial de amostragem [m]
    !>   n:  Tamanho total do arranjo
    !> Retorno:
    !>   trap_val: Valor complexo da integral
    function trap(d, ni, nf, dx, n) result(trap_val)
        integer, intent(in)          :: ni, nf, n
        real(dp), intent(in)         :: dx
        complex(dp), intent(in)      :: d(n)
        complex(dp)                  :: trap_val
        integer                      :: j

        trap_val = (0.0_dp, 0.0_dp)
        do j = ni + 2, nf - 1
            trap_val = trap_val + d(j) * dx
        end do
        trap_val = trap_val + 0.5_dp * (d(ni) + d(nf)) * dx
    end function trap

    !> Cálculo de derivadas espaciais centradas nos pontos interiores e diferenças finitas nos extremos
    !> Parâmetros:
    !>   y:  Arranjo de entrada (malha uniforme com passo dx)
    !>   dx: Passo da malha [m]
    !>   n:  Tamanho do arranjo
    !>   dy: Arranjo de saída com as derivadas espaciais dy/dx
    subroutine calculate_derivative_real(y, dx, n, dy)
        integer, intent(in)    :: n
        real(dp), intent(in)   :: dx
        real(dp), intent(in)   :: y(n)
        real(dp), intent(out)  :: dy(n)
        integer                :: k

        ! Diferença avançada no extremo esquerdo
        dy(1) = (y(2) - y(1)) / dx
        ! Diferença atrasada no extremo direito
        dy(n) = (y(n) - y(n-1)) / dx
        ! Diferenças centrais nos pontos interiores
        do k = 2, n - 1
            dy(k) = (y(k+1) - y(k-1)) / (2.0_dp * dx)
        end do
    end subroutine calculate_derivative_real

    !> Cálculo de derivadas espaciais para arranjos complexos (usado nas funções de onda)
    subroutine calculate_derivative_complex(y, dx, n, dy)
        integer, intent(in)       :: n
        real(dp), intent(in)      :: dx
        complex(dp), intent(in)   :: y(n)
        complex(dp), intent(out)  :: dy(n)
        integer                   :: k

        dy(1) = (y(2) - y(1)) / dx
        dy(n) = (y(n) - y(n-1)) / dx
        do k = 2, n - 1
            dy(k) = (y(k+1) - y(k-1)) / (2.0_dp * dx)
        end do
    end subroutine calculate_derivative_complex

end module math_utils


!------------------------------------------------------------------------------
! MÓDULO 3: GEOMETRIA DA SUPER-REDE, MALHA E PERFIS DE MATERIAIS
!------------------------------------------------------------------------------
module superlattice_structure
    use simulation_constants, only: dp, nm, me, Vo, Ew, Eb, m_InGaAs, m_InAlAs, e_charge
    implicit none

    ! Tipo derivado que encapsula todos os parâmetros geométricos da estrutura
    type :: superlattice_config
        integer  :: num_qb_left     ! Número de períodos poço/barreira no espelho de Bragg esquerdo
        real(dp) :: L_qw_left       ! Largura do poço quântico esquerdo [m]
        real(dp) :: L_qb_left       ! Largura da barreira quântica esquerda [m]
        real(dp) :: L_cqw           ! Largura do poço quântico central (defeito) [m]
        integer  :: num_qb_right    ! Número de períodos barreira/poço no espelho de Bragg direito
        real(dp) :: L_qw_right      ! Largura do poço quântico direito [m]
        real(dp) :: L_qb_right      ! Largura da barreira quântica direita [m]
    end type superlattice_config

contains

    !> Calcula os limites do domínio espacial L_left e L_right e o número de nós n
    !> Parâmetros de entrada:
    !>   cfg: Configuração da super-rede (dimensões em metros)
    !>   dx:  Passo espacial [m]
    !> Parâmetros de saída:
    !>   L_left:  Limite espacial esquerdo [m]
    !>   L_right: Limite espacial direito [m]
    !>   n:       Número de nós da malha
    !>   x:       Arranjo de coordenadas espaciais x(j) [m] (alocável)
    subroutine compute_grid_geometry(cfg, dx, L_left, L_right, n, x)
        type(superlattice_config), intent(in) :: cfg
        real(dp), intent(in)                  :: dx
        real(dp), intent(out)                 :: L_left, L_right
        integer, intent(out)                  :: n
        real(dp), allocatable, intent(out)    :: x(:)

        real(dp) :: L_bm_left, L_bm_right, L0_left, L0_right
        integer  :: j

        L_bm_left  = -real(cfg%num_qb_left, dp) * (cfg%L_qb_left + cfg%L_qw_left)
        L_bm_right =  real(cfg%num_qb_right, dp) * (cfg%L_qb_right + cfg%L_qw_right)

        if (-L_bm_left >= L_bm_right) then
            L0_left  = -50.0_dp * nm
            L_left   = L_bm_left + L0_left
            L0_right = -L_left - cfg%L_cqw - L_bm_right
            L_right  = cfg%L_cqw + L_bm_right + L0_right + cfg%L_cqw
        else
            L0_right = 50.0_dp * nm
            L_right  = cfg%L_cqw + L_bm_right + L0_right + cfg%L_cqw
            L0_left  = -L_right - L_bm_left
            L_left   = L_bm_left + L0_left
        end if

        n = int((L_right - L_left) / dx) + 2

        allocate(x(n))
        do j = 1, n
            x(j) = real(j - 1, dp) * dx + L_left
        end do
    end subroutine compute_grid_geometry

    !> Constrói o perfil de potencial V(x) e determina o ponto de junção x_cut
    !> Parâmetros de entrada:
    !>   cfg: Configuração geométrica
    !>   x:   Malha de coordenadas espaciais [m]
    !>   n:   Tamanho da malha
    !> Parâmetros de saída:
    !>   v:     Perfil de potencial eletrostático/heteroestrutura [J]
    !>   x_cut: Índice do nó de partição para a junção das funções de onda
    subroutine build_potential(cfg, x, n, v, x_cut)
        type(superlattice_config), intent(in) :: cfg
        integer, intent(in)                   :: n
        real(dp), intent(in)                  :: x(n)
        real(dp), intent(out)                 :: v(n)
        integer, intent(out)                  :: x_cut

        integer  :: i, j, k, n_qb_left, n_qb_right
        real(dp) :: L_bm_left_len, L0, L0_left_len

        L_bm_left_len = real(cfg%num_qb_left, dp) * (cfg%L_qb_left + cfg%L_qw_left)
        L0_left_len   = L_bm_left_len
        L0            = L_bm_left_len

        ! 1. Barreira externa esquerda
        j = 1
        do while (j <= n)
            if (x(j) <= -L0) then
                v(j) = Vo
                j = j + 1
            else
                exit
            end if
        end do

        ! 2. Super-rede à esquerda do poço central (poço / barreira)
        k = 1
        n_qb_left = 0
        i = j
        do while (i <= n)
            L0 = L0_left_len - real(n_qb_left, dp) * (cfg%L_qb_left + cfg%L_qw_left)
            if (x(i) >= -L0 .and. x(i) <= -L0 + cfg%L_qw_left) then
                v(i) = 0.0_dp
            else if (x(i) >= -L0 + cfg%L_qw_left .and. x(i) <= -L0 + cfg%L_qw_left + cfg%L_qb_left) then
                v(i) = Vo
            else
                v(i) = 0.0_dp
                n_qb_left = n_qb_left + 1
                k = k + 1
            end if
            if (k > cfg%num_qb_left) exit
            i = i + 1
        end do

        x_cut = i

        ! 3. Poço quântico central (defeito)
        j = i
        do while (j <= n)
            if (x(j) >= 0.0_dp .and. x(j) <= cfg%L_cqw) then
                v(j) = 0.0_dp
                j = j + 1
            else
                exit
            end if
        end do

        ! 4. Super-rede à direita do poço central (barreira / poço)
        k = 1
        n_qb_right = 0
        i = j
        do while (i <= n)
            L0 = cfg%L_cqw + real(n_qb_right, dp) * (cfg%L_qw_right + cfg%L_qb_right)
            if (x(i) >= L0 .and. x(i) <= L0 + cfg%L_qb_right) then
                v(i) = Vo
            else if (x(i) >= L0 + cfg%L_qb_right .and. x(i) <= L0 + cfg%L_qb_right + cfg%L_qw_right) then
                v(i) = 0.0_dp
            else
                v(i) = Vo
                n_qb_right = n_qb_right + 1
                k = k + 1
            end if
            if (k > cfg%num_qb_right) exit
            i = i + 1
        end do

        ! 5. Barreira externa direita
        do j = i, n
            v(j) = Vo
        end do

        ! Condições de contorno de potencial infinito nas extremidades para estados ligados
        v(1) = 10000.0_dp * e_charge
        v(n) = 10000.0_dp * e_charge
    end subroutine build_potential

    !> Constrói o perfil de massa efetiva m(x) ao longo da heteroestrutura
    !> Parámetros de entrada:
    !>   cfg:  Configuração geométrica
    !>   x:    Malha espacial [m]
    !>   n:    Número de nós
    !>   m_qw: Massa efetiva nos poços [kg]
    !>   m_qb: Massa efetiva nas barreiras [kg]
    !> Parâmetros de saída:
    !>   m:    Perfil de massa efetiva m(j) [kg]
    subroutine build_effective_mass(cfg, x, n, m_qw, m_qb, m)
        type(superlattice_config), intent(in) :: cfg
        integer, intent(in)                   :: n
        real(dp), intent(in)                  :: x(n), m_qw, m_qb
        real(dp), intent(out)                 :: m(n)

        integer  :: i, j, k, n_qb_left, n_qb_right
        real(dp) :: L_bm_left_len, L0, L0_left_len

        L_bm_left_len = real(cfg%num_qb_left, dp) * (cfg%L_qb_left + cfg%L_qw_left)
        L0_left_len   = L_bm_left_len
        L0            = L_bm_left_len

        ! 1. Primeira barreira esquerda
        j = 1
        do while (j <= n)
            if (x(j) <= -L0) then
                m(j) = m_qb
                j = j + 1
            else
                exit
            end if
        end do

        ! 2. Super-rede esquerda
        k = 1
        n_qb_left = 0
        i = j
        do while (i <= n)
            L0 = L0_left_len - real(n_qb_left, dp) * (cfg%L_qb_left + cfg%L_qw_left)
            if (x(i) >= -L0 .and. x(i) <= -L0 + cfg%L_qw_left) then
                m(i) = m_qw
            else if (x(i) >= -L0 + cfg%L_qw_left .and. x(i) <= -L0 + cfg%L_qw_left + cfg%L_qb_left) then
                m(i) = m_qb
            else
                m(i) = m_qw
                n_qb_left = n_qb_left + 1
                k = k + 1
            end if
            if (k > cfg%num_qb_left) exit
            i = i + 1
        end do

        ! 3. Poço quântico central
        j = i
        do while (j <= n)
            if (x(j) >= 0.0_dp .and. x(j) <= cfg%L_cqw) then
                m(j) = m_qw
                j = j + 1
            else
                exit
            end if
        end do

        ! 4. Super-rede direita
        k = 1
        n_qb_right = 0
        i = j
        do while (i <= n)
            L0 = cfg%L_cqw + real(n_qb_right, dp) * (cfg%L_qw_right + cfg%L_qb_right)
            if (x(i) >= L0 .and. x(i) <= L0 + cfg%L_qb_right) then
                m(i) = m_qb
            else if (x(i) >= L0 + cfg%L_qb_right .and. x(i) <= L0 + cfg%L_qb_right + cfg%L_qw_right) then
                m(i) = m_qw
            else
                m(i) = m_qb
                n_qb_right = n_qb_right + 1
                k = k + 1
            end if
            if (k > cfg%num_qb_right) exit
            i = i + 1
        end do

        ! 5. Barreira externa direita
        do j = i, n
            m(j) = m_qb
        end do
    end subroutine build_effective_mass

    !> Calcula as massas efetivas dependentes da energia (efeito não-parabólico)
    !> Parâmetros de entrada:
    !>   Ener: Energia do estado [J]
    !>   E0:   Energia do estado fundamental [J] (relevante apenas se is_photocurrent=.true.)
    !>   is_photocurrent: .true. se for cálculo no contínuo com energia do fóton Ener e estado ligado E0
    !> Parâmetros de saída:
    !>   m_qw: Massa corrigida no poço de InGaAs [kg]
    !>   m_qb: Massa corrigida na barreira de InAlAs [kg]
    subroutine get_nonparabolic_mass(Ener, E0, is_photocurrent, m_qw, m_qb)
        real(dp), intent(in)  :: Ener, E0
        logical, intent(in)   :: is_photocurrent
        real(dp), intent(out) :: m_qw, m_qb

        if (is_photocurrent) then
            m_qw = m_InGaAs * (1.0_dp + Ener / Ew)
            m_qb = m_InAlAs * (1.0_dp + (E0 + Ener - Vo) / Eb)
        else
            m_qw = m_InGaAs * (1.0_dp + Ener / Ew)
            m_qb = m_InAlAs * (1.0_dp + (Ener - Vo) / Eb)
        end if
    end subroutine get_nonparabolic_mass

end module superlattice_structure


!------------------------------------------------------------------------------
! MÓDULO 4: SOLUCIONADOR POR MATRIZ DE TRANSFERÊNCIA E AUTOESTADOS
!------------------------------------------------------------------------------
module transfer_matrix_solver
    use simulation_constants, only: dp, hbar, ic, tol, max_interactions, e_charge
    use math_utils, only: trap
    use superlattice_structure, only: superlattice_config, build_effective_mass, get_nonparabolic_mass
    implicit none

contains

    !> Calcula a matriz de transferência 2x2 para uma determinada energia
    !> Implementa as condições de contorno de BenDaniel-Duke em cada interface
    !> Parâmetros de entrada:
    !>   Ener: Energia [J]
    !>   dx:   Passo espacial da malha [m]
    !>   n:    Número de nós espaciais
    !>   v:    Perfil de potencial [J]
    !>   m:    Perfil de massa efetiva [kg]
    !> Parâmetros de saída:
    !>   T:    Matriz de transferência total 2x2 complexa
    subroutine compute_transfer_matrix(Ener, dx, n, v, m, T)
        real(dp), intent(in)     :: Ener, dx
        integer, intent(in)      :: n
        real(dp), intent(in)     :: v(n), m(n)
        complex(dp), intent(out) :: T(2,2)

        integer     :: i
        complex(dp) :: ki_i, ki_next, beta
        complex(dp) :: P_inicial(2,2), P_final(2,2), Dn(2,2), Ti(2,2), aux(2,2), T0(2,2)

        T0(1,1) = (1.0_dp, 0.0_dp); T0(1,2) = (0.0_dp, 0.0_dp)
        T0(2,1) = (0.0_dp, 0.0_dp); T0(2,2) = (1.0_dp, 0.0_dp)

        do i = 1, n - 1
            if (Ener > v(i)) then
                ki_i = sqrt(2.0_dp * m(i) * (Ener - v(i)) / (hbar * hbar))
            else
                ki_i = ic * sqrt(2.0_dp * m(i) * (v(i) - Ener) / (hbar * hbar))
            end if

            if (Ener > v(i+1)) then
                ki_next = sqrt(2.0_dp * m(i+1) * (Ener - v(i+1)) / (hbar * hbar))
            else
                ki_next = ic * sqrt(2.0_dp * m(i+1) * (v(i+1) - Ener) / (hbar * hbar))
            end if

            ! Matriz de propagação na camada atual
            P_inicial(1,1) = exp(ic * ki_i * (dx / 2.0_dp))
            P_inicial(1,2) = (0.0_dp, 0.0_dp)
            P_inicial(2,1) = (0.0_dp, 0.0_dp)
            P_inicial(2,2) = exp(-ic * ki_i * (dx / 2.0_dp))

            ! Matriz de propagação na camada seguinte
            P_final(1,1) = exp(ic * ki_next * (dx / 2.0_dp))
            P_final(1,2) = (0.0_dp, 0.0_dp)
            P_final(2,1) = (0.0_dp, 0.0_dp)
            P_final(2,2) = exp(-ic * ki_next * (dx / 2.0_dp))

            ! Matriz de descontinuidade (conservação da corrente de probabilidade)
            beta = (ki_i * m(i+1)) / (ki_next * m(i))
            Dn(1,1) = 0.5_dp * (1.0_dp + beta)
            Dn(1,2) = 0.5_dp * (1.0_dp - beta)
            Dn(2,1) = 0.5_dp * (1.0_dp - beta)
            Dn(2,2) = 0.5_dp * (1.0_dp + beta)

            aux = matmul(P_final, Dn)
            Ti  = matmul(aux, P_inicial)
            T   = matmul(Ti, T0)
            T0  = T
        end do
    end subroutine compute_transfer_matrix

    !> Cálculo da função de onda de um estado ligado por junção bidirecional (split)
    !> Integra da esquerda (1 -> x_cut) e da direita (n -> x_cut), ajusta em x_cut
    !> e normaliza a densidade de probabilidade integral(\psi^* \psi dx) = 1
    !> Parâmetros de entrada:
    !>   n:     Número de nós
    !>   x_cut: Índice do nó de junção
    !>   dx:    Passo espacial [m]
    !>   k:     Índice do autoestado (1, 2, ..., usado para paridade de junção (-1)^k)
    !>   Ener:  Autoenergia do estado [J]
    !>   v:     Perfil de potencial [J]
    !>   m:     Perfil de massa efetiva para esta energia [kg]
    !> Parâmetros de saída:
    !>   psi:   Função de onda complexa normalizada psi(x)
    subroutine compute_wavefunction_split(n, x_cut, dx, k, Ener, v, m, psi)
        integer, intent(in)       :: n, x_cut, k
        real(dp), intent(in)      :: dx, Ener
        real(dp), intent(in)      :: v(n), m(n)
        complex(dp), intent(out)  :: psi(n)

        integer     :: i
        real(dp)    :: soma
        complex(dp) :: ki_L_curr, ki_L_next, ki_R_curr, ki_R_prev, beta
        complex(dp) :: P_inicial_L(2,2), P_final_L(2,2), P_inicial_R(2,2), P_final_R(2,2)
        complex(dp) :: Dn(2,2), Ti(2,2), aux(2,2)
        complex(dp) :: WaveFunc_L(2,2), WaveFunc_R(2,2)
        complex(dp) :: psi_L(n), psi_R(n), prod(n)

        ! 1. Integração da esquerda para a direita (1 -> x_cut)
        WaveFunc_L(1,1) = (-1.0_dp, 0.0_dp); WaveFunc_L(1,2) = (0.0_dp, 0.0_dp)
        WaveFunc_L(2,1) = ( 1.0_dp, 0.0_dp); WaveFunc_L(2,2) = (0.0_dp, 0.0_dp)

        do i = 1, x_cut
            if (Ener > v(i)) then
                ki_L_curr = sqrt(2.0_dp * m(i) * (Ener - v(i)) / (hbar * hbar))
            else
                ki_L_curr = ic * sqrt(2.0_dp * m(i) * (v(i) - Ener) / (hbar * hbar))
            end if

            if (Ener > v(i+1)) then
                ki_L_next = sqrt(2.0_dp * m(i+1) * (Ener - v(i+1)) / (hbar * hbar))
            else
                ki_L_next = ic * sqrt(2.0_dp * m(i+1) * (v(i+1) - Ener) / (hbar * hbar))
            end if

            P_inicial_L(1,1) = exp(ic * ki_L_curr * (dx / 2.0_dp))
            P_inicial_L(1,2) = (0.0_dp, 0.0_dp)
            P_inicial_L(2,1) = (0.0_dp, 0.0_dp)
            P_inicial_L(2,2) = exp(-ic * ki_L_curr * (dx / 2.0_dp))

            P_final_L(1,1) = exp(ic * ki_L_next * (dx / 2.0_dp))
            P_final_L(1,2) = (0.0_dp, 0.0_dp)
            P_final_L(2,1) = (0.0_dp, 0.0_dp)
            P_final_L(2,2) = exp(-ic * ki_L_next * (dx / 2.0_dp))

            beta = (ki_L_curr * m(i+1)) / (ki_L_next * m(i))
            Dn(1,1) = 0.5_dp * (1.0_dp + beta)
            Dn(1,2) = 0.5_dp * (1.0_dp - beta)
            Dn(2,1) = 0.5_dp * (1.0_dp - beta)
            Dn(2,2) = 0.5_dp * (1.0_dp + beta)

            aux = matmul(P_final_L, Dn)
            Ti  = matmul(aux, P_inicial_L)
            WaveFunc_L = matmul(Ti, WaveFunc_L)
            psi_L(i) = WaveFunc_L(1,1) + WaveFunc_L(2,1)
        end do

        ! 2. Integração da direita para a esquerda (n -> x_cut com incremento -dx)
        WaveFunc_R(1,1) = (-1.0_dp, 0.0_dp); WaveFunc_R(1,2) = (0.0_dp, 0.0_dp)
        WaveFunc_R(2,1) = ( 1.0_dp, 0.0_dp); WaveFunc_R(2,2) = (0.0_dp, 0.0_dp)

        do i = n, x_cut, -1
            if (Ener > v(i)) then
                ki_R_curr = sqrt(2.0_dp * m(i) * (Ener - v(i)) / (hbar * hbar))
            else
                ki_R_curr = ic * sqrt(2.0_dp * m(i) * (v(i) - Ener) / (hbar * hbar))
            end if

            if (Ener > v(i-1)) then
                ki_R_prev = sqrt(2.0_dp * m(i-1) * (Ener - v(i-1)) / (hbar * hbar))
            else
                ki_R_prev = ic * sqrt(2.0_dp * m(i-1) * (v(i-1) - Ener) / (hbar * hbar))
            end if

            P_inicial_R(1,1) = exp(ic * ki_R_curr * (-dx / 2.0_dp))
            P_inicial_R(1,2) = (0.0_dp, 0.0_dp)
            P_inicial_R(2,1) = (0.0_dp, 0.0_dp)
            P_inicial_R(2,2) = exp(-ic * ki_R_curr * (-dx / 2.0_dp))

            P_final_R(1,1) = exp(ic * ki_R_prev * (-dx / 2.0_dp))
            P_final_R(1,2) = (0.0_dp, 0.0_dp)
            P_final_R(2,1) = (0.0_dp, 0.0_dp)
            P_final_R(2,2) = exp(-ic * ki_R_prev * (-dx / 2.0_dp))

            beta = (ki_R_curr * m(i-1)) / (ki_R_prev * m(i))
            Dn(1,1) = 0.5_dp * (1.0_dp + beta)
            Dn(1,2) = 0.5_dp * (1.0_dp - beta)
            Dn(2,1) = 0.5_dp * (1.0_dp - beta)
            Dn(2,2) = 0.5_dp * (1.0_dp + beta)

            aux = matmul(P_final_R, Dn)
            Ti  = matmul(aux, P_inicial_R)
            WaveFunc_R = matmul(Ti, WaveFunc_R)
            psi_R(i) = WaveFunc_R(1,1) + WaveFunc_R(2,1)
        end do

        ! 3. Ajuste de paridade e junção em x_cut
        do i = n, x_cut, -1
            psi_R(i) = ((-1.0_dp)**k) * psi_R(i)
        end do

        do i = 1, x_cut
            psi_L(i) = psi_L(i) * psi_R(x_cut) / psi_L(x_cut)
        end do

        ! Montagem da função de onda total
        do i = 1, x_cut
            psi(i) = psi_L(i)
        end do
        do i = x_cut, n
            psi(i) = psi_R(i)
        end do

        ! 4. Normalização: integral(|\psi|^2 dx) = 1
        prod(:) = conjg(psi(:)) * psi(:)
        soma = real(trap(prod, 1, n, dx, n), dp)
        psi(:) = psi(:) / sqrt(soma)
    end subroutine compute_wavefunction_split

    !> Busca de autoenergias ligadas por bissecção em T(2,2) e cálculo das funções de onda
    !> Parámetros de entrada:
    !>   cfg:             Configuração geométrica
    !>   x:               Malha espacial
    !>   v:               Perfil de potencial
    !>   n:               Tamanho da malha
    !>   dx:              Passo espacial [m]
    !>   x_cut:           Índice de junção
    !>   Ei, Ef, dE:      Intervalo e passo de energia para a varredura [meV]
    !>   max_states:      Número máximo de estados a calcular (nn)
    !>   file_unit_energy: Unidade de arquivo para registrar as energias encontradas (ou <= 0 para não gravar)
    !> Parâmetros de saída:
    !>   num_found: Quantidade de autoestados ligados localizados
    !>   energy:    Vetor de autoenergias [J] (dimensão max_states)
    !>   psi0:      Matriz de funções de onda complexas psi0(n, max_states)
    subroutine find_eigenstates(cfg, x, v, n, dx, x_cut, Ei, Ef, dE, max_states, &
                                num_found, energy, psi0, file_unit_energy)
        type(superlattice_config), intent(in) :: cfg
        integer, intent(in)                   :: n, x_cut, max_states, file_unit_energy
        real(dp), intent(in)                  :: dx, Ei, Ef, dE
        real(dp), intent(in)                  :: x(n), v(n)
        integer, intent(out)                  :: num_found
        real(dp), intent(out)                 :: energy(max_states)
        complex(dp), intent(out)              :: psi0(n, max_states)

        integer     :: jn, En, interaction, i
        real(dp)    :: Ener, Ener_old, Ener_new, T22_old, T22_new, mult2, m_qw, m_qb
        real(dp)    :: m(n)
        complex(dp) :: T(2,2), psi_single(n)

        En = int((Ef - Ei) / dE)
        num_found = 0

        ! Estado inicial em jn = 1
        Ener_old = (Ei + 1.0_dp * dE) * e_charge * 1.0E-3_dp
        Ener     = Ener_old
        call get_nonparabolic_mass(Ener, 0.0_dp, .false., m_qw, m_qb)
        call build_effective_mass(cfg, x, n, m_qw, m_qb, m)
        call compute_transfer_matrix(Ener, dx, n, v, m, T)
        T22_old = real(T(2,2), dp)

        ! Varredura em energia procurando cruzamentos por zero em T(2,2)
        do jn = 2, En
            Ener_new = (Ei + real(jn, dp) * dE) * e_charge * 1.0E-3_dp
            Ener     = Ener_new
            call get_nonparabolic_mass(Ener, 0.0_dp, .false., m_qw, m_qb)
            call build_effective_mass(cfg, x, n, m_qw, m_qb, m)
            call compute_transfer_matrix(Ener, dx, n, v, m, T)
            T22_new = real(T(2,2), dp)

            mult2 = T22_old * T22_new

            ! Cruzamento por zero detectado -> Refinamento por Bissecção
            if (mult2 < 0.0_dp) then
                interaction = 0
                do while (abs(Ener_new - Ener_old) > tol)
                    Ener = (Ener_old + Ener_new) / 2.0_dp
                    call get_nonparabolic_mass(Ener, 0.0_dp, .false., m_qw, m_qb)
                    call build_effective_mass(cfg, x, n, m_qw, m_qb, m)
                    call compute_transfer_matrix(Ener, dx, n, v, m, T)
                    T22_new = real(T(2,2), dp)

                    mult2 = T22_old * T22_new
                    if (mult2 < 0.0_dp) then
                        Ener_new = Ener
                    else
                        Ener_old = Ener
                    end if

                    interaction = interaction + 1
                    if (interaction > max_interactions) exit
                end do

                if (num_found < max_states) then
                    num_found = num_found + 1
                    energy(num_found) = Ener
                    if (file_unit_energy > 0) then
                        write(file_unit_energy, *) 1.0E3_dp * Ener / e_charge
                    end if
                end if
            end if

            ! Preparar próximo passo
            Ener     = (Ei + real(jn, dp) * dE) * e_charge * 1.0E-3_dp
            call get_nonparabolic_mass(Ener, 0.0_dp, .false., m_qw, m_qb)
            call build_effective_mass(cfg, x, n, m_qw, m_qb, m)
            call compute_transfer_matrix(Ener, dx, n, v, m, T)
            T22_old  = real(T(2,2), dp)
            Ener_old = Ener
        end do

        ! Cálculo das funções de onda para cada autoenergia encontrada
        do i = 1, num_found
            Ener = energy(i)
            call get_nonparabolic_mass(Ener, 0.0_dp, .false., m_qw, m_qb)
            call build_effective_mass(cfg, x, n, m_qw, m_qb, m)
            call compute_wavefunction_split(n, x_cut, dx, i, Ener, v, m, psi_single)
            psi0(:, i) = psi_single(:)
        end do
    end subroutine find_eigenstates

end module transfer_matrix_solver


!------------------------------------------------------------------------------
! MÓDULO 5: PROPRIEDADES ÓPTICAS (FORÇA DE OSCILADOR E FOTOCORRENTE)
!------------------------------------------------------------------------------
module optical_properties
    use simulation_constants, only: dp, hbar, ic, Vo, e_charge
    use math_utils, only: trap, calculate_derivative_real, calculate_derivative_complex
    use superlattice_structure, only: superlattice_config, build_effective_mass, get_nonparabolic_mass
    implicit none

contains

    !> Calcula as forças de oscilador intersubbanda f_{1 -> i} considerando não-parabolicidade
    !> Referência: Nelson et al. Phys. Rev. B 50, 8663 (1994)
    !> Parâmetros de entrada:
    !>   cfg:           Configuração geométrica
    !>   x:             Malha espacial
    !>   n:             Número de nós
    !>   dx:            Passo espacial [m]
    !>   num_states:    Número de estados disponíveis
    !>   psi0:          Funções de onda psi0(n, num_states)
    !>   energy:        Autoenergias ligadas [J]
    !>   file_unit_osc: Unidade de arquivo para gravar OscStr_SL.txt (ou <= 0 para não gravar)
    !> Parâmetros de saída:
    !>   f_osc: Arranjo de forças de oscilador calculadas para os estados i = 2 .. num_states
    subroutine compute_oscillator_strengths(cfg, x, n, dx, num_states, psi0, energy, f_osc, file_unit_osc)
        type(superlattice_config), intent(in) :: cfg
        integer, intent(in)                   :: n, num_states, file_unit_osc
        real(dp), intent(in)                  :: dx
        real(dp), intent(in)                  :: x(n), energy(num_states)
        complex(dp), intent(in)               :: psi0(n, num_states)
        real(dp), intent(out)                 :: f_osc(num_states)

        integer     :: i
        real(dp)    :: m_qw, m_qb, dE
        real(dp)    :: meff_ground(n), meff_state(n), mass_deri(n)
        complex(dp) :: psi0_deri(n), prod(n), soma

        f_osc(:) = 0.0_dp

        ! Massa efetiva do estado fundamental
        call get_nonparabolic_mass(energy(1), 0.0_dp, .false., m_qw, m_qb)
        call build_effective_mass(cfg, x, n, m_qw, m_qb, meff_ground)
        call calculate_derivative_real(meff_ground, dx, n, mass_deri)

        do i = 2, num_states
            call get_nonparabolic_mass(energy(i), 0.0_dp, .false., m_qw, m_qb)
            call build_effective_mass(cfg, x, n, m_qw, m_qb, meff_state)

            ! Derivada espacial da função de onda do estado excitado
            call calculate_derivative_complex(psi0(:, i), dx, n, psi0_deri)

            ! Integrando do elemento de matriz com correção por não-parabolicidade:
            ! prod = - psi_1^* psi_i (dm/dx)/m_1^2 + psi_1^* (dpsi_i/dx)/m_1 + psi_1^* (dpsi_i/dx)/m_i
            prod(:) = - conjg(psi0(:, 1)) * psi0(:, i) * mass_deri(:) / (meff_ground(:) * meff_ground(:)) &
                      + conjg(psi0(:, 1)) * psi0_deri(:) / meff_ground(:) &
                      + conjg(psi0(:, 1)) * psi0_deri(:) / meff_state(:)

            soma = trap(prod, 1, n, dx, n)
            dE   = energy(i) - energy(1)

            ! Força de oscilador adimensional f_i
            f_osc(i) = m_qw * (hbar * hbar) * (abs(soma)**2) / (2.0_dp * dE)

            if (file_unit_osc > 0) then
                write(file_unit_osc, *) 1.0E3_dp * dE / e_charge, f_osc(i)
            end if
        end do
    end subroutine compute_oscillator_strengths

    !> Espectro de fotocorrente fotovoltaica no contínuo (propagação coerente e interação elétron-fóton)
    !> Referência: Vinter et al. Phys. Rev. B 60, R13993 (1999)
    !> Parâmetros de entrada:
    !>   cfg:             Configuração geométrica
    !>   x:               Malha espacial
    !>   v:               Perfil de potencial adaptado para o contínuo (com V(1)=Vo e V(n)=Vo)
    !>   n:               Número de nós
    !>   dx:              Passo espacial [m]
    !>   psi_ground:      Função de onda do estado fundamental psi0(:, 1)
    !>   E0:              Energia do estado fundamental [J]
    !>   Ei, Ef, dE:      Intervalo espectral e passo de varredura do fóton [meV]
    !>   Fd:              Amplitude do campo elétrico externo [J/m]
    !>   file_unit_pc:    Unidade de arquivo para gravar Photocurrent_SL.txt
    !>   file_unit_trans: Unidade de arquivo para gravar Transmission_SL.txt
    subroutine compute_continuum_photocurrent(cfg, x, v, n, dx, psi_ground, E0, &
                                              Ei, Ef, dE, Fd, file_unit_pc, file_unit_trans)
        type(superlattice_config), intent(in) :: cfg
        integer, intent(in)                   :: n, file_unit_pc, file_unit_trans
        real(dp), intent(in)                  :: dx, E0, Ei, Ef, dE, Fd
        real(dp), intent(in)                  :: x(n), v(n)
        complex(dp), intent(in)               :: psi_ground(n)

        integer     :: jn, En, i
        real(dp)    :: Ener, m_qw, m_qb
        real(dp)    :: m(n)
        complex(dp) :: ki_curr, ki_next, beta, cte, beta_plus, beta_minus
        complex(dp) :: P_inicial(2,2), P_final(2,2), Dn(2,2), Ti(2,2), aux(2,2), T(2,2), T0(2,2)
        complex(dp) :: F(2), ctecur_left, ctecur_right, Je, Jd
        complex(dp) :: ki_arr(n)

        En = int((Ef - Ei) / dE)

        do jn = 1, En
            Ener = (Ei + real(jn, dp) * dE) * e_charge * 1.0E-3_dp  ! Energia do fóton [J]

            ! Correção de não-parabolicidade no contínuo
            call get_nonparabolic_mass(Ener, E0, .true., m_qw, m_qb)
            call build_effective_mass(cfg, x, n, m_qw, m_qb, m)

            T0(1,1) = (1.0_dp, 0.0_dp); T0(1,2) = (0.0_dp, 0.0_dp)
            T0(2,1) = (0.0_dp, 0.0_dp); T0(2,2) = (1.0_dp, 0.0_dp)

            F(1) = (0.0_dp, 0.0_dp)
            F(2) = (0.0_dp, 0.0_dp)

            ! Propagação regressiva do extremo direito n-1 até 1
            do i = n - 1, 1, -1
                ki_curr = sqrt(2.0_dp * m(i)   * (Ener + E0 - v(i))   / (hbar * hbar))
                ki_next = sqrt(2.0_dp * m(i+1) * (Ener + E0 - v(i+1)) / (hbar * hbar))
                ki_arr(i)   = ki_curr
                ki_arr(i+1) = ki_next

                P_inicial(1,1) = exp(-ic * ki_next * x(i))
                P_inicial(1,2) = (0.0_dp, 0.0_dp)
                P_inicial(2,1) = (0.0_dp, 0.0_dp)
                P_inicial(2,2) = exp( ic * ki_next * x(i))

                P_final(1,1) = exp( ic * ki_curr * x(i))
                P_final(1,2) = (0.0_dp, 0.0_dp)
                P_final(2,1) = (0.0_dp, 0.0_dp)
                P_final(2,2) = exp(-ic * ki_curr * x(i))

                beta = (ki_curr * m(i+1)) / (ki_next * m(i))
                Dn(1,1) = 0.5_dp * (1.0_dp + beta)
                Dn(1,2) = 0.5_dp * (1.0_dp - beta)
                Dn(2,1) = 0.5_dp * (1.0_dp - beta)
                Dn(2,2) = 0.5_dp * (1.0_dp + beta)

                aux = matmul(P_inicial, Dn)
                Ti  = matmul(aux, P_final)
                T   = matmul(T0, Ti)
                T0  = T

                ! Interação elétron-fóton na interface i
                cte = (m(i) * Fd) / (2.0_dp * ic * (hbar * hbar) * ki_curr)
                beta_plus  = -cte * (exp( ic * ki_curr * x(i))   * x(i)   * psi_ground(i) &
                                   - exp( ic * ki_curr * x(i+1)) * x(i+1) * psi_ground(i+1)) * dx
                beta_minus =  cte * (exp(-ic * ki_curr * x(i))   * x(i)   * psi_ground(i) &
                                   - exp(-ic * ki_curr * x(i+1)) * x(i+1) * psi_ground(i+1)) * dx

                F(1) = F(1) + (T(1,1) * beta_minus + T(1,2) * beta_plus)
                F(2) = F(2) + (T(2,1) * beta_minus + T(2,2) * beta_plus)
            end do

            ! Densidades de corrente para a esquerda (Je) e para a direita (Jd)
            ctecur_left  = (hbar * ki_arr(1)) / m(1)
            ctecur_right = (hbar * ki_arr(n)) / m(n)

            Je = ctecur_left  * Fd * (-F(2) / T(2,2)) * conjg(-F(2) / T(2,2))
            Jd = ctecur_right * Fd * (F(1) - F(2) * T(1,2) / T(2,2)) * conjg(F(1) - F(2) * T(1,2) / T(2,2))

            if (file_unit_pc > 0) then
                write(file_unit_pc, *) 1.0E3_dp * Ener / e_charge, real(Jd - Je, dp)
            end if

            if (file_unit_trans > 0) then
                write(file_unit_trans, *) 1.0E3_dp * Ener / e_charge, real(1.0_dp / abs(T(1,1)), dp)
            end if
        end do
    end subroutine compute_continuum_photocurrent

end module optical_properties


!==============================================================================
! PROGRAMA PRINCIPAL: ORQUESTRAÇÃO MODULAR DA SIMULAÇÃO
!==============================================================================
program Photocurrent_Transfer_Matrix_Modular
    use simulation_constants, only: dp, nm, e_charge, Vo
    use superlattice_structure, only: superlattice_config, compute_grid_geometry, &
                                     build_potential, build_effective_mass
    use transfer_matrix_solver, only: find_eigenstates
    use optical_properties, only: compute_oscillator_strengths, compute_continuum_photocurrent
    implicit none

    ! Parâmetros da simulação
    type(superlattice_config) :: cfg
    character(len=400)        :: sample_folder
    character(len=100)        :: arg
    real(dp)                  :: dx, Fd, total_start, total_finish
    integer                   :: n, nn, num_states_found, x_cut, i, j

    ! Arranjos dinâmicos do domínio
    real(dp), allocatable     :: x(:), v(:), m(:), energy(:), f_osc(:)
    complex(dp), allocatable  :: psi0(:,:)

    ! -------------------------------------------------------------------------
    ! 1. LEITURA DOS ARGUMENTOS DE LINHA DE COMANDO
    ! -------------------------------------------------------------------------
    call cpu_time(total_start)

    call get_command_argument(1, arg); read(arg, *) cfg%num_qb_left
    call get_command_argument(2, arg); read(arg, *) cfg%L_qw_left
    call get_command_argument(3, arg); read(arg, *) cfg%L_qb_left
    call get_command_argument(4, arg); read(arg, *) cfg%L_cqw
    call get_command_argument(5, arg); read(arg, *) cfg%num_qb_right
    call get_command_argument(6, arg); read(arg, *) cfg%L_qw_right
    call get_command_argument(7, arg); read(arg, *) cfg%L_qb_right
    call get_command_argument(8, sample_folder)

    ! Conversão de dimensões de nanômetros para metros [m]
    cfg%L_qw_left  = cfg%L_qw_left  * nm
    cfg%L_qb_left  = cfg%L_qb_left  * nm
    cfg%L_cqw      = cfg%L_cqw      * nm
    cfg%L_qw_right = cfg%L_qw_right * nm
    cfg%L_qb_right = cfg%L_qb_right * nm

    dx = 0.1_dp * nm               ! Passo espacial da discretização [m]
    nn = 100                       ! Número máximo de autoestados
    Fd = 1.0E5_dp * e_charge       ! Campo elétrico externo de teste [J/m]

    ! -------------------------------------------------------------------------
    ! 2. ABERTURA DOS ARQUIVOS DE SAÍDA (Compatibilidade exata com a interface GUI)
    ! -------------------------------------------------------------------------
    open(26, file=trim(sample_folder)//'wavefunction_SL.txt')
    open(36, file=trim(sample_folder)//'Energy_SL.txt')
    open(7,  file=trim(sample_folder)//'Transmission_SL.txt')
    open(8,  file=trim(sample_folder)//'Photocurrent_SL.txt')
    open(25, file=trim(sample_folder)//'Potencial_SL.txt')
    open(28, file=trim(sample_folder)//'OscStr_SL.txt')
    open(29, file=trim(sample_folder)//'OscStr_SL_norm.txt')

    ! -------------------------------------------------------------------------
    ! 3. CONSTRUÇÃO DA MALHA ESPACIAL E PERFIS INICIAIS
    ! -------------------------------------------------------------------------
    block
        real(dp) :: L_left, L_right
        call compute_grid_geometry(cfg, dx, L_left, L_right, n, x)
    end block

    allocate(v(n), m(n), energy(nn), f_osc(nn), psi0(n, nn))

    call build_potential(cfg, x, n, v, x_cut)

    ! -------------------------------------------------------------------------
    ! 4. CÁLCULO DOS AUTOESTADOS LIGADOS E FUNÇÕES DE ONDA (TRANSFER MATRIX)
    ! -------------------------------------------------------------------------
    call find_eigenstates(cfg, x, v, n, dx, x_cut, &
                          0.0_dp, 700.0_dp, 0.1_dp, nn, &
                          num_states_found, energy, psi0, 36)

    ! Escrita das funções de onda para plotagem gráfica
    do i = 1, n
        write(26, *) real(x(i) / nm, dp), &
            ((real(psi0(i, j)**2, dp) + aimag(psi0(i, j)**2)) * 0.4E-6_dp + 1.0E3_dp * energy(j) / e_charge, j = 1, nn)
    end do

    ! -------------------------------------------------------------------------
    ! 5. CÁLCULO DAS FORÇAS DE OSCILADOR
    ! -------------------------------------------------------------------------
    call compute_oscillator_strengths(cfg, x, n, dx, nn, psi0, energy, f_osc, 28)

    ! -------------------------------------------------------------------------
    ! 6. PREPARAÇÃO DO POTENCIAL E ESCRITA PARA O CONTÍNUO
    ! -------------------------------------------------------------------------
    v(1) = Vo
    v(n) = Vo
    do j = 1, n
        write(25, *) x(j) / nm, 1.0E3_dp * v(j) / e_charge
    end do

    ! -------------------------------------------------------------------------
    ! 7. ESPECTRO DE FOTOCORRENTE E TRANSMISSÃO NO CONTÍNUO
    ! -------------------------------------------------------------------------
    call compute_continuum_photocurrent(cfg, x, v, n, dx, psi0(:, 1), energy(1), &
                                        0.0_dp, 500.0_dp, 0.1_dp, Fd, 8, 7)

    ! -------------------------------------------------------------------------
    ! 8. FECHAMENTO DOS ARQUIVOS E REGISTRO DO TEMPO DE EXECUÇÃO
    ! -------------------------------------------------------------------------
    close(26)
    close(36)
    close(7)
    close(8)
    close(25)
    close(28)
    close(29)

    open(30, file=trim(sample_folder)//'FimPrograma.txt')
    call cpu_time(total_finish)
    write(30, *) total_finish - total_start
    close(30)

    stop
end program Photocurrent_Transfer_Matrix_Modular
