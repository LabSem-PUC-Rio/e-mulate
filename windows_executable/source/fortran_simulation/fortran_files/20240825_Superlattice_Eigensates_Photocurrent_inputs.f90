!---------------------------------------------------------------------------
!					Quinta versao do calculo de fotocorrente
! Nesta versao o metodo de calculados das funcoes de ondas e suas respectivas energias foi trocado para matriz transferencia
!---------------------------------------------------------------------------

module var
!---------------------------------------------------
!                 input
!-----------------------------------------------------
real*8, parameter    :: pi = 3.14159265359d0
real*8, parameter    :: me = 9.10938356E-31 ![kg]
real*8, parameter    :: hbar = 1.054571817E-34 ![J][s]
real*8, parameter    :: e_charge = 1.602176634E-19 ![C]

real*8, parameter    :: eo_InGaAs = 14.3d0 !constante dieletrica InGaAs
real*8, parameter    :: eo_InAlAs = 12.7d0 !constante dieletrica InAlAs
real*8, parameter    :: m_InGaAs = 0.0436d0*me 
real*8, parameter    :: m_InAlAs = 0.0836d0*me 
real*8, parameter    :: nm = 1E-9 ![m]
real*8, parameter    :: Vo=0.503d0*e_charge ![J]
real*8, parameter    :: barreira=0.503d0 ![eV]
					 !fonte:DOI: 10.1103/PhysRevB.67.085318	
real*8, parameter    :: Nonparabw = 1.3E-18
real*8, parameter    :: Ew = hbar*hbar/(2*m_InGaAs*Nonparabw)    ! gap energy of InGaAs in J! 0.880*e_charge!
real*8, parameter    :: Eb = Ew*m_InAlAs/m_InGaAs                ! gap energy of InAlAs in J ! 1.490*e_charge!

real*8, parameter    :: a0_InGaAs = 0.53d0*eo_InGaAs/m_InGaAs 
real*8, parameter    :: a0_InAlAs = 0.53d0*eo_InAlAs/m_InAlAs 

real*8, parameter    :: tol=e_charge*5E-4*1.0d-12 ! tolerancia
integer, parameter   :: max_interactions = 1000

complex*8, parameter :: ic = (0.d0,1.d0)
real*8, allocatable  :: v(:),  x(:), m(:)
END MODULE var
!--------------------------------------------------------------------------
!--------------------------------------------------------------------------
program Photocurrent_Transfer_Matrix
use var
implicit none
integer        :: j, k, jn, i, En, ne, x_cut, num_QB_left, num_QB_right, n, nn
real*8         :: ener,dE, E_initial, E_final, E0, E0a, fs, start, finish, dx, total_finish, total_start
real*8 	       :: m_qw, m_qb, efs_InGaAs, efs_InAlAs, a0, ry, Fd, soma, largest
real*8         :: L0_left, L_qb_left, L_qw_left, L_bm_left, L_left, L_cqw, L_qb_right, L_qw_right, L_bm_right, L0_right, L_right
double complex :: aux1, beta, Je, Jd, trap, J_norm
double complex :: P_inicial(2,2), P_final(2,2), Dn(2,2), T(2,2), AUX(2,2), AUX2(2,2), T0(2,2), Ti(2,2)
double complex :: W(2,2), W0(2,2), Wi(2,2), Wf(2,2), F(2), beta_minus, beta_plus, cte, ctecur_left, ctecur_right
character*1024   :: char_QBl, char_Lqw_l, char_Lqb_l, char_cqw, char_QBr, char_Lqw_r, char_Lqb_r, SL
character (len = 50) :: SL_info, SL_folder
character(len=20) :: name
real*8         :: factor_r
real*8         :: aux_r1, aux_r2
character(len=400) :: sample_folder
character(len=100) :: arg

real*8, allocatable  		 :: energy(:)
double complex, allocatable  :: psi(:), prod(:), ki(:), psi0(:,:) 

m_qw = m_InGaAs
m_qb = m_InAlAs
Fd = 1*1E5*e_charge ! Amplitude of the external electric field [J]/[M]

call cpu_time(total_start)
! ------------------------------------------------------------------
!                 dimension of the heterostructure [m]
! ------------------------------------------------------------------

! quantum bragg mirror to left of central quantum well
call get_command_argument(1, arg)
read(arg, *) num_QB_left			!Number of well/barrier of the Bragg mirror

call get_command_argument(2, arg)
read(arg, *) L_qw_left				!Width of lateral quantum well [nm]

call get_command_argument(3, arg)
read(arg, *) L_qb_left				!Width of quantum barrier [nm]


! central quantum well
call get_command_argument(4, arg)
read(arg, *) L_cqw					!Width of central quantum well [nm]

! quantum bragg mirror to right of central quantum well
call get_command_argument(5, arg)
read(arg, *) num_QB_right			!Number of well/barrier of the Bragg mirror

call get_command_argument(6, arg)
read(arg, *) L_qw_right				!Width of lateral quantum well [nm]

call get_command_argument(7, arg)
read(arg, *) L_qb_right				!Width of quantum barrier [nm]

call get_command_argument(8, sample_folder)  ! No es necesario leer el string, ya está en text_input


dx           = 0.1*nm   ! step size dx
nn           = 100       ! The number of states you want to calculate
!------------------------------------------------------------------
! Transformando as informacoes da super-rede(numero de pocos/barreiras e espessuras) em caracteres para criar a pasta da simulacao  

if(num_QB_left.lt.10)then 
	write(char_QBl,'(I1)') num_QB_left
else
	write(char_QBl,'(I2)') num_QB_left
end if

if(L_qw_left.lt.10)then
	write(char_Lqw_l,'(f4.2)')L_qw_left
else
	write(char_Lqw_l,'(f5.2)')L_qw_left
end if

if(L_qb_left.lt.10)then
	write(char_Lqb_l,'(f4.2)')L_qb_left
else
	write(char_Lqb_l,'(f5.2)')L_qb_left
end if

if(L_cqw.lt.10)then
	write(char_cqw,'(f4.2)')L_cqw
else
	write(char_cqw,'(f5.2)')L_cqw
end if

if(num_QB_right.lt.10)then
	write(char_QBr,'(I1)') num_QB_right
else 
	write(char_QBr,'(I2)') num_QB_right
end if

if(L_qb_right.lt.10)then
	write(char_Lqb_r,'(f4.2)')L_qb_right
else
	write(char_Lqb_r,'(f5.2)')L_qb_right
end if

if(L_qw_right.lt.10)then
	write(char_Lqw_r,'(f4.2)')L_qw_right
else
	write(char_Lqw_r,'(f5.2)')L_qw_right
end if

!------------------------------------------------------------------
! Creating the filename with info of the suprlattice
SL_info = trim(char_QBl)//"x("//trim(char_Lqw_l)//','//trim(char_Lqb_l)//')-'//trim(char_cqw)//'-'&
&//trim(char_QBr)//'x('//trim(char_Lqw_r)//','//trim(char_Lqb_r)//')' 

! Creating the filename of the folder of the results 
SL_folder = trim(char_QBl)//"x"//trim(char_Lqw_l)//"+"//trim(char_Lqb_l)//"-"//trim(char_cqw)//'-'//&
&trim(char_QBr)//"x"//trim(char_Lqw_r)//"+"//trim(char_Lqb_r)

! Making the folder where the results will be saved 
! call system('mkdir -p Structures/'//adjustl(trim(SL_folder))) ![LEMBRAR DE COMENTAR A LINHA QUANDO FOR ENVIAR PRO JOSE]


open(26, file=trim(sample_folder)//'wavefunction_SL.txt')
open(36, file=trim(sample_folder)//'Energy_SL.txt')
open(7,  file=trim(sample_folder)//'Transmission_SL.txt') 
open(8,  file=trim(sample_folder)//'Photocurrent_SL.txt')
open(25, file=trim(sample_folder)//'Potencial_SL.txt') 
open(28, file=trim(sample_folder)//'OscStr_SL.txt')
open(29, file=trim(sample_folder)//'OscStr_SL_norm.txt')

!--------------------------------------------------------------------
! quantum bragg mirror to left of central quantum well
L_qw_left    = L_qw_left*nm 	!Width of lateral quantum well
L_qb_left    = L_qb_left*nm 	!Width of quantum barrier
L_bm_left    = -num_QB_left*(L_qb_left+L_qw_left) !The thickness of left superlattice

! quantum bragg mirror to right of central quantum well
L_qb_right   = L_qb_right*nm 	!Width of quantum barrier
L_qw_right   = L_qw_right*nm 	!Width of lateral quantum well
L_bm_right   = num_QB_right*(L_qb_right+L_qw_right)

! central quantum well
L_cqw        = L_cqw*nm 	!Width of central quantum well

!This "if" is used to find the largest superlattice (right or left side). The largest superlattice will have a barrier thickness of 50 nm, 
!and the barrier thickness of the other superlattice will be calculated.
if(-L_bm_left.ge.L_bm_right)then
	L0_left      = -50*nm		!Thickness of the first barrier 
	L_left       = (L_BM_left + L0_left)
	L0_right     = -L_left - L_cqw - L_BM_right !Thickness of the last barrier
	L_right      = L_cqw + L_BM_right + L0_right + L_cqw
else
	L0_right     = +50*nm
	L_right		 =  L_cqw + L_BM_right + L0_right + L_cqw
	L0_left      =  -L_right - L_bm_left 
	L_left       = L_BM_left + L0_left
end if

n = int((L_right-L_left)/dx) + 2

allocate (v(n),  x(n), m(n), psi(n), prod(n), ki(n), psi0(n,nn), energy(nn))
!------------------------------------------------------------------

do j=1,n
	x(j) = ((j-1)*dx + L_left)
end do



call mass(m_qw, m_qb, num_qb_left, L_qw_left, L_qb_left, L_cqw, num_qb_right, L_qw_right, L_qb_right, n)
fs= 0*efs_InGaAs
call potencial(fs, num_qb_left, L_qw_left, L_qb_left, L_cqw, num_qb_right, L_qw_right, L_qb_right, x_cut, n)
  
!---------------------------------------------------------------------
!   Computing the wavefunction of the ground state using the Numerov method
!---------------------------------------------------------------------
call cpu_time(start)

E_initial = 000.0d0   ![meV] 
E_final   = 700.0d0 ![meV] 
dE        = 0.1    ![meV]
En        = (E_final - E_initial)/dE
Ener      = E_initial

call Eigenstates_Transfer_Matrix(num_qb_left, L_qw_left, L_qb_left, L_cqw, num_qb_right, & 
                          L_qw_right, L_qb_right, x_cut, dx, E_initial, E_final, dE, En, ne, Ener, psi0, energy, n, nn)

psi(:) = psi0(:,1) ! wavefucntion of the ground state
E0 = Energy(1)![J] energy of the ground state

!---------------------------------------------------------------
!Escrevendo as funcoes de onda do eletron
do i=1,n
	write(26,*) real(x(i)/nm), ((real(psi0(i,j)**2) + imag(psi0(i,j)**2))*0.4E-6 + 1E3*Energy(j)/e_charge, j=1,nn)
end do



!------------------------------------------------------------------------
!     Calculanting the oscillator strength using non-parabolicity 
!	  https://doi.org/10.1103/PhysRevB.50.8663
call oscillator_strength(num_qb_left, L_qw_left, L_qb_left, L_cqw, num_qb_right, L_qw_right, L_qb_right, L0_left, &   
					   & L_left, ne, nn, psi0, energy, dx, n)
!------------------------------------------------------------------------

!---------------------------------------------------------------------



!---------------------------------------------------------------------
!   Computing the photovoltaic photocurrent spectrum using the coherent carrier propagation in the continuum 
!   doi: https://doi.org/10.1103/PhysRevB.60.R13993
!---------------------------------------------------------------------

v(1) = vo
v(n) = vo
do j=1,n
	write(25,*) x(j)/nm, 1E3*v(j)/e_charge!, m(j)/me!
end do	

E_initial = 0.d0 ![meV] 
E_final   = 500.d0 ![meV] 
dE        = 0.1    ![meV]
En        = (E_final - E_initial)/dE

do  jn=1,En

	Ener= (E_initial + jn*dE) ![meV]
	Ener= Ener*e_charge*1E-3  ![J]
	
	! non-parabolicity 	  
    m_qw = m_InGaAs*(1.d0 + ener/Ew)
    m_qb = m_InAlAs*(1.d0  + (E0+ ener-vo)/Eb)	  
    call mass(m_qw, m_qb, num_qb_left, L_qw_left, L_qb_left, L_cqw, num_qb_right, L_qw_right, L_qb_right, n) 	
	
	
	T0 (1,1) = 1.d0 
	T0 (1,2) = 0.d0
	T0 (2,1) = 0.d0
	T0 (2,2) = 1.d0
	
	W0 (1,1) = 1.d0 
	W0 (1,2) = 0.d0
	W0 (2,1) = 0.d0
	W0 (2,2) = 1.d0
	
	F(1) = 0.d0
	F(2) = 0.d0
	
	do i=n-1,1, -1
		aux1 = (2*m(i)*(Ener+E0-v(i)))/hbar/hbar
		ki(i) = cdsqrt(aux1)

		aux1 = (2*m(i+1)*(Ener+E0-v(i+1)))/hbar/hbar
		ki(i+1) = cdsqrt(aux1)		

		!Matriz propagacao para i=1
		P_inicial(1,1) = zexp(-ic*ki(i+1)*(x(i))) 
		P_inicial(1,2) = 0.d0
		P_inicial(2,1) = 0.d0
		P_inicial(2,2) = zexp(+ic*ki(i+1)*(x(i)))
		
		P_final(1,1) = zexp(+ic*ki(i)*(x(i))) 
		P_final(1,2) = 0.d0
		P_final(2,1) = 0.d0
		P_final(2,2) = zexp(-ic*ki(i)*(x(i)))		
	
		!Matriz descontinuidade para i=1 
		beta = (ki(i)*m(i+1))/(ki(i+1)*m(i))
		Dn(1,1) = 0.5d0*(1 + beta)
		Dn(1,2) = 0.5d0*(1 - beta)
		Dn(2,1) = 0.5d0*(1 - beta)
		Dn(2,2) = 0.5d0*(1 + beta)

	
		aux = matmul(P_inicial, Dn)
		Ti = matmul(Aux, P_final)
		T = matmul(T0,Ti)
		T0 = T	


		!----------------------------------------------------
		!Calculo da interacao eletron-foton na interface i
		!----------------------------------------------------
		cte = (m(i)*Fd)/(2.d0*ic*hbar*hbar*ki(i))
		beta_plus  = -cte*(zexp(+ic*ki(i)*x(i))*x(i)*psi(i)-zexp(+ic*ki(i)*x(i+1))*x(i+1)*psi(i+1))*dx
		beta_minus = +cte*(zexp(-ic*ki(i)*x(i))*x(i)*psi(i)-zexp(-ic*ki(i)*x(i+1))*x(i+1)*psi(i+1))*dx
		
		F(1) = F(1) + (T(1,1)*beta_minus + T(1,2)*beta_plus)
		F(2) = F(2) + (T(2,1)*beta_minus + T(2,2)*beta_plus)  	
				
	end do
	
	!	------------------------------------------------------------
	!	Calculo da fotocorrente em funcao da energia do foton (E-E0)
	!   ------------------------------------------------------------

	ctecur_left  = (hbar*ki(1))/m(1)
	ctecur_right = (hbar*ki(n))/m(n)

	Je = ctecur_left*Fd*(-F(2)/T(2,2)*dconjg(-F(2)/T(2,2)))
	Jd = ctecur_right*Fd*(F(1)-F(2)*T(1,2)/T(2,2))*dconjg((F(1)-F(2)*T(1,2)/T(2,2)))

	write(8,*)1E3*Ener/e_charge,real(Jd-Je)!/2.134818419e-11
	write(7,*) 1E3*Ener/e_charge, real(1/cdabs(T(1,1))) 

enddo

call cpu_time(finish)
! write(*,*) finish-start
open(30, file=trim(sample_folder)//'FimPrograma.txt')
call cpu_time(total_finish)
write(30,*) total_finish-total_start
stop
end program Photocurrent_Transfer_Matrix



!-----------------------------------------------------------------
!             transfer matrix
!-----------------------------------------------------------------
subroutine transfer_matrix(Ener, dx, n, T)
	use var
	implicit none
	double complex :: aux1, ki(n), P_inicial(2,2), P_final(2,2), beta, Dn(2,2)
	double complex :: T0(2,2), T(2,2), Ti(2,2), aux(2,2)
	real*8         :: Ener, dx
	integer        :: i, n
	
	T0 (1,1) = 1.d0 
	T0 (1,2) = 0.d0
	T0 (2,1) = 0.d0
	T0 (2,2) = 1.d0
	
	
	do i = 1, n-1
		
		if(ener.gt.v(i))then
			ki(i) = sqrt((2*m(i)*(Ener-v(i)))/hbar/hbar)
		else
			ki(i) = ic*sqrt((2*m(i)*(v(i) - Ener))/hbar/hbar)
		end if 
		
		if(ener.gt.v(i+1))then
			ki(i+1) = sqrt((2*m(i+1)*(Ener-v(i+1)))/hbar/hbar)
		else
			ki(i+1) = ic*sqrt((2*m(i+1)*(v(i+1) - Ener))/hbar/hbar)
		end if 

		!Matriz propagacao para i=1
		!faco o incremento de tempo negativo pois estou fazendo a propagacao da onda da esquerda para a direita, ou seja, andando dx
		
		P_inicial(1,1) = exp(ic*ki(i)*(dx)/2) 
		P_inicial(1,2) = 0.d0
		P_inicial(2,1) = 0.d0
		P_inicial(2,2) = exp(-ic*ki(i)*(dx)/2)
		
		P_final(1,1) = exp(ic*ki(i+1)*(dx)/2) 
		P_final(1,2) = 0.d0
		P_final(2,1) = 0.d0
		P_final(2,2) = exp(-ic*ki(i+1)*(dx)/2)
	
	
		!Matriz descontinuidade para i=1 
		beta = (ki(i)*m(i+1))/(ki(i+1)*m(i))
		Dn(1,1) = 0.5d0*(1 + beta)
		Dn(1,2) = 0.5d0*(1 - beta)
		Dn(2,1) = 0.5d0*(1 - beta)
		Dn(2,2) = 0.5d0*(1 + beta)
		
		aux = matmul(P_final,Dn)
		Ti = matmul(aux,P_inicial)
		T = matmul(Ti,T0)
		T0 = T
						
	end do
	return
end subroutine transfer_matrix

!---------------------------------------------------------------
!         Computing the energies and wavefunction using Transfer matrix
!---------------------------------------------------------------  
  
subroutine Eigenstates_Transfer_Matrix(num_qb_left, L_qw_left, L_qb_left, L_cqw, num_qb_right, L_qw_right, &
	 									L_qb_right, x_cut, dx, Ei, Ef, dE, En, ne, Ener, psi0, energy, n, nn)

	use var
	Implicit none
	Integer :: i, j, x_cut, jn, k, ne, num_qb_left, num_qb_right, En, n, nn, interaction	
	real*8  :: E_initial, T22_new, T22_old 	
	Real*8  :: Ei, Ef, dE, Ener, Ener_Old, Ener_New, Energy(nn), mult2, m_qw, m_qb, soma 
	real*8  :: dx, L_qw_left, L_qb_left, L_cqw, L_qw_right, L_qb_right
	Double complex :: psi(n), psi_Old, psi_New, prod(n), trap, psi0(n,nn)
	Double complex :: T0(2,2), T(2,2), Ti(2,2), aux(2,2)
	
 	Ener_old = (Ei + 1*dE) ![meV]
 	Ener_old = Ener_old*e_charge*1E-3  ![J]
 	Ener = Ener_old
 	m_qw = m_InGaAs*(1.d0 + Ener/Ew)
 	m_qb = m_InAlAs*(1.d0 + (Ener-vo)/Eb) 
 	call mass(m_qw, m_qb, num_qb_left, L_qw_left, L_qb_left, L_cqw, num_qb_right, L_qw_right, L_qb_right, n) 
 	call transfer_matrix(Ener, dx, n, T)
 	T22_old = real(T(2,2))
	nn = 0      !Zerando o contador de autoestados
	
	do jn = 2, En
		!-------------------------------------------------------------------
		!      Calculando a ener_new para comecar o loomp	
		Ener_new = (Ei + jn*dE) ![meV]
		! write(201,*) 'Ener_new:', Ener_new
		Ener_new = Ener_new*e_charge*1E-3  ![J]
		Ener = Ener_new
		m_qw = m_InGaAs*(1.d0 + Ener/Ew)
		m_qb = m_InAlAs*(1.d0 + (Ener-vo)/Eb)	  
	 	call mass(m_qw, m_qb, num_qb_left, L_qw_left, L_qb_left, L_cqw, num_qb_right, L_qw_right, L_qb_right, n) 
	 	call transfer_matrix(Ener, dx, n, T)
		T22_new = real(T(2,2))
		!-----------------------------------------------------------------
		!           Metodo da bissecao para encontrar as autoenergias	
		!-----------------------------------------------------------------	
		mult2 = T22_old*T22_new
		
		if(mult2.lt.0)then
			do while (abs(Ener_new - Ener_old).gt.tol)
				Ener = (Ener_old + Ener_new)/2
				m_qw = m_InGaAs*(1.d0 + Ener/Ew)
				m_qb = m_InAlAs*(1.d0 + (Ener-vo)/Eb)	  
			 	call mass(m_qw, m_qb, num_qb_left, L_qw_left, L_qb_left, L_cqw, num_qb_right, L_qw_right, L_qb_right, n) 
			 	call transfer_matrix(Ener, dx, n, T)
				T22_new = real(T(2,2))
			
				mult2 = T22_old*T22_new
				if (mult2.lt.0.d0)then
					Ener_new = Ener
				else 
					Ener_old = Ener
				end if
			
				interaction = interaction + 1
			
				if (interaction.gt.max_interactions)then
					go to 15
				end if
				15 continue			
			end do
			nn = nn + 1
			energy(nn) = ener
			write(36,*) 1E3*Ener/e_charge
		end if

		!-------------------------------------------------------------------
		!      Calculando a ener_old para recomecar o loomp					
	
		Ener = Ei + dE*jn
		Ener = Ener*e_charge*1E-3  ![J]
		m_qw = m_InGaAs*(1.d0 + Ener/Ew)
		m_qb = m_InAlAs*(1.d0 + (Ener-vo)/Eb)
	 	call mass(m_qw, m_qb, num_qb_left, L_qw_left, L_qb_left, L_cqw, num_qb_right, L_qw_right, L_qb_right, n) 
	 	call transfer_matrix(Ener, dx, n, T)
		T22_old = real(T(2,2))
		Ener_old = Ener
		!--------------------------------------------------------------------	
	    !write(7,*) 1E3*Ener/e_charge, real(1/cdabs(T(2,2))) 
		
	end do 

	do i = 1, nn
		Ener = energy(i)
		m_qw = m_InGaAs*(1.d0 + Ener/Ew)
		m_qb = m_InAlAs*(1.d0 + (Ener-vo)/Eb)
		call mass(m_qw, m_qb, num_qb_left, L_qw_left, L_qb_left, L_cqw, num_qb_right, L_qw_right, L_qb_right, n) 
		!if(Ener.lt.vo)then
		call WaveFunc_transfer_matrix_split(n, x_cut, dx, i, Ener, psi)
		!else
			!call wavefunction_transfer_matrix(n, x_cut, dx, Ener, psi)
		!	call WaveFunc_transfer_matrix_split(n, x_cut, dx, i, Ener, psi)
		!end if

		psi0(:,i) = psi(:)

	end do
	
	return 
end subroutine Eigenstates_Transfer_Matrix

!---------------------------------------------------------------
!          Rotina para o calculo das funcoes de onda utilizando o metodo numerov 
!---------------------------------------------------------------  
  
subroutine wavefunction_transfer_matrix(n, x_cut, dx, Ener, psi)
														
	 use var
	 Implicit none
	 Integer :: i, x_cut, jn, k, ne, num_qb_left, num_qb_right, En, n, nn			
	 double complex :: aux1, ki(n), P_inicial(2,2), P_final(2,2), beta, Dn(2,2), prod(n)
 	 double complex :: T0(2,2), T(2,2), Ti(2,2), aux(2,2), psi(n), WaveFunc(2,2), trap
 	 real*8         :: Ener, soma, dx
	 
 	T0 (1,1) = 1.d0
 	T0 (1,2) = 0.d0
 	T0 (2,1) = 0.d0
 	T0 (2,2) = 1.d0
	
 	WaveFunc(1,1) = -1.0d0
 	WaveFunc(2,1) = +1.0d0
	
	
	do i = 1, n-1 
				
 		if(ener.gt.v(i))then
 			ki(i) = sqrt((2*m(i)*(Ener - v(i)))/hbar/hbar)
 		else
 			ki(i) = ic*sqrt((2*m(i)*(v(i) - Ener))/hbar/hbar)
 		end if 
		
 		if(ener.gt.v(i+1))then
 			ki(i+1) = sqrt((2*m(i+1)*(Ener-v(i+1)))/hbar/hbar)
 		else
 			ki(i+1) = ic*sqrt((2*m(i+1)*(v(i+1) - Ener))/hbar/hbar)
 		end if

 		!Matriz propagacao para i=1
 		!faco o incremento positivo pois estou fazendo a propagacao da onda da esquerda para a direita, ou seja, andando dx
		
 		P_inicial(1,1) = exp(ic*ki(i)*(dx)/2) 
 		P_inicial(1,2) = 0.d0
 		P_inicial(2,1) = 0.d0
 		P_inicial(2,2) = exp(-ic*ki(i)*(dx)/2)
		
 		P_final(1,1) = exp(ic*ki(i+1)*(dx)/2) 
 		P_final(1,2) = 0.d0
 		P_final(2,1) = 0.d0
 		P_final(2,2) = exp(-ic*ki(i+1)*(dx)/2)
			
	
 		!Matriz descontinuidade para i=1 
 		beta = (ki(i)*m(i+1))/(ki(i+1)*m(i))
 		Dn(1,1) = 0.5d0*(1 + beta)
 		Dn(1,2) = 0.5d0*(1 - beta)
 		Dn(2,1) = 0.5d0*(1 - beta)
 		Dn(2,2) = 0.5d0*(1 + beta)
		
 !       Fazendo a multiplicacao das matrizes para calcular a funcao de onda		
 		aux = matmul(P_final,Dn)
 		Ti = matmul(aux,P_inicial)
 		WaveFunc = matmul(Ti,WaveFunc)
 !       Funcao de onda sera a soma das duas amplitudes complexas	
 		psi(i) = WaveFunc(1,1) + WaveFunc(2,1)
			
 	end do
	
	
 !------------------------------------------------------------
 !   Normalizando as funcoes de onda calculadas 	
 	prod(:) = conjg(psi(:))*psi(:)
 	soma = trap(prod, 1, n, dx, n)	
 	!funcao de onda normalizada
 	psi(:) = psi(:)/(sqrt(soma))
	 
	
	return 
end subroutine wavefunction_transfer_matrix

!---------------------------------------------------------------
!          Rotina para o calculo das funcoes de onda utilizando o metodo numerov 
!---------------------------------------------------------------  
  
subroutine WaveFunc_transfer_matrix_split(n, x_cut, dx, k, Ener, psi)
														
	use var
	Implicit none
	Integer :: i, x_cut, jn, k, ne, num_qb_left, num_qb_right, En, n, nn			
	double complex :: aux1, ki(n), P_inicial(2,2), P_final(2,2), beta, Dn(2,2), prod(n)
	double complex :: T0(2,2), T(2,2), Ti(2,2), aux(2,2), psi(n), trap, ki_L(n), ki_R(n)
	double complex :: psi_R(n), psi_L(n), WaveFunc_R(2,2), WaveFunc_L(2,2)
	double complex :: P_inicial_R(2,2), P_inicial_L(2,2),P_final_R(2,2), P_final_L(2,2)
	real*8         :: Ener, soma, dx
	character(len=20) :: name

	 
 	T0 (1,1) = 1.d0
 	T0 (1,2) = 0.d0
 	T0 (2,1) = 0.d0
 	T0 (2,2) = 1.d0
	
 	WaveFunc_L(1,1) = -1.0d0
	WaveFunc_L(1,2) = 0.0d0
 	WaveFunc_L(2,1) = +1.0d0
	WaveFunc_L(2,2) = 0.0d0

 	WaveFunc_R(1,1) = -1.0d0
	WaveFunc_R(1,2) = 0.0d0
 	WaveFunc_R(2,1) = +1.0d0
	WaveFunc_R(2,2) = 0.0d0
	
 	! Calculando a funcao de onda da direta para esquerda com incremento -dx	
 	do i = 1, x_cut
				
 		if(ener.gt.v(i))then
 			ki_L(i) = sqrt((2*m(i)*(Ener - v(i)))/hbar/hbar)
 		else
 			ki_L(i) = ic*sqrt((2*m(i)*(v(i) - Ener))/hbar/hbar)
 		end if 
		
 		if(ener.gt.v(i+1))then
 			ki_L(i+1) = sqrt((2*m(i+1)*(Ener-v(i+1)))/hbar/hbar)
 		else
 			ki_L(i+1) = ic*sqrt((2*m(i+1)*(v(i+1) - Ener))/hbar/hbar)
 		end if

 		!Matriz propagacao para i=1
 		!faco o incremento positivo pois estou fazendo a propagacao da onda da esquerda para a direita, ou seja, andando dx
		
 		P_inicial_L(1,1) = exp(ic*ki_L(i)*(dx)/2) 
 		P_inicial_L(1,2) = 0.d0
 		P_inicial_L(2,1) = 0.d0
 		P_inicial_L(2,2) = exp(-ic*ki_L(i)*(dx)/2)

 		P_final_L(1,1) = exp(ic*ki_L(i+1)*(dx)/2) 
 		P_final_L(1,2) = 0.d0
 		P_final_L(2,1) = 0.d0
 		P_final_L(2,2) = exp(-ic*ki_L(i+1)*(dx)/2)
	
 		!Matriz descontinuidade para i=1 
 		beta = (ki_L(i)*m(i+1))/(ki_L(i+1)*m(i))
 		Dn(1,1) = 0.5d0*(1 + beta)
 		Dn(1,2) = 0.5d0*(1 - beta)
 		Dn(2,1) = 0.5d0*(1 - beta)
 		Dn(2,2) = 0.5d0*(1 + beta)
		
		
 		! Fazendo a multiplicacao das matrizes para calcular a funcao de onda		
		aux = matmul(P_final_L,Dn)
 		Ti = matmul(aux,P_inicial_L)
 		WaveFunc_L = matmul(Ti,WaveFunc_L)
 		!       Funcao de onda sera a soma das duas amplitudes complexas	
 		psi_L(i) = WaveFunc_L(1,1) + WaveFunc_L(2,1)
			
 	end do
	
 	T0 (1,1) = 1.d0
 	T0 (1,2) = 0.d0
 	T0 (2,1) = 0.d0
 	T0 (2,2) = 1.d0
	
 	! Calculando a funcao de onda da direta para esquerda com incremento -dx	
 	do i = n, x_cut, -1
				
 		if(ener.gt.v(i))then
 			ki_R(i) = sqrt((2*m(i)*(Ener - v(i)))/hbar/hbar)
 		else
 			ki_R(i) = ic*sqrt((2*m(i)*(v(i) - Ener))/hbar/hbar)
 		end if 
		
 		if(ener.gt.v(i-1))then
 			ki_R(i-1) = sqrt((2*m(i-1)*(Ener-v(i-1)))/hbar/hbar)
 		else
 			ki_R(i-1) = ic*sqrt((2*m(i-1)*(v(i-1) - Ener))/hbar/hbar)
 		end if
		
		 

 		!Matriz propagacao para i=1
 		!faco o incremento positivo pois estou fazendo a propagacao da onda da esquerda para a direita, ou seja, andando dx
		
 		P_inicial_R(1,1) = exp(ic*ki_R(i)*(-dx)/2) 
 		P_inicial_R(1,2) = 0.d0
 		P_inicial_R(2,1) = 0.d0
 		P_inicial_R(2,2) = exp(-ic*ki_R(i)*(-dx)/2)
		
 		P_final_R(1,1) = exp(ic*ki_R(i-1)*(-dx)/2) 
 		P_final_R(1,2) = 0.d0
 		P_final_R(2,1) = 0.d0
 		P_final_R(2,2) = exp(-ic*ki_R(i-1)*(-dx)/2)
			
	
 		!Matriz descontinuidade para i=1 
 		beta = (ki_R(i)*m(i-1))/(ki_R(i-1)*m(i))
 		Dn(1,1) = 0.5d0*(1 + beta)
 		Dn(1,2) = 0.5d0*(1 - beta)
 		Dn(2,1) = 0.5d0*(1 - beta)
 		Dn(2,2) = 0.5d0*(1 + beta)
		
 		! Fazendo a multiplicacao das matrizes para calcular a funcao de onda		
 		aux = matmul(P_final_R,Dn)
 		Ti = matmul(aux,P_inicial_R)
		
 		WaveFunc_R = matmul(Ti,WaveFunc_R)
		! Funcao de onda sera a soma das duas amplitudes complexas	
 		psi_R(i) = WaveFunc_R(1,1) + WaveFunc_R(2,1)
 	end do
	
	!-------------------------------------------------------------
	!	Juntando as funcoes de onda 

	!A funcao de onda a direita apresenta uma paridade inversa que a funcao de onda da esquerda
	!para resolver o problema estou multiplicando por -1 as funcoes de onda impares
	do i = n, x_cut, -1 
 		psi_R(i) = (-1)**k*psi_R(i)
 	end do

 	!normalizando a funcao de onda da esquerda com a amplitude da funcao de onda da direita
 	do i = 1, x_cut
 		psi_L(i) = psi_L(i)*psi_R(x_cut)/psi_L(x_cut)
 	end do

 	!juntando as funcoes de onda esquerda e direita em apenas uma funcao de onda
 	do i = 1, x_cut
 		psi(i) = psi_L(i)
 	end do

 	do i = x_cut, n 
 		psi(i) = psi_R(i)
 	end do

	
	!------------------------------------------------------------
	!   Normalizando as funcoes de onda calculadas 	
 	prod(:) = conjg(psi(:))*psi(:)
 	soma = trap(prod, 1, n, dx, n)	
 	psi(:) = psi(:)/(sqrt(soma))
	 
	return 
end subroutine WaveFunc_transfer_matrix_split	
		
	
!----------------------------------------------------------------
!		Normalizando as funcoes de onda
!----------------------------------------------------------------

subroutine norm_psi(psi, ni, nf, dx, n)
	use var
	implicit none
	double complex :: psi(n), prod(n), trap
	real*8         :: norm, dx 
	integer 	   :: ni, nf, j, n

	do j = ni, nf
		prod(j) = conjg(psi(j))*psi(j)
	end do

	norm = trap(prod, ni, nf, dx, n)

	do j = ni, nf
		psi(j) = psi(j)/sqrt(norm)
	end do
	return 
end subroutine norm_psi

	
!---------------------------------------------------------------------
!       Integracao - usando o metodo do trapezio 
!---------------------------------------------------------------------
Function trap(d, ni, nf, dx, n)
	use var
	Implicit none
	Integer        :: ni, nf, j, n
	double complex :: d(n), trap
	real*8		   :: dx			

	trap = 0

	do j=ni+2,(nf-1)
		trap = trap +d(j)*dx
	end do

	trap = trap+0.5*(d(ni)+d(nf))*dx
	return
end function trap
	
		
!----------------------------------------------------------------
!		Calculando a forca de oscilador
!----------------------------------------------------------------
subroutine oscillator_strength(num_qb_left, L_qw_left, L_qb_left, L_cqw, num_qb_right, L_qw_right, L_qb_right, L0_left, &   
					& L_left, ne, nn, psi0, energy, dx, n)
	use var
	implicit none
	integer 	   :: i, ne, n, num_QB_left, num_QB_right, jn, k, j, nn
	double complex :: psi0(n,ne), prod(n), trap
	real*8		   :: L_qw_left, L_qb_left, L_cqw, L_qw_right, L_qb_right, m_qw, m_qb, L0_left, L_left
	real*8         :: norm, energy(nn), f(nn),dE, soma, dx, meff(n,nn), mass_deri(n), psi0_deri(n), largest
	
	m_qw = m_InGaAs*(1.d0 + energy(1)/Ew)
	m_qb = m_InAlAs*(1.d0 + (energy(1)-vo)/Eb)
	call mass(m_qw, m_qb, num_qb_left, L_qw_left, L_qb_left, L_cqw, num_qb_right, L_qw_right, L_qb_right, n) 
	meff(:,1) = m(:)
			
	do i = 2, nn
		m_qw = m_InGaAs*(1.d0 + energy(i)/Ew)
		m_qb = m_InAlAs*(1.d0 + (energy(i)-vo)/Eb)
		call mass(m_qw, m_qb, num_qb_left, L_qw_left, L_qb_left, L_cqw, num_qb_right, L_qw_right, L_qb_right, n) 
		meff(:,i) = m(:)
		
		!-------------------------------------------------------------------------------------------------------
		! calculando a primeira derivada da funcao de onda do estado excitado e da massa efetiva 
		
		! Derivada à direita no ponto i = 1
		mass_deri(1) = (meff(2,1) - meff(1,1))/dx
		psi0_deri(1) = (psi0(2,i) - psi0(1,i))/(dx)
		
		! Derivada à esquerda no ponto i = n
		mass_deri(n) = (meff(n,1) - meff(n-1,1))/dx
		psi0_deri(n) = (psi0(n,i) - psi0(n-1,i))/(dx)
		
		! Derivada central no ponto i
		do k = 2, n-1
			mass_deri(k) = (meff(k+1,1) - meff(k-1,1))/(2*dx)
			psi0_deri(k) = (psi0(k+1,i) - psi0(k-1,i))/(2*dx)
		end do
		
		!-----------------------------------------------------------------------------------------------------
		! Calculo da forca do oscilador considerando os efeitos da nao parabolicidade 
		prod(:) = - conjg(psi0(:,1))*psi0(:,i)*mass_deri(:)/meff(:, 1)/meff(:, 1) + &
		& conjg(psi0(:,1))*psi0_deri(:)/meff(:, 1) + conjg(psi0(:,1))*psi0_deri(:)/meff(:, i) 
		soma = trap(prod, 1, n, dx, n)
		dE = (Energy(i) - Energy(1))
		
		f(i) = m_qw*hbar*hbar*abs(soma)**2/2.d0/dE
		write(28,*) 1E3*dE/e_charge, f(i)
	end do
	
	!largest = maxval(f)
	!do i = 2,nn
	!	dE = (Energy(i) - Energy(1))
	!	f(i) = f(i)/largest
	!	write(29,*) 1E3*dE/e_charge, f(i)
	!end do
		
		
	
	return 
end subroutine oscillator_strength		
	

!---------------------------------------------------------------
!                    mass
!---------------------------------------------------------------
subroutine mass(m_qw, m_qb, num_qb_left, L_qw_left, L_qb_left, L_cqw, num_qb_right, L_qw_right, L_qb_right, n)  
	use var
	Implicit none
	Integer :: i, j, k, num_qb_left, num_qb_right, n_qb_left, n_qb_right, n
	Real*8  :: m_qw, m_qb, L_qw_left, L_qb_left, L_cqw, L_qw_right, L_qb_right
	Real*8  :: L0, L0_left, L_bm_left
	
	L_bm_left = num_QB_left*(L_qb_left+L_qw_left)    ! Comprimento do espelho de Bragg a esquerda do poco quantico de defeito
	L0        =  L_bm_left							 ! L0 e a comprimento do ultimo poco quantico da super-rede	
	L0_left   = L0                                   ! Guardando esse valor para ser usado no looping da construcao da super-rede


	!----------------------------------------------------------------------
	!		Escrevendo a massa do eletron da primeira barreira quantica a esquerda da amostra
	!----------------------------------------------------------------------		
		
		do j = 1, n	
			if(x(j).le.-L0)then
				m(j) = m_qb
			else 
				go to 10
			end if
		end do		
	10      continue	
	!----------------------------------------------------------------------


	!----------------------------------------------------------------------
	!	    Escrevendo a massa do eletron da super-rede a esquerda do poco quantico central
	!		Aqui a repeticao vai ser poco/barreira
	!----------------------------------------------------------------------
		k = 1         ! inteiro que vai ser usada para determinar o fim da repeticao poco/barreira
		n_qb_left = 0 !numero de repeticao do conjunto poco/barreira indo de 0 ate 5
		
		do i = j, n
			L0 = L0_left - n_qb_left*(L_qb_left+L_qw_left)
			if(x(i).ge.-L0.and.x(i).le.-L0+L_qw_left)then
				m(i) = m_qw
			else if(x(i).ge.-L0+L_qw_left.and.x(i).le.-L0+L_qw_left+L_qb_left)then
				m(i) = m_qb
			else
			m(i) = m_qw!m_qb
			n_qb_left = n_qb_left + 1 ! adicione uma repeticao poco/barreira no potencial
			k = k+1
			end if	
			if(k.gt.num_qb_left) go to 20 ! se o numero de repeticao (k) for igual a numero fornecido no input (um_qb_left), finalize o looping
		end do
	20      continue

	!----------------------------------------------------------------------
			
			
	!----------------------------------------------------------------------
	!		Escrevendo a massa do eletron do poco quantico central
	!----------------------------------------------------------------------		

		do j = i, n
			if(x(j).ge.0.d0.and.x(j).le.L_cqw)then
				m(j) = m_qw
			else
			go to 30
		end if
	    end do		
	30		continue 
	!----------------------------------------------------------------------



	!----------------------------------------------------------------------
	!	    Escrevendo a massa do eletron da super-rede a direita do poco quantico central
	!		Aqui a repeticao vai ser barreira/poco 
	!----------------------------------------------------------------------		
		k = 1
		n_qb_right = 0
		do i=j,n
		L0 = L_cqw + n_qb_right*(L_qw_right+L_qb_right)
		if(x(i).ge.L0.and.x(i).le.L0+L_qb_right)then
			m(i) = m_qb
		else if(x(i).ge.L0+L_qb_right.and.x(i).le.L0+L_qb_right+L_qw_right)then
			m(i) = m_qw
		else
			m(i) = m_qb!m_qw
			n_qb_right = n_qb_right + 1
			k = k+1	
		end if
		if(k.gt.num_qb_right) go to 40	
		end do
	40		continue

	!----------------------------------------------------------------------



	!----------------------------------------------------------------------
	!		Escrevendo a massa do eletron da primeira barreira quantica a esquerda da amostra
	!----------------------------------------------------------------------	
		do j=i,n
			m(j) = m_qb
	    end do
	!----------------------------------------------------------------------		
	  return
end subroutine mass
	
!---------------------------------------------------------------
!                     potencial 
!---------------------------------------------------------------
subroutine potencial(fs, num_qb_left, L_qw_left, L_qb_left, L_cqw, num_qb_right, L_qw_right, L_qb_right, x_cut, n)
	use var
	Implicit none
	Integer :: j, i, x_cut, k, num_qb_left, num_qb_right, n_qb_left, n_qb_right, n	
	Real*8  :: fs, L_cqw, L_qb,L_lqw, L_qw_left, L_qb_left, L_qw_right, L_qb_right, L_left, L0
	Real*8  :: L0_left, L_bm_left


	L_bm_left = num_QB_left*(L_qb_left+L_qw_left)    ! Comprimento do espelho de Bragg a esquerda do poco quantico de defeito
	L0        =  L_bm_left							 ! L0 e a comprimento do ultimo poco quantico da super-rede	
	L0_left   = L0                                   ! Guardando esse valor para ser usado no looping da construcao da super-rede
	!----------------------------------------------------------------------
	!		Escrevendo o potencial da primeira barreira quantica a esquerda da amostra
	!----------------------------------------------------------------------		
		
	do j = 1, n	
		if(x(j).le.-L0)then
			v(j) = Vo
		else 
			go to 10
		end if
	end do		
	10      continue	
	!----------------------------------------------------------------------


	!----------------------------------------------------------------------
	!	    Escrevendo o potencial da super-rede a esquerda do poco quantico central
	!		Aqui a repeticao vai ser poco/barreira
	!----------------------------------------------------------------------
	k = 1         ! inteiro que vai ser usada para determinar o fim da repeticao poco/barreira
	n_qb_left = 0 !numero de repeticao do conjunto poco/barreira indo de 0 ate 5
	
	do i = j, n
		L0 = L0_left - n_qb_left*(L_qb_left+L_qw_left)
		if(x(i).ge.-L0.and.x(i).le.-L0+L_qw_left)then
			v(i) = 0
		else if(x(i).ge.-L0+L_qw_left.and.x(i).le.-L0+L_qw_left+L_qb_left)then
			v(i) = vo
		else
			v(i) = 0!vo
			n_qb_left = n_qb_left + 1 ! adicione uma repeticao poco/barreira no potencial
			k = k+1
		end if	
		if(k.gt.num_qb_left) go to 20 ! se o numero de repeticao (k) for igual a numero fornecido no input (um_qb_left), finalize o looping
	end do
		20      continue

		!----------------------------------------------------------------------

		x_cut = i ! inteiro x_cut para ser usado para calcular a funcao de onda pelo Numerov				
		!----------------------------------------------------------------------
		!		Escrevendo o potencial do poco quantico central
		!----------------------------------------------------------------------		

		do j = i, n
			if(x(j).ge.0.d0.and.x(j).le.L_cqw)then
				v(j) = 0
			else
			go to 30
		end if
	end do		
	30		continue 

	!----------------------------------------------------------------------

	!----------------------------------------------------------------------
	!	    Escrevendo o potencial da super-rede a direita do poco quantico central
	!		Aqui a repeticao vai ser barreira/poco 
	!----------------------------------------------------------------------		
		k = 1
		n_qb_right = 0
		do i=j,n
		L0 = L_cqw + n_qb_right*(L_qw_right+L_qb_right)
		if(x(i).ge.L0.and.x(i).le.L0+L_qb_right)then
			v(i) = vo
		else if(x(i).ge.L0+L_qb_right.and.x(i).le.L0+L_qb_right+L_qw_right)then
			v(i) = 0.d0
		else
			v(i) = vo!0.d0
			n_qb_right = n_qb_right + 1
			k = k+1	
		end if
		if(k.gt.num_qb_right) go to 40	
		end do
	40		continue

	!----------------------------------------------------------------------

	!----------------------------------------------------------------------
	!		Escrevendo o potencial da primeira barreira quantica a esquerda da amostra
	!----------------------------------------------------------------------	
		do j=i,n
			v(j) = vo
	    end do
	!----------------------------------------------------------------------			 
		v(1) = 10000*e_charge
		v(n) = 10000*e_charge

	      return
end subroutine potencial
	
