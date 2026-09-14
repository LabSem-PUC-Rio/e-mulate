import sys, os

# Obtén la ruta absoluta del archivo actual
current_file = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.append(project_path)
print(project_path)


if __name__ == '__main__':
    file_ed = 'source/fortran_simulation/fortran_files/20240825_Superlattice_Eigensates_Photocurrent_inputs.f90'
    file_ed_exe = 'source/fortran_simulation/fortran_files/20240825_Superlattice_Eigensates_Photocurrent_inputs.exe'
    
    compile = True
    run = False
    
    if compile:
        comand_to_compile = f'gfortran {file_ed} -o {file_ed_exe}'
        comand_to_compile = f'gfortran {file_ed} -o {file_ed_exe} -static-libgfortran'
        comand_to_compile = f'gfortran {file_ed} -o {file_ed_exe} -static'
        os.system(comand_to_compile)
        # aguarda até que exista o arquivo .exe
        while not os.path.exists(file_ed_exe):
            pass
                
        print('finish to compile')
        
    if run:
        os.makedirs("/home/joseruiz/codes/QBMD_GA/temp_files/fortran_test_ds/temp/000_05x02.0_07.0__02.5__01x02.0_07.0/", exist_ok=True)
        comand_to_run = f'{file_ed_exe} 6 2.0d0 7.0d0 2.6d0 4 2.0d0 7.0d0 "/home/joseruiz/codes/QBMD_GA/temp_files/fortran_test_ds/temp/000_05x02.0_07.0__02.5__01x02.0_07.0/"'
        print(comand_to_run)
        os.system(comand_to_run)



