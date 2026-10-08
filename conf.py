# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'e-mulate'
copyright = '2022, Penello, Pereira, Torelly, Ruiz'
author = 'Penello, Pereira, Torelly, Ruiz'

# The full version, including alpha/beta/rc tags
release = 'v0.4'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Paths configuration (Dynamic & Configurable) ----------------------------
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

output_fortran_folder = os.path.abspath(os.path.join(BASE_DIR, "windows_executable/temp_database")).replace("\\", "/") + "/"
file_ed_exe = os.path.abspath(os.path.join(BASE_DIR, "windows_executable/photocurrent_sim_windows.exe")).replace("\\", "/")

def save_conf_paths(new_output_folder=None, new_exe_path=None):
    """
    Atualiza as variáveis output_fortran_folder e file_ed_exe em memória e grava no conf.py.
    """
    global output_fortran_folder, file_ed_exe
    if new_output_folder:
        clean_out = os.path.normpath(new_output_folder).replace("\\", "/")
        if not clean_out.endswith("/"):
            clean_out += "/"
        output_fortran_folder = clean_out
    if new_exe_path:
        file_ed_exe = os.path.normpath(new_exe_path).replace("\\", "/")

    # Sincroniza sys.modules
    for mod_name in ("conf",):
        mod = sys.modules.get(mod_name)
        if mod:
            setattr(mod, "output_fortran_folder", output_fortran_folder)
            setattr(mod, "file_ed_exe", file_ed_exe)

    # Persiste no arquivo conf.py
    conf_file = os.path.abspath(__file__)
    try:
        with open(conf_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Encontra início do bloco de caminhos
        cut_idx = len(lines)
        for i, line in enumerate(lines):
            if "# -- Paths configuration" in line:
                cut_idx = i
                break
            elif "output_fortran_folder =" in line:
                cut_idx = max(0, i - 2)
                break

        header = "".join(lines[:cut_idx]).rstrip() + "\n\n"

        # Se os caminhos estiverem dentro do BASE_DIR, grava como caminhos relativos portáteis
        base_clean = BASE_DIR.replace("\\", "/")
        if output_fortran_folder.startswith(base_clean):
            rel_out = os.path.relpath(output_fortran_folder, BASE_DIR).replace("\\", "/")
            out_expr = f'os.path.abspath(os.path.join(BASE_DIR, "{rel_out}")).replace("\\\\", "/") + "/"'
        else:
            out_expr = f'"{output_fortran_folder}"'

        if file_ed_exe.startswith(base_clean):
            rel_exe = os.path.relpath(file_ed_exe, BASE_DIR).replace("\\", "/")
            exe_expr = f'os.path.abspath(os.path.join(BASE_DIR, "{rel_exe}")).replace("\\\\", "/")'
        else:
            exe_expr = f'r"{file_ed_exe}"'

        code_block = (
            "# -- Paths configuration (Dynamic & Configurable) ----------------------------\n"
            "import os\n"
            "import sys\n\n"
            "BASE_DIR = os.path.dirname(os.path.abspath(__file__))\n\n"
            f"output_fortran_folder = {out_expr}\n"
            f"file_ed_exe = {exe_expr}\n\n"
        )

        # Copia a função save_conf_paths
        with open(__file__, "r", encoding="utf-8") as f:
            cur_all = f.read()
        func_part = ""
        if "def save_conf_paths" in cur_all:
            func_part = cur_all[cur_all.index("def save_conf_paths"):]

        new_content = header + code_block + func_part
        with open(conf_file, "w", encoding="utf-8") as f:
            f.write(new_content)
        return True
    except Exception as e:
        print(f"Erro ao salvar conf.py: {e}")
        return False

def resolve_simulation_paths(folder_path):
    r"""
    Normaliza caminhos e identifica a estrutura de pastas independentemente do SO (Windows / Linux).
    Lida com caminhos usando / ou \, com ou sem subpasta 'temp', e com aspas.
    Retorna uma tupla (database_dir, simulations_dir, pkls_dir, cache_file):
      - database_dir: Diretório raiz da base de dados (onde ficam 'temp', 'pkls', 'optimizations', 'samples_cache.json')
      - simulations_dir: Diretório onde ficam as subpastas das simulações individuais
      - pkls_dir: Diretório onde ficam os arquivos .pkl
      - cache_file: Caminho completo para o arquivo samples_cache.json
    """
    if not folder_path:
        return "", "", "", ""

    # Remove aspas e normaliza delimitadores de acordo com o sistema operacional
    clean = os.path.normpath(str(folder_path).strip().strip('"').strip("'"))

    simulations_dir = None
    database_dir = None

    # 1. Verifica se a própria pasta clean já contém diretamente subpastas de simulação
    # (identificadas por contendo '__' no nome da subpasta ou o arquivo Photocurrent_SL.txt dentro dela)
    if os.path.isdir(clean):
        try:
            entries = os.listdir(clean)
            subdirs = [d for d in entries if os.path.isdir(os.path.join(clean, d))]
            has_sims = any('__' in d or os.path.exists(os.path.join(clean, d, 'Photocurrent_SL.txt')) for d in subdirs[:30])
            if has_sims:
                simulations_dir = clean
                if os.path.basename(clean).lower() == 'temp':
                    database_dir = os.path.dirname(clean)
                else:
                    database_dir = clean
        except Exception:
            pass

    # 2. Se não detectou diretamente simulações dentro de clean, verifica se clean/temp existe
    if simulations_dir is None:
        temp_sub = os.path.join(clean, "temp")
        if os.path.isdir(temp_sub):
            simulations_dir = temp_sub
            database_dir = clean
        elif os.path.basename(clean).lower() == "temp":
            simulations_dir = clean
            database_dir = os.path.dirname(clean)
        else:
            database_dir = clean
            simulations_dir = temp_sub

    pkls_dir = os.path.join(database_dir, "pkls")

    # 3. Localização do cache JSON
    cache_in_db = os.path.join(database_dir, "samples_cache.json")
    cache_in_sim = os.path.join(simulations_dir, "samples_cache.json")
    if os.path.exists(cache_in_db):
        cache_file = cache_in_db
    elif os.path.exists(cache_in_sim):
        cache_file = cache_in_sim
    else:
        cache_file = cache_in_db

    return database_dir, simulations_dir, pkls_dir, cache_file
