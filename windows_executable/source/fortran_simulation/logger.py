#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jose Ruiz
@modified_date: 2024-08-24
"""


import sys, os
import logging

# Obtén la ruta absoluta del archivo actual
current_file = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.append(project_path)

log_dir = os.path.join(project_path, 'temp_files')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'QBMD_GA.log')

# Configuración básica del logging
logging.basicConfig(
    level=logging.INFO,  # Nivel de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_file,  # Nombre del archivo de log
    filemode='w'  # Modo 'a' para añadir registros, 'w' para sobrescribir
)
logger = logging.getLogger('QBMD_GA')


