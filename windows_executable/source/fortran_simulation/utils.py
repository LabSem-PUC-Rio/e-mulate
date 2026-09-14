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


def set_structure_values(values):
    if len(values) == 3:
        w_l = 5
        qw_l = round(values[0]/10, 1)
        qb_l = round(values[1]/10, 1)

        defe = round(values[2]/10, 1)

        w_r = 1
        qw_r = round(values[0]/10, 1)
        qb_r = round(values[1]/10, 1)

        structure = [w_l,qw_l,qb_l, defe, w_r,qw_r,qb_r]

    elif len(values) == 5:
        w_l = round(values[0], 0)
        qw_l = round(values[1]/10, 1)
        qb_l = round(values[2]/10, 1)

        defe = round(values[3]/10, 1)

        w_r = round(values[4], 0)
        qw_r = round(values[1]/10, 1)
        qb_r = round(values[2]/10, 1)

        structure = [w_l,qw_l,qb_l, defe, w_r,qw_r,qb_r]

    elif len(values) == 7:
        w_l = round(values[0], 0)
        qw_l = round(values[1]/10, 1)
        qb_l = round(values[2]/10, 1)

        defe = round(values[3]/10, 1)

        w_r = round(values[4], 0)
        qw_r = round(values[5]/10, 1)
        qb_r = round(values[6]/10, 1)

        structure = [w_l,qw_l,qb_l, defe, w_r,qw_r,qb_r]

    # modifica os valores do pocos e numero de pocos se um valor eh 0
    if 0 in [w_l, qw_l, qb_l]:
        w_l, qw_l, qb_l = 0, 0, 0
        structure = [w_l,qw_l,qb_l, defe, w_r,qw_r,qb_r]
    if 0 in [w_r, qw_r, qb_r]:
        w_r, qw_r, qb_r = 0, 0, 0
        structure = [w_l,qw_l,qb_l, defe, w_r,qw_r,qb_r]

    if w_r > w_l:
        structure = [w_r,qw_r,qb_r, defe, w_l,qw_l,qb_l]
    
    return structure
