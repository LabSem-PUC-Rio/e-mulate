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

from enum import Enum

class PlotType(Enum):
    COMPLETE = "complete"
    COMPARISON = "comparison"
    HORIZONTAL = "complete_horizontal"
    NONE = None
    
class ParametersSimulation(Enum):
    OS = "OscillatorStrength"
    PC = "PhotoCurrent"
    OS_PC = "OscillatorStrength_PhotoCurrent"
