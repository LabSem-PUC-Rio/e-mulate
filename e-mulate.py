"""
e-mulate
This is the main script of the e-mulate software.
"""
# Qt5
from PyQt5 import uic
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize, QPoint, QRect
from PyQt5.QtGui import QPixmap, QIcon, QPainter, QColor, QBrush, QPen, QPolygon, QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplashScreen,
    QSplitter,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # enable highdpi scaling
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # use highdpi icons

# To deal with files, time, paths, imports...
import ast
import concurrent.futures
from datetime import datetime
import html
import itertools
import json
import math
import traceback
import pandas as pd
import configparser
from copy import deepcopy

# from multiprocessing import cpu_count
import numpy as np
import os

# from pathos.multiprocessing import ProcessingPool as Pool
import pickle
import sys
import time
import webbrowser

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Toolbar shown on the figures
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

# Changes to rcParams are system-wide, so at the end of the program, the defaults need
# to be reset
from matplotlib import rcParams

# Guarantees that every part of the figure is inside the canvas
rcParams.update({"figure.autolayout": True})

# File containing the main class
import simdata
import interface_config

# Adding the GUI files directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), "GUI"))
sys.path.append(os.path.join(os.path.dirname(__file__), "windows_executable"))

from windows_executable.source.utils.utils_main_fortran import run_fortran_simulation
from windows_executable.source.fortran_simulation.SimulationOptions import SimulationOptions
from windows_executable.source.fortran_simulation.FortranSimulator import FortranSimulator
from windows_executable.source.fortran_simulation.Plotter import Plotter
from windows_executable.source.fortran_simulation.utils import set_structure_values
from windows_executable.source.geneticalgorithm.deap_ga import (
    DeapGAOptimizer, DEFAULT_BOUNDS, CSV_COLUMNS, parse_seeds_text, clamp_individual
)
import conf
from conf import output_fortran_folder

NM = 1.0e-9

GA_PARAM_INFO = {
    "num_params": (
        "Número de Parâmetros a Otimizar",
        "Define o espaço de busca e quais dimensões da heteroestrutura serão variadas pelo algoritmo genético:\n\n"
        "• 3 Parâmetros: Otimiza a espessura dos poços quânticos (QW), das barreiras (QB) e do poço central/defeito (MQW). "
        "O número de poços é fixo (5 poços à esquerda e 1 à direita), e poços/barreiras de ambos os lados têm espessuras idênticas.\n\n"
        "• 5 Parâmetros: Otimiza o número de poços à esquerda (w_l) e à direita (w_r), além da espessura dos poços (QW), "
        "barreiras (QB) e poço central (MQW). Os poços e barreiras de ambos os lados compartilham a mesma espessura.\n\n"
        "• 7 Parâmetros: Otimização assimétrica completa: número de poços à esquerda (w_l), espessura do poço esquerdo, "
        "barreira esquerda, poço central (MQW), número de poços à direita (w_r), poço direito e barreira direita.\n\n"
        "★ RESTRIÇÃO FÍSICA OBRIGATÓRIA (DEFEITO CENTRAL):\n"
        "O poço central (MQW / defeito) deve ser maior ou igual aos poços laterais (LQW e RQW) por 0.5 nm ou mais (+5 unidades inteiras):\n"
        "• MQW >= LQW + 0.5 nm (MQW >= LQW + 5)\n"
        "• MQW >= RQW + 0.5 nm (MQW >= RQW + 5)\n"
        "Esta condição é assegurada em 100% dos indivíduos gerados, recombinados ou mutados pelo algoritmo genético."
    ),
    "pop_size": (
        "Tamanho da População (Indivíduos)",
        "Quantidade de indivíduos (estruturas candidatas) avaliados em cada geração.\n\n"
        "• Populações maiores (ex: 30 a 80) exploram melhor o espaço de parâmetros, reduzindo o risco de convergência prematura em mínimos locais, mas exigem mais tempo total de computação.\n"
        "• Populações menores (ex: 10 a 25) convergem mais rapidamente, ideal para testes preliminares.\n"
        "• Valor recomendado: 30 a 50."
    ),
    "generations": (
        "Número de Gerações (Iterações)",
        "Quantidade total de ciclos evolutivos que o algoritmo genético executará.\n\n"
        "• A cada geração, a população sofre seleção por torneio, cruzamento e mutação, evoluindo rumo a maiores aptidões (fitness).\n"
        "• Ao continuar uma otimização por CSV, este valor representa quantas novas gerações serão simuladas e adicionadas ao histórico.\n"
        "• Valor recomendado: 20 a 50 gerações."
    ),
    "elitism": (
        "Elitismo (Indivíduos de Elite)",
        "Quantidade dos melhores indivíduos da geração atual que são preservados intactos e transferidos diretamente para a próxima geração sem sofrer mutação ou recombinação.\n\n"
        "• O elitismo assegura que a melhor solução já descoberta nunca seja perdida ou degradada entre gerações.\n"
        "• Valor recomendado: 1 a 2 indivíduos."
    ),
    "crossover_prob": (
        "Probabilidade de Cruzamento (Crossover)",
        "Probabilidade (entre 0.0 e 1.0) de que dois indivíduos progenitores troquem características genéticas para gerar dois novos descendentes.\n\n"
        "• Valores altos (ex: 0.70 a 0.90) estimulam a combinação e recombinação de blocos úteis de diferentes estruturas.\n"
        "• Valor recomendado: 0.80 (80%)."
    ),
    "crossover_type": (
        "Tipo de Cruzamento (Crossover Type)",
        "Estratégia matemática utilizada para recombinar os genes dos pais:\n\n"
        "• Uniforme: Cada gene do descendente tem probabilidade independente (50%) de ser herdado do pai 1 ou do pai 2. Excelente capacidade exploratória.\n"
        "• Ponto Único (One-Point): Um ponto de corte na estrutura é sorteado; genes antes do corte vêm do primeiro progenitor, e genes após o corte vêm do segundo.\n"
        "• Dois Pontos (Two-Point): Dois pontos de corte são sorteados, intercalando segmentos entre os dois progenitores."
    ),
    "mutation_prob": (
        "Probabilidade de Mutação",
        "Probabilidade (entre 0.0 e 1.0) de que um gene de um indivíduo sofra alteração aleatória para outro valor dentro dos limites físicos permitidos.\n\n"
        "• A mutação introduz novidade genética e impede o confinamento precoce em mínimos locais.\n"
        "• Valores recomendados: 0.10 a 0.20 (10% a 20%). Valores muito elevados aproximam a evolução de uma busca puramente aleatória."
    ),
    "goal": (
        "Objetivos e Restrição de Energia",
        "Configuração dos critérios de aptidão (fitness) para avaliação das soluções:\n\n"
        "1. Otimizar Fotocorrente (PC): Maximiza o pico absoluto de corrente gerada pela estrutura.\n"
        "2. Otimizar Força de Oscilador (OS): Maximiza a força de oscilador óptica da transição calculada na simulação Fortran.\n"
        "   (Pode-se selecionar um ou ambos os objetivos simultaneamente para otimização mono ou multi-objetivo).\n\n"
        "3. Limitar à Faixa de Energia Proposta:\n"
        "• Se DESMARCADO: O algoritmo busca maximizar PC e/ou OS livremente para qualquer energia onde o pico ocorra no espectro.\n"
        "• Se MARCADO: O algoritmo direciona a busca para a janela de interesse [Energia Alvo ± Margem] (ex: 300 ± 20 meV = 280 a 320 meV).\n\n"
        "★ POLÍTICA DE PENALIZAÇÃO CONTÍNUA (SEM DESCARTE ABRUPTO):\n"
        "Soluções cujo pico fique ligeiramente fora da faixa (por exemplo, 320.01 meV quando a faixa é 280-320 meV) "
        "NÃO são descartadas sumariamente! Em vez de eliminar uma estrutura de excelente intensidade por uma fração infinitesimal de meV, "
        "é aplicada uma penalização contínua e proporcional ao quanto ela se afasta da margem:\n"
        "• Se o pico está dentro da margem: Pontuação máxima integral de PC/OS somada a um bônus por proximidade da energia central.\n"
        "• Se o pico ultrapassa levemente a borda (ex: 320.01 meV): A penalização é mínima (preserva ~99.9% do fitness).\n"
        "• Se o pico afasta-se consideravelmente: A aptidão decai suavemente de forma quadrática, preservando o gradiente genético "
        "para guiar as mutações e cruzamentos de volta à faixa desejada sem perder diversidade estrutural."
    ),
    "target_energy": (
        "Energia Alvo do Pico (meV)",
        "Energia central desejada (em meV) para a transição óptica / fotocorrente.\n\n"
        "• Exemplo: 300 meV corresponde à faixa de infravermelho médio (~4.1 μm).\n"
        "• O algoritmo pontua mais alto soluções que tenham seu pico posicionado próximo a esta energia."
    ),
    "target_margin": (
        "Margem de Tolerância (meV)",
        "Semi-largura da janela de energia desejada em torno da Energia Alvo (ex: se Alvo = 300 meV e Margem = 20 meV, a janela é de 280 a 320 meV).\n\n"
        "• Indivíduos dentro da janela recebem a pontuação máxima de PC/OS somada a um bônus por proximidade da energia central.\n"
        "• Indivíduos que ultrapassam a margem (por exemplo, 320.01 meV) NÃO são descartados; sofrem uma penalização contínua proporcional "
        "à distância da borda, permitindo que indivíduos excelentes guiem a convergência sem perda abrupta de soluções promissoras."
    ),
    "resume": (
        "Continuar Otimização Existente (CSV)",
        "Permite selecionar um arquivo CSV de uma otimização anterior (formato AAAAMMDD_HHMMSS_otim_X_parametros.csv):\n\n"
        "• O programa lê a última geração completa do arquivo e restaura os indivíduos como população inicial.\n"
        "• O gráfico da interface exibirá imediatamente toda a evolução histórica.\n"
        "• Ao clicar em 'Iniciar', as novas gerações serão avaliadas e adicionadas diretamente ao final do mesmo CSV."
    ),
    "parallel": (
        "Simulações em Paralelo",
        "Executa múltiplas simulações de fotocorrente em paralelo utilizando múltiplos núcleos do processador (CPU):\n\n"
        "• O limite máximo de núcleos é detectado automaticamente conforme o hardware do computador.\n"
        "• Reduz drasticamente o tempo total por geração em CPUs multi-core modernos.\n"
        "• Recomenda-se reservar 1 ou 2 núcleos para manter o sistema operacional ágil e responsivo."
    ),
    "seeds": (
        "População Inicial / Sementes",
        "Permite fornecer indivíduos ou estruturas conhecidas para compor a primeira geração (Geração 1) do algoritmo genético:\n\n"
        "• Digite ou cole indivíduos (um por linha, ex: [20, 70, 25]) ou clique no botão 'Adicionar Estrutura Atual' para capturar os parâmetros atuais da simulação Fortran.\n"
        "• Os indivíduos restantes necessários para completar o tamanho da população serão gerados aleatoriamente pelo DEAP.\n"
        "• O histórico no CSV registrará esses indivíduos com origem 'seed', diferenciando-os dos indivíduos aleatórios 'init'."
    )
}


class NumericTableWidgetItem(QTableWidgetItem):
    """
    QTableWidgetItem that sorts numerically based on float/int values
    instead of default lexicographical string sorting.
    """
    def __init__(self, text: str, sort_val=None):
        super().__init__(text)
        if sort_val is None:
            try:
                sort_val = float(text)
            except (ValueError, TypeError):
                sort_val = text
        self.sort_val = sort_val

    def __lt__(self, other):
        if hasattr(other, "sort_val"):
            try:
                return float(self.sort_val) < float(other.sort_val)
            except (ValueError, TypeError):
                return str(self.sort_val) < str(other.sort_val)
        return super().__lt__(other)


class ExcelFilterHeaderView(QHeaderView):
    """
    Custom horizontal header view with Excel-style dropdown filter buttons [ ▼ ]
    on each column section.
    - Left-clicking the arrow button or right-clicking anywhere on the section deploys the Excel filter menu.
    - Double-clicking anywhere on the section deploys the Excel filter menu.
    - Left-clicking the header text outside the button executes normal column sorting.
    - Active filters show a prominent green badge [ 🔻 ] with a funnel icon.
    """
    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setSectionsClickable(True)
        self.setHighlightSections(True)
        self.setMouseTracking(True)
        self.setMinimumHeight(44)
        self._filter_callback = None
        self._active_filters = {}
        self._hovered_btn = -1

    def setFilterCallback(self, cb):
        self._filter_callback = cb

    def setActiveFilters(self, filters_dict):
        self._active_filters = filters_dict or {}
        self.viewport().update()

    def _button_rect_for_section(self, logicalIndex):
        x = self.sectionViewportPosition(logicalIndex)
        w = self.sectionSize(logicalIndex)
        h = self.height()
        btn_w = 20
        btn_h = 20
        bx = x + w - btn_w - 4
        by = (h - btn_h) // 2
        return QRect(bx, by, btn_w, btn_h)

    def paintSection(self, painter, rect, logicalIndex):
        super().paintSection(painter, rect, logicalIndex)

        btn_rect = self._button_rect_for_section(logicalIndex)
        is_filtered = logicalIndex in self._active_filters and bool(self._active_filters[logicalIndex])

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        # Button background
        if is_filtered:
            painter.setBrush(QBrush(QColor("#16a34a")))  # Excel green
            painter.setPen(QPen(QColor("#15803d"), 1.5))
        elif self._hovered_btn == logicalIndex:
            painter.setBrush(QBrush(QColor("#cbd5e1")))
            painter.setPen(QPen(QColor("#94a3b8"), 1))
        else:
            painter.setBrush(QBrush(QColor("#e2e8f0")))
            painter.setPen(QPen(QColor("#cbd5e1"), 1))

        painter.drawRoundedRect(btn_rect, 4, 4)

        # Arrow / Funnel
        arrow_color = QColor("#ffffff") if is_filtered else QColor("#334155")
        painter.setBrush(QBrush(arrow_color))
        painter.setPen(Qt.NoPen)
        cx = btn_rect.center().x()
        cy = btn_rect.center().y()

        if is_filtered:
            # Funnel icon for active filter
            funnel = [
                QPoint(cx - 5, cy - 4),
                QPoint(cx + 5, cy - 4),
                QPoint(cx + 1, cy + 1),
                QPoint(cx + 1, cy + 5),
                QPoint(cx - 1, cy + 5),
                QPoint(cx - 1, cy + 1)
            ]
            painter.drawPolygon(QPolygon(funnel))
        else:
            # Down arrow
            triangle = [
                QPoint(cx - 4, cy - 2),
                QPoint(cx + 4, cy - 2),
                QPoint(cx, cy + 3)
            ]
            painter.drawPolygon(QPolygon(triangle))

        painter.restore()

    def mouseMoveEvent(self, event):
        pos = event.pos()
        col = self.logicalIndexAt(pos)
        old_hover = self._hovered_btn
        if col >= 0 and self._button_rect_for_section(col).contains(pos):
            self._hovered_btn = col
        else:
            self._hovered_btn = -1
        if old_hover != self._hovered_btn:
            self.viewport().update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._hovered_btn = -1
        self.viewport().update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        pos = event.pos()
        col = self.logicalIndexAt(pos)
        if col >= 0:
            btn_rect = self._button_rect_for_section(col)
            # If clicked on dropdown button or right-clicked on section -> trigger filter popup!
            if (event.button() == Qt.LeftButton and btn_rect.contains(pos)) or (event.button() == Qt.RightButton):
                if self._filter_callback:
                    vx = self.sectionViewportPosition(col)
                    global_pt = self.viewport().mapToGlobal(QPoint(vx, self.height()))
                    self._filter_callback(col, global_pt)
                return
        super().mousePressEvent(event)


class ExcelColumnFilterPopup(QDialog):
    """
    Excel-style floating dropdown menu anchored to a specific table column header.
    Features:
    1. Sort ascending (A-Z / Min-Max).
    2. Sort descending (Z-A / Max-Min).
    3. Clear filter of this column.
    4. Numeric condition filter (>, >=, <, <=, ==, faixa).
    5. Real-time search box.
    6. Distinct values checkboxes (cascading with other columns).
    7. Select All / Deselect All.
    8. Apply and Cancel buttons.
    """
    def __init__(self, parent_win, table, col_idx, col_name, active_filters, on_applied_callback):
        super().__init__(parent_win, Qt.Popup | Qt.FramelessWindowHint)
        self.parent_win = parent_win
        self.table = table
        self.col_idx = col_idx
        self.col_name = col_name
        self.active_filters = active_filters
        self.on_applied_callback = on_applied_callback
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setMinimumWidth(270)
        self.setMaximumWidth(330)
        self.setStyleSheet('''
            QDialog {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
            }
            QLabel { color: #1e293b; font-size: 11px; }
            QPushButton.menu-action {
                background-color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 12px;
                text-align: left;
                color: #1e293b;
            }
            QPushButton.menu-action:hover {
                background-color: #f1f5f9;
                color: #0f172a;
            }
            QPushButton.menu-action:disabled {
                color: #94a3b8;
            }
            QLineEdit, QComboBox, QDoubleSpinBox {
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 4px 6px;
                font-size: 11px;
                background-color: #ffffff;
                color: #1e293b;
            }
            QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {
                border: 1.5px solid #2563eb;
            }
            QListWidget {
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                background-color: #ffffff;
                color: #1e293b;
            }
            QListWidget::item {
                padding: 3px 5px;
            }
            QListWidget::item:hover {
                background-color: #f8fafc;
            }
        ''')
        self.InitUI()

    def InitUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Header Title
        clean_title = self.col_name.replace('\n', ' ')
        header_layout = QHBoxLayout()
        lbl_title = QLabel(f"<b>Filtro: {clean_title}</b>")
        lbl_title.setStyleSheet("font-size: 12px; color: #1e293b;")
        btn_close = QPushButton("✕")
        btn_close.setFixedSize(22, 22)
        btn_close.setStyleSheet(
            "QPushButton { border: none; font-weight: bold; color: #64748b; font-size: 12px; border-radius: 11px; } "
            "QPushButton:hover { background-color: #fee2e2; color: #dc2626; }"
        )
        btn_close.clicked.connect(self.reject)
        header_layout.addWidget(lbl_title, 1)
        header_layout.addWidget(btn_close)
        layout.addLayout(header_layout)

        # Divider
        div0 = QFrame()
        div0.setFrameShape(QFrame.HLine)
        div0.setStyleSheet("color: #e2e8f0;")
        layout.addWidget(div0)

        # Sort Section
        self.btn_sort_asc = QPushButton("AZ  Ordenar de menor a maior")
        self.btn_sort_asc.setProperty("class", "menu-action")
        self.btn_sort_asc.clicked.connect(self.OnSortAsc)
        layout.addWidget(self.btn_sort_asc)

        self.btn_sort_desc = QPushButton("ZA  Ordenar de mayor a menor")
        self.btn_sort_desc.setProperty("class", "menu-action")
        self.btn_sort_desc.clicked.connect(self.OnSortDesc)
        layout.addWidget(self.btn_sort_desc)

        # Clear Filter for this column
        self.btn_clear_col = QPushButton(f"🗑  Limpiar filtro de '{clean_title}'")
        self.btn_clear_col.setProperty("class", "menu-action")
        has_filter = self.col_idx in self.active_filters
        self.btn_clear_col.setEnabled(has_filter)
        if has_filter:
            self.btn_clear_col.setStyleSheet("color: #dc2626; font-weight: bold; text-align: left; padding: 6px 10px;")
        self.btn_clear_col.clicked.connect(self.OnClearCol)
        layout.addWidget(self.btn_clear_col)

        # Divider
        div1 = QFrame()
        div1.setFrameShape(QFrame.HLine)
        div1.setStyleSheet("color: #e2e8f0;")
        layout.addWidget(div1)

        # Numeric Filter Section
        lbl_num = QLabel("<b>Filtro numérico:</b>")
        layout.addWidget(lbl_num)

        num_layout = QHBoxLayout()
        self.cbox_op = QComboBox()
        self.cbox_op.addItems([
            "(Sin filtro numérico)",
            "Mayor que (>)",
            "Mayor o igual a (≥)",
            "Menor que (<)",
            "Menor o igual a (≤)",
            "Igual a (=)",
            "Entre (faixa)"
        ])
        self.cbox_op.currentIndexChanged.connect(self.OnNumOpChanged)
        num_layout.addWidget(self.cbox_op, 1)
        layout.addLayout(num_layout)

        self.val_widget = QWidget()
        val_layout = QHBoxLayout(self.val_widget)
        val_layout.setContentsMargins(0, 0, 0, 0)
        val_layout.setSpacing(4)

        self.lbl_spb1 = QLabel("Valor:")
        self.spb_val = QDoubleSpinBox()
        self.spb_val.setRange(-1e9, 1e9)
        self.spb_val.setDecimals(4)

        self.lbl_spb2 = QLabel("e:")
        self.spb_val2 = QDoubleSpinBox()
        self.spb_val2.setRange(-1e9, 1e9)
        self.spb_val2.setDecimals(4)

        val_layout.addWidget(self.lbl_spb1)
        val_layout.addWidget(self.spb_val, 1)
        val_layout.addWidget(self.lbl_spb2)
        val_layout.addWidget(self.spb_val2, 1)
        layout.addWidget(self.val_widget)
        self.val_widget.setVisible(False)

        # Divider
        div2 = QFrame()
        div2.setFrameShape(QFrame.HLine)
        div2.setStyleSheet("color: #e2e8f0;")
        layout.addWidget(div2)

        # Search Box
        self.search_txt = QLineEdit()
        self.search_txt.setPlaceholderText("🔍 Buscar valor...")
        self.search_txt.textChanged.connect(self.OnSearchTextChanged)
        layout.addWidget(self.search_txt)

        # Select All Checkbox
        self.select_all_cb = QCheckBox("Seleccionar todo")
        self.select_all_cb.setChecked(True)
        self.select_all_cb.stateChanged.connect(self.OnSelectAllToggled)
        layout.addWidget(self.select_all_cb)

        # List Widget for Distinct Values
        self.list_widget = QListWidget()
        self.list_widget.setMaximumHeight(150)
        layout.addWidget(self.list_widget)

        # Populate
        self.PopulateDistinctValues()

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setStyleSheet("background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 4px; padding: 5px 12px; font-weight: 600; text-align: center;")
        btn_cancel.clicked.connect(self.reject)

        btn_apply = QPushButton("Aplicar")
        btn_apply.setStyleSheet("background-color: #16a34a; color: white; border: none; border-radius: 4px; padding: 5px 14px; font-weight: bold; text-align: center;")
        btn_apply.clicked.connect(self.OnApply)

        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_apply)
        layout.addLayout(btn_layout)

    def OnSortAsc(self):
        self.table.sortItems(self.col_idx, Qt.AscendingOrder)
        self.accept()

    def OnSortDesc(self):
        self.table.sortItems(self.col_idx, Qt.DescendingOrder)
        self.accept()

    def OnClearCol(self):
        self.active_filters.pop(self.col_idx, None)
        if self.on_applied_callback:
            self.on_applied_callback()
        self.accept()

    def OnNumOpChanged(self, idx):
        if idx == 0:
            self.val_widget.setVisible(False)
        elif idx == 6:  # 'Entre'
            self.val_widget.setVisible(True)
            self.lbl_spb1.setText("Min:")
            self.lbl_spb2.setVisible(True)
            self.spb_val2.setVisible(True)
        else:
            self.val_widget.setVisible(True)
            self.lbl_spb1.setText("Valor:")
            self.lbl_spb2.setVisible(False)
            self.spb_val2.setVisible(False)
        self.adjustSize()

    def OnSearchTextChanged(self, txt):
        query = txt.strip().lower()
        for i in range(self.list_widget.count()):
            it = self.list_widget.item(i)
            it.setHidden(query not in it.text().lower())

    def OnSelectAllToggled(self, state):
        chk = (state == Qt.Checked)
        self.list_widget.blockSignals(True)
        for i in range(self.list_widget.count()):
            it = self.list_widget.item(i)
            if not it.isHidden():
                it.setCheckState(Qt.Checked if chk else Qt.Unchecked)
        self.list_widget.blockSignals(False)

    def PopulateDistinctValues(self):
        col_idx = self.col_idx
        total_rows = self.table.rowCount()
        distinct_vals = {}
        num_vals = []

        for r in range(total_rows):
            match_others = True
            for c, cond in self.active_filters.items():
                if c == col_idx:
                    continue
                item = self.table.item(r, c)
                if not item:
                    match_others = False
                    break
                txt = item.text().strip()
                raw = getattr(item, "sort_val", None)
                num_op = cond.get("num_op", "none")
                if num_op != "none" and raw is not None:
                    try:
                        v = float(raw)
                        target = cond.get("num_val", 0.0)
                        if num_op == ">" and not (v > target): match_others = False; break
                        elif num_op == ">=" and not (v >= target): match_others = False; break
                        elif num_op == "<" and not (v < target): match_others = False; break
                        elif num_op == "<=" and not (v <= target): match_others = False; break
                        elif num_op == "==" and not (abs(v - target) < 1e-4): match_others = False; break
                        elif num_op == "range" and not (cond.get("min", -1e9) <= v <= cond.get("max", 1e9)): match_others = False; break
                    except (ValueError, TypeError):
                        pass
                if cond.get("checked") is not None and txt not in cond["checked"]:
                    match_others = False
                    break

            if match_others:
                item_c = self.table.item(r, col_idx)
                if item_c:
                    txt = item_c.text().strip()
                    raw = getattr(item_c, "sort_val", txt)
                    distinct_vals[txt] = raw
                    try:
                        num_vals.append(float(raw))
                    except (ValueError, TypeError):
                        pass

        def sort_key(pair):
            v = pair[1]
            try:
                return (0, float(v))
            except (ValueError, TypeError):
                return (1, str(v))

        sorted_items = sorted(distinct_vals.items(), key=sort_key)
        curr_filter = self.active_filters.get(col_idx, {})
        checked_set = curr_filter.get("checked")

        self.list_widget.clear()
        for txt, raw in sorted_items:
            it = QListWidgetItem(txt)
            it.setData(Qt.UserRole, txt)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            if checked_set is None or txt in checked_set:
                it.setCheckState(Qt.Checked)
            else:
                it.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(it)

        # Initialize existing numeric condition if any
        num_op = curr_filter.get("num_op", "none")
        op_map = {"none": 0, ">": 1, ">=": 2, "<": 3, "<=": 4, "==": 5, "range": 6}
        self.cbox_op.setCurrentIndex(op_map.get(num_op, 0))
        if num_op != "none":
            self.spb_val.setValue(curr_filter.get("num_val", 0.0))
            if num_op == "range":
                self.spb_val.setValue(curr_filter.get("min", min(num_vals) if num_vals else 0.0))
                self.spb_val2.setValue(curr_filter.get("max", max(num_vals) if num_vals else 100.0))
        else:
            if num_vals:
                self.spb_val.setValue(round(float(sum(num_vals) / len(num_vals)), 2))
                self.spb_val2.setValue(max(num_vals))

    def OnApply(self):
        col_idx = self.col_idx
        total_items = self.list_widget.count()
        checked_vals = set()
        for i in range(total_items):
            it = self.list_widget.item(i)
            if it.checkState() == Qt.Checked:
                checked_vals.add(it.data(Qt.UserRole))

        op_idx = self.cbox_op.currentIndex()
        op_codes = ["none", ">", ">=", "<", "<=", "==", "range"]
        num_op = op_codes[op_idx]

        all_checked = (len(checked_vals) == total_items)
        if all_checked and num_op == "none":
            self.active_filters.pop(col_idx, None)
        else:
            self.active_filters[col_idx] = {
                "checked": checked_vals if not all_checked else None,
                "num_op": num_op,
                "num_val": self.spb_val.value(),
                "min": self.spb_val.value() if num_op == "range" else None,
                "max": self.spb_val2.value() if num_op == "range" else None,
            }

        if self.on_applied_callback:
            self.on_applied_callback()
        self.accept()


class ExcelFilterDialog(QDialog):
    """Backwards compatibility wrapper pointing to per-column Excel filter."""
    def __init__(self, parent, table, col_names, active_filters, current_col=0):
        super().__init__(parent)
        self.parent_win = parent
        self.table = table
        self.col_names = col_names
        self.active_filters = active_filters
        self.current_col = current_col

    def exec_(self):
        if hasattr(self.parent_win, "OpenExcelFilterDialogForCol"):
            self.parent_win.OpenExcelFilterDialogForCol(self.current_col)
            return QDialog.Accepted
        return QDialog.Rejected


class FortranSimWorker(QThread):
    sig_started = pyqtSignal(str)                        # struct desc
    sig_finished = pyqtSignal(bool, object, str, float)  # success, sample_data, error_msg, elapsed_seconds

    def __init__(self, struct, sim_options, output_folder):
        super().__init__()
        self.struct = struct
        self.sim_options = sim_options
        self.output_folder = output_folder

    def run(self):
        w_l, qw_l, qb_l, mqw, w_r, qw_r, qb_r = set_structure_values(self.struct)
        desc = f"{int(w_l)}x{qw_l:.1f}_{qb_l:.1f} __{mqw:.1f}__ {int(w_r)}x{qw_r:.1f}_{qb_r:.1f}"
        self.sig_started.emit(desc)
        t0 = time.time()
        try:
            simulator = FortranSimulator(self.struct, sim_options=self.sim_options, output_folder=self.output_folder)
            sample_data = simulator.simulate()
            elapsed = time.time() - t0
            self.sig_finished.emit(True, sample_data, "", elapsed)
        except Exception as e:
            traceback.print_exc()
            elapsed = time.time() - t0
            self.sig_finished.emit(False, None, str(e), elapsed)


class AutomateWorker(QThread):
    sig_sample_started = pyqtSignal(int, int, str)        # sample_idx, total, desc
    sig_progress = pyqtSignal(int, int, dict, str)        # completed, total, res, msg
    sig_finished = pyqtSignal(int, float)                 # completed, total_elapsed
    sig_status = pyqtSignal(str)                          # general status message
    sig_error = pyqtSignal(str)                           # error message

    def __init__(self, param_grid, parallel=True, n_cores=4, output_folder="temp_files/fortran_test_ds/dev/"):
        super().__init__()
        self.param_grid = param_grid
        self.parallel = parallel
        self.n_cores = max(1, n_cores)
        self.output_folder = output_folder
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True
        self.requestInterruption()

    def run(self):
        combos = list(itertools.product(*self.param_grid))
        total = len(combos)
        if total == 0:
            self.sig_finished.emit(0, 0.0)
            return

        completed = 0
        from conf import output_fortran_folder
        folder = self.output_folder or output_fortran_folder
        t_start = time.time()

        self.sig_status.emit(f"🚀 Iniciando lote de {total} simulações ({self.n_cores if self.parallel else 1} núcleo(s))...")

        if self.parallel and self.n_cores > 1:
            n_threads = min(self.n_cores, total)
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
                future_to_combo = {
                    executor.submit(self._eval_combo, c, folder, i + 1, total): (i + 1, c)
                    for i, c in enumerate(combos)
                }
                for fut in concurrent.futures.as_completed(future_to_combo):
                    if self._stop_requested:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    idx, c = future_to_combo[fut]
                    try:
                        res = fut.result()
                        if res is not None:
                            completed += 1
                            msg = f"[{completed}/{total}] {res['desc']} concluída em {res['time']:.1f}s"
                            self.sig_progress.emit(completed, total, res, msg)
                    except Exception as e:
                        traceback.print_exc()
                        self.sig_error.emit(f"Erro na simulação {idx}/{total} ({c}): {str(e)}")
        else:
            for i, c in enumerate(combos):
                if self._stop_requested:
                    break
                try:
                    res = self._eval_combo(c, folder, i + 1, total)
                    if res is not None:
                        completed += 1
                        msg = f"[{completed}/{total}] {res['desc']} concluída em {res['time']:.1f}s"
                        self.sig_progress.emit(completed, total, res, msg)
                except Exception as e:
                    traceback.print_exc()
                    self.sig_error.emit(f"Erro na simulação {i + 1}/{total} ({c}): {str(e)}")

        total_elapsed = time.time() - t_start
        self.sig_finished.emit(completed, total_elapsed)

    def _eval_combo(self, combo, folder, idx, total):
        if self._stop_requested:
            return None
        s0, s1, s2, s3, s4, s5, s6 = combo
        desc = f"{int(s0)}x{s1:.1f}_{s2:.1f}__{s3:.1f}__{int(s4)}x{s5:.1f}_{s6:.1f}"
        self.sig_sample_started.emit(idx, total, desc)

        t0 = time.time()
        struct = [int(s0), round(s1 * 10, 1), round(s2 * 10, 1), round(s3 * 10, 1), int(s4), round(s5 * 10, 1), round(s6 * 10, 1)]
        sim_options = SimulationOptions(force_parser=False, force_simulation=False)
        simulator = FortranSimulator(struct, sim_options=sim_options, output_folder=folder)
        sample_data = simulator.simulate()
        elapsed = time.time() - t0

        return {
            "s0": int(s0),
            "s1": float(s1),
            "s2": float(s2),
            "s3": float(s3),
            "s4": int(s4),
            "s5": float(s5),
            "s6": float(s6),
            "pc_max": float(sample_data.max_abs_photocurrent),
            "pc_e": float(sample_data.max_e_abs_photocurrent),
            "os_max": float(sample_data.max_e_oscstr),
            "os_e": float(sample_data.max_e_transition),
            "time": elapsed,
            "desc": desc
        }


class GAWorker(QThread):
    sig_generation_finished = pyqtSignal(int, int, list, float, float, str, dict)
    sig_individual_evaluated = pyqtSignal(int, int, list, float)
    sig_status = pyqtSignal(str)
    sig_finished = pyqtSignal(bool, str, list, float, str, dict)

    def __init__(self, optimizer_kwargs):
        super().__init__()
        self.optimizer_kwargs = optimizer_kwargs
        self._is_stopped = False
        self.optimizer = None

    def is_cancelled(self) -> bool:
        return self._is_stopped or self.isInterruptionRequested()

    def stop(self):
        self._is_stopped = True
        self.requestInterruption()

    def run(self):
        try:
            self.optimizer_kwargs["is_cancelled"] = self.is_cancelled
            self.optimizer_kwargs["on_individual_done"] = (
                lambda i, tot, ind, fit: self.sig_individual_evaluated.emit(i, tot, ind, fit)
            )
            self.optimizer_kwargs["on_generation_done"] = (
                lambda gen, tot_gens, best_ind, best_f, mean_f, csv_p, best_res=None:
                self.sig_generation_finished.emit(gen, tot_gens, best_ind, best_f, mean_f, csv_p, best_res or {})
            )
            self.optimizer_kwargs["on_log"] = lambda msg: self.sig_status.emit(msg)

            self.optimizer = DeapGAOptimizer(**self.optimizer_kwargs)
            best_ind, best_fit, csv_path = self.optimizer.run()
            best_res = getattr(self.optimizer, "best_result", {})

            if self.is_cancelled():
                self.sig_finished.emit(False, "Otimização cancelada pelo usuário.", best_ind, best_fit, csv_path, best_res)
            else:
                self.sig_finished.emit(True, "Otimização concluída com sucesso!", best_ind, best_fit, csv_path, best_res)
        except Exception as e:
            traceback.print_exc()
            self.sig_finished.emit(False, f"Erro durante otimização: {str(e)}", [], 0.0, "", {})


MODERN_APP_STYLESHEET = """
/* Modern Clean Theme for e-mulate */
* {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', 'Helvetica Neue', sans-serif;
    font-size: 12px;
}

QMainWindow, QDialog {
    background-color: #f8fafc;
}

QWidget {
    color: #1e293b;
}

/* Tab Widget & Bar */
QTabWidget::pane {
    border: 1px solid #cbd5e1;
    background-color: #ffffff;
    border-radius: 8px;
    margin-top: -1px;
}

QTabBar::tab {
    background-color: #f1f5f9;
    color: #334155;
    padding: 10px 24px 12px 24px;
    font-weight: 700;
    font-size: 13px;
    border: 1px solid #cbd5e1;
    border-top: 3px solid transparent;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 5px;
    min-height: 32px;
    min-width: 95px;
}

QTabBar::tab:selected {
    background-color: #ffffff;
    color: #1d4ed8;
    border-top: 3.5px solid #2563eb;
    border-bottom: 1px solid #ffffff;
    font-weight: 800;
}

QTabBar::tab:hover:!selected {
    background-color: #e2e8f0;
    color: #0f172a;
}

/* Group Boxes (Cards) */
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    margin-top: 14px;
    padding: 14px 10px 10px 10px;
    font-weight: 600;
    color: #1e293b;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    background-color: #ffffff;
    color: #1e40af;
    font-weight: 700;
}

/* Push Buttons */
QPushButton {
    background-color: #f8fafc;
    color: #1e293b;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 5px 14px;
    font-weight: 500;
    min-height: 22px;
}

QPushButton:hover {
    background-color: #e2e8f0;
    border-color: #94a3b8;
}

QPushButton:pressed {
    background-color: #cbd5e1;
}

QPushButton:disabled {
    background-color: #f1f5f9;
    color: #94a3b8;
    border-color: #e2e8f0;
}

/* Primary Accent Action Buttons */
QPushButton#ga_run_btn, QPushButton#btn_run_fortran, QPushButton#ga_apply_to_fortran_btn, QPushButton#ga_plot_csv_btn, QPushButton#auto_run_btn, QPushButton#auto_apply_to_fortran_btn {
    background-color: #2563eb;
    color: #ffffff;
    border: 1px solid #1d4ed8;
    font-weight: 600;
}

QPushButton#ga_run_btn:hover, QPushButton#btn_run_fortran:hover, QPushButton#ga_apply_to_fortran_btn:hover, QPushButton#ga_plot_csv_btn:hover, QPushButton#auto_run_btn:hover, QPushButton#auto_apply_to_fortran_btn:hover {
    background-color: #1d4ed8;
    border-color: #1e40af;
}

QPushButton#ga_run_btn:pressed, QPushButton#btn_run_fortran:pressed, QPushButton#ga_apply_to_fortran_btn:pressed, QPushButton#ga_plot_csv_btn:pressed, QPushButton#auto_run_btn:pressed, QPushButton#auto_apply_to_fortran_btn:pressed {
    background-color: #1e40af;
}

QPushButton#ga_stop_btn:enabled, QPushButton#auto_stop_btn:enabled {
    background-color: #dc2626;
    color: #ffffff;
    border: 1px solid #b91c1c;
    font-weight: 600;
}

QPushButton#ga_stop_btn:enabled:hover, QPushButton#auto_stop_btn:enabled:hover {
    background-color: #b91c1c;
}

QPushButton#ga_copy_text_btn {
    background-color: #f0fdf4;
    color: #166534;
    border: 1px solid #86efac;
    font-weight: 600;
}

QPushButton#ga_copy_text_btn:hover {
    background-color: #dcfce7;
    border-color: #4ade80;
}

/* Input Fields */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 4px 8px;
    background-color: #ffffff;
    color: #0f172a;
    selection-background-color: #bfdbfe;
    selection-color: #1e3a8a;
    min-height: 20px;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1.5px solid #2563eb;
    background-color: #ffffff;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid #cbd5e1;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}

/* Checkboxes */
QCheckBox {
    spacing: 8px;
    color: #1e293b;
    font-weight: 500;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1.5px solid #94a3b8;
    background-color: #ffffff;
}

QCheckBox::indicator:hover {
    border-color: #2563eb;
}

QCheckBox::indicator:checked {
    background-color: #2563eb;
    border-color: #2563eb;
}

/* Table Widget */
QTableWidget {
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    background-color: #ffffff;
    gridline-color: #f1f5f9;
    alternate-background-color: #f8fafc;
    selection-background-color: #dbeafe;
    selection-color: #1e3a8a;
}

QHeaderView::section {
    background-color: #f1f5f9;
    color: #334155;
    font-weight: 600;
    padding: 6px 10px;
    border: none;
    border-bottom: 2px solid #cbd5e1;
    border-right: 1px solid #e2e8f0;
}

/* Progress Bar */
QProgressBar {
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    text-align: center;
    background-color: #f1f5f9;
    font-weight: 600;
    color: #1e293b;
    min-height: 18px;
}

QProgressBar::chunk {
    background-color: #10b981;
    border-radius: 5px;
}

/* Scroll Areas */
QScrollArea {
    background: transparent;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background: transparent;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: #f1f5f9;
    width: 8px;
    border-radius: 4px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #cbd5e1;
    border-radius: 4px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background: #94a3b8;
}

QScrollBar::horizontal {
    border: none;
    background: #f1f5f9;
    height: 8px;
    border-radius: 4px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background: #cbd5e1;
    border-radius: 4px;
    min-width: 20px;
}

QScrollBar::handle:horizontal:hover {
    background: #94a3b8;
}

QScrollBar::add-line, QScrollBar::sub-line {
    width: 0px;
    height: 0px;
}

/* Splitter handles */
QSplitter::handle {
    background-color: #e2e8f0;
}

QSplitter::handle:hover {
    background-color: #3b82f6;
}
"""


class MainWindow(QMainWindow):
    """
    Main window
    """

    def __init__(self, parent=None):
        """
        Initialization of the main window
        """
        super(MainWindow, self).__init__(parent)
        # Loads the ui
        self.base_path = os.path.dirname(os.path.realpath(__file__))
        uic.loadUi(
            os.path.join(self.base_path, "GUI", "TM_tabs.ui"),
            self,
        )
        # Sets the window icon
        self.setWindowIcon(
            QIcon(
                os.path.join(
                    self.base_path,
                    "Imagens",
                    "favicon.ico",
                )
            )
        )

        # Reads the configuration file and create the corresponding variables
        self.mat = configparser.ConfigParser()
        self.mat.read(os.path.join(self.base_path, "materials.data"))

        # Creating the list containing all simulations
        self.sim_list = []
        # Stores the number of the current simulation
        self.current_number = 1

        # GA variables
        self.ga_worker = None
        self.ga_best_individual = None
        self.ga_best_fitness = 0.0
        self.ga_best_summary_text = ""
        self.ga_selected_individual = None
        self.ga_selected_row = None
        self.ga_last_df = None
        self._ga_selected_scatter = None
        self._ga_current_display_text = ""
        self._fortran_table_loaded = False
        self.fortran_table_filters = {}
        self._fortran_table_plot_data = None
        self.auto_worker = None
        self.auto_results_data = []
        self.fortran_worker = None
        self._sim_start_time = 0.0
        self._log_count = 0

        self.ApplyModernStyle()
        self.InitBottomStatusAndLogBar()

        self.ConnectSignals()
        # Calls an auxiliary script that contains interface configuration functions
        interface_config.run(self)
        self.SyncOutputFolderDisplays()
        self.CreatePlots()
        self.InitializeStructTable()
        self.InitializeDataTable()
        self.InitializeFortranSamplesTable()
        self.InitializeGATab()
        self.InitializeAutomateTab()
        self.SetupConfigMenu()
        self.SetupHelpMenu()
        self.UpdateInterface()

    def ApplyModernStyle(self):
        """Applies a modern, polished visual stylesheet with rounded corners and elegant accents."""
        self.setStyleSheet(MODERN_APP_STYLESHEET)

    def InitBottomStatusAndLogBar(self):
        """Initializes the unified activity log and simulator status bar at the bottom of the window."""
        self._sim_start_time = 0.0
        self._sim_elapsed_timer = QTimer(self)
        self._sim_elapsed_timer.timeout.connect(self._UpdateSimElapsedDisplay)

        self.status_bar_container = QWidget()
        self.status_bar_container.setObjectName("status_bar_container")
        self.status_bar_container.setStyleSheet(
            "QWidget#status_bar_container {"
            "  background-color: #ffffff;"
            "  border-top: 1.5px solid #cbd5e1;"
            "  padding: 3px 6px;"
            "}"
        )

        main_vbox = QVBoxLayout(self.status_bar_container)
        main_vbox.setContentsMargins(6, 4, 6, 4)
        main_vbox.setSpacing(4)

        # Top Bar Row
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)

        # 1. State Badge
        self.sim_indicator_badge = QLabel("🟢 SIMULADOR PRONTO")
        self.sim_indicator_badge.setObjectName("sim_indicator_badge")
        self.sim_indicator_badge.setStyleSheet(
            "background-color: #ecfdf5; color: #065f46; border: 1.5px solid #10b981; "
            "border-radius: 6px; font-weight: 800; font-size: 11px; padding: 3px 10px;"
        )
        top_row.addWidget(self.sim_indicator_badge)

        # 2. Active Operation Pill
        self.sim_active_op_lbl = QLabel("")
        self.sim_active_op_lbl.setStyleSheet(
            "background-color: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; "
            "border-radius: 5px; font-weight: 700; font-size: 11px; padding: 2px 8px;"
        )
        self.sim_active_op_lbl.setVisible(False)
        top_row.addWidget(self.sim_active_op_lbl)

        # 3. Live Message / Status text
        self.sim_status_msg_lbl = QLabel("Aplicação pronta. Nenhuma simulação em execução.")
        self.sim_status_msg_lbl.setStyleSheet("color: #334155; font-size: 11px; font-weight: 500;")
        top_row.addWidget(self.sim_status_msg_lbl, 1)

        # 4. Mini Progress Bar
        self.sim_status_progress = QProgressBar()
        self.sim_status_progress.setMaximumHeight(14)
        self.sim_status_progress.setMaximumWidth(130)
        self.sim_status_progress.setStyleSheet(
            "QProgressBar { border: 1px solid #cbd5e1; border-radius: 4px; text-align: center; font-size: 10px; background: #f1f5f9; }"
            "QProgressBar::chunk { background-color: #2563eb; border-radius: 3px; }"
        )
        self.sim_status_progress.setVisible(False)
        top_row.addWidget(self.sim_status_progress)

        # 5. Elapsed Timer
        self.sim_status_timer_lbl = QLabel("⏱ 00:00")
        self.sim_status_timer_lbl.setStyleSheet("color: #475569; font-size: 11px; font-family: monospace; font-weight: bold;")
        self.sim_status_timer_lbl.setVisible(False)
        top_row.addWidget(self.sim_status_timer_lbl)

        # 6. Toggle Log Button
        self.btn_toggle_log = QPushButton("📜 Registro / Log (0) ▾")
        self.btn_toggle_log.setStyleSheet(
            "QPushButton { background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 5px; padding: 3px 9px; font-size: 11px; font-weight: 600; }"
            "QPushButton:hover { background-color: #e2e8f0; border-color: #94a3b8; }"
        )
        self.btn_toggle_log.clicked.connect(self.ToggleLogDrawer)
        top_row.addWidget(self.btn_toggle_log)

        main_vbox.addLayout(top_row)

        # Log Drawer (Collapsible)
        self.log_drawer_widget = QWidget()
        self.log_drawer_widget.setVisible(False)
        drawer_layout = QVBoxLayout(self.log_drawer_widget)
        drawer_layout.setContentsMargins(0, 4, 0, 0)
        drawer_layout.setSpacing(4)

        # Drawer Toolbar
        drawer_top = QHBoxLayout()
        drawer_title = QLabel("📋 Histórico de Atividades e Logs de Simulação")
        drawer_title.setStyleSheet("font-weight: 700; color: #1e293b; font-size: 11px;")
        drawer_top.addWidget(drawer_title)
        drawer_top.addStretch()

        btn_clear_log = QPushButton("Limpar")
        btn_clear_log.setStyleSheet("padding: 2px 8px; font-size: 10px;")
        btn_clear_log.clicked.connect(self.ClearGlobalLog)
        drawer_top.addWidget(btn_clear_log)

        btn_copy_log = QPushButton("Copiar")
        btn_copy_log.setStyleSheet("padding: 2px 8px; font-size: 10px;")
        btn_copy_log.clicked.connect(self.CopyGlobalLog)
        drawer_top.addWidget(btn_copy_log)

        btn_hide_log = QPushButton("Ocultar ▴")
        btn_hide_log.setStyleSheet("padding: 2px 8px; font-size: 10px;")
        btn_hide_log.clicked.connect(lambda: self.log_drawer_widget.setVisible(False))
        drawer_top.addWidget(btn_hide_log)

        drawer_layout.addLayout(drawer_top)

        # Log Console
        self.txt_global_log = QTextEdit()
        self.txt_global_log.setReadOnly(True)
        self.txt_global_log.setMaximumHeight(120)
        self.txt_global_log.setStyleSheet(
            "QTextEdit { background-color: #0f172a; color: #f8fafc; font-family: 'Consolas', 'Cascadia Code', monospace; "
            "font-size: 11px; border: 1px solid #334155; border-radius: 6px; padding: 4px; }"
        )
        drawer_layout.addWidget(self.txt_global_log)

        main_vbox.addWidget(self.log_drawer_widget)

        # Add to centralwidget layout
        if hasattr(self, "centralwidget") and self.centralwidget.layout() is not None:
            self.centralwidget.layout().addWidget(self.status_bar_container)

        self._log_count = 0
        self.LogMessage("Aplicação inicializada com sucesso. Simulador pronto.", "info")

    def _UpdateSimElapsedDisplay(self):
        if self._sim_start_time > 0:
            elapsed_sec = int(time.time() - self._sim_start_time)
            mins = elapsed_sec // 60
            secs = elapsed_sec % 60
            self.sim_status_timer_lbl.setText(f"⏱ {mins:02d}:{secs:02d}")

    def SetSimulatorState(self, state: str, op_name: str = ""):
        """Updates simulator state badge and timer (state: 'idle' or 'running')."""
        if state == "running":
            self.sim_indicator_badge.setText("⚡ SIMULADOR EXECUTANDO...")
            self.sim_indicator_badge.setStyleSheet(
                "background-color: #fef3c7; color: #92400e; border: 2px solid #f59e0b; "
                "border-radius: 6px; font-weight: 800; font-size: 11px; padding: 3px 10px;"
            )
            self.sim_active_op_lbl.setText(f"[{op_name}]" if op_name else "[Executando]")
            self.sim_active_op_lbl.setVisible(True)
            self.sim_status_progress.setVisible(True)
            self.sim_status_timer_lbl.setVisible(True)
            self._sim_start_time = time.time()
            self._sim_elapsed_timer.start(500)
            self._UpdateSimElapsedDisplay()
        else:
            self.sim_indicator_badge.setText("🟢 SIMULADOR PRONTO")
            self.sim_indicator_badge.setStyleSheet(
                "background-color: #ecfdf5; color: #065f46; border: 1.5px solid #10b981; "
                "border-radius: 6px; font-weight: 700; font-size: 11px; padding: 3px 10px;"
            )
            self.sim_active_op_lbl.setVisible(False)
            self.sim_status_progress.setVisible(False)
            self.sim_status_timer_lbl.setVisible(False)
            self._sim_elapsed_timer.stop()
            self._sim_start_time = 0.0

    def LogMessage(self, msg: str, level: str = "info"):
        """Logs timestamped message to both status bar and log drawer."""
        now_str = datetime.now().strftime("%H:%M:%S")
        self.sim_status_msg_lbl.setText(f"[{now_str}] {msg}")
        if hasattr(self, "txt_global_log"):
            colors = {
                "info": "#94a3b8",
                "running": "#38bdf8",
                "success": "#4ade80",
                "warning": "#facc15",
                "error": "#f87171"
            }
            c = colors.get(level, "#cbd5e1")
            self.txt_global_log.append(f"<span style='color: #64748b;'>[{now_str}]</span> <span style='color: {c};'>{html.escape(msg)}</span>")
            self.txt_global_log.verticalScrollBar().setValue(self.txt_global_log.verticalScrollBar().maximum())
            self._log_count = getattr(self, "_log_count", 0) + 1
            if hasattr(self, "btn_toggle_log"):
                vis = self.log_drawer_widget.isVisible()
                self.btn_toggle_log.setText(f"📜 Log ({self._log_count}) {'▴' if vis else '▾'}")

    def ToggleLogDrawer(self):
        vis = not self.log_drawer_widget.isVisible()
        self.log_drawer_widget.setVisible(vis)
        self.btn_toggle_log.setText(f"📜 Log ({getattr(self, '_log_count', 0)}) {'▴' if vis else '▾'}")

    def ClearGlobalLog(self):
        if hasattr(self, "txt_global_log"):
            self.txt_global_log.clear()
            self._log_count = 0
            if hasattr(self, "btn_toggle_log"):
                self.btn_toggle_log.setText("📜 Log (0) ▾")

    def CopyGlobalLog(self):
        if hasattr(self, "txt_global_log"):
            QApplication.clipboard().setText(self.txt_global_log.toPlainText())
            self.LogMessage("Conteúdo do log copiado para a área de transferência.", "info")

    # GUI ##############################################################################
    def ConnectSignals(self):
        """
        Connect the signals from buttons to functions.
        Disables any button that should only be enabled after a new simulation is
        created.
        """
        # Lower bar
        # self.new_btn.clicked.connect(self.new_sim_window.show)
        self.new_btn.clicked.connect(self.OpenNewSimWindow)
        self.save_btn.clicked.connect(self.SaveSimulation)
        self.load_btn.clicked.connect(self.LoadSimulation)
        self.delete_btn.clicked.connect(self.DeleteSimulation)
        self.output_folder_btn.clicked.connect(self.ChooseOutputFolder)
        self.simulation_cbox.currentIndexChanged.connect(self.UpdateInterface)
        self.rename_btn.clicked.connect(self.RenameSimulation)
        self.copy_btn.clicked.connect(self.CopySimulation)

        # Design tab
        self.add_well_btn.clicked.connect(lambda: self.UpdateStructure("AddWell"))
        self.add_barrier_btn.clicked.connect(lambda: self.UpdateStructure("AddBarrier"))
        self.replace_well_btn.clicked.connect(
            lambda: self.UpdateStructure("ReplaceWell")
        )
        self.replace_barrier_btn.clicked.connect(
            lambda: self.UpdateStructure("ReplaceBarrier")
        )
        self.insert_well_btn.clicked.connect(lambda: self.UpdateStructure("InsertWell"))
        self.insert_barrier_btn.clicked.connect(
            lambda: self.UpdateStructure("InsertBarrier")
        )
        self.remove_selected_btn.clicked.connect(
            lambda: self.UpdateStructure("RemoveSelected")
        )
        self.remove_last_btn.clicked.connect(lambda: self.UpdateStructure("RemoveLast"))
        self.remove_all_btn.clicked.connect(lambda: self.UpdateStructure("RemoveAll"))
        self.well_ml_spb.valueChanged.connect(self.UpdateUnits)
        self.well_nm_spb.valueChanged.connect(self.UpdateUnits)
        self.barrier_ml_spb.valueChanged.connect(self.UpdateUnits)
        self.barrier_nm_spb.valueChanged.connect(self.UpdateUnits)

        # Simulation tab
        self.sim_Efield_btn.clicked.connect(self.Campo)
        # Electric field button and spinbox are disabled because they don't work
        self.sim_Efield_btn.setEnabled(False)
        self.sim_Efield_spb.setEnabled(False)
        self.sim_run_btn.clicked.connect(lambda: self.RunSimulation())
        self.sim_plot_results_btn.clicked.connect(lambda: self.PlotSimResults())
        self.sim_plot_structure_btn.clicked.connect(lambda: self.PlotStructure())
        self.sim_clear_plot_btn.clicked.connect(self.ClearSimPlot)
        self.sim_dx_ml_spb.valueChanged.connect(self.UpdateUnits)
        self.sim_dx_nm_spb.valueChanged.connect(self.UpdateUnits)
        self.sim_central_layer_spb.valueChanged.connect(lambda: self.PlotStructure())

        # Absorption tab
        self.abs_run_btn.clicked.connect(lambda: self.RunAbsorption())
        self.abs_plot_btn.clicked.connect(lambda: self.PlotAbsorption())
        self.abs_clear_plot_btn.clicked.connect(self.ClearAbsPlot)

        # Transmission tab
        self.tra_run_btn.clicked.connect(lambda: self.RunTransmission())
        self.tra_plot_btn.clicked.connect(lambda: self.PlotTransmission())
        self.tra_clear_plot_btn.clicked.connect(lambda: self.ClearTransPlot())

        # Photocurrent tab
        self.pc_run_btn.clicked.connect(lambda: self.RunPhotocurrent())
        self.pc_clear_plot_btn.clicked.connect(lambda: self.ClearPhotocurrentPlot())

        # GA tab
        # self.ga_btn.clicked.connect(self.Sobre)

        # Automation tab (signals connected in InitializeAutomateTab)

        # Fortran Sim tab
        if hasattr(self, "tabWidget"):
            self.tabWidget.currentChanged.connect(self.OnMainTabChanged)
        self.fortran_run_btn.clicked.connect(self.RunFortranSimulation)
        self.fortran_s0_spb.valueChanged.connect(self.PlotFortranBandStructure)
        self.fortran_s1_spb.valueChanged.connect(self.PlotFortranBandStructure)
        self.fortran_s2_spb.valueChanged.connect(self.PlotFortranBandStructure)
        self.fortran_s3_spb.valueChanged.connect(self.PlotFortranBandStructure)
        self.fortran_s4_spb.valueChanged.connect(self.PlotFortranBandStructure)
        self.fortran_s5_spb.valueChanged.connect(self.PlotFortranBandStructure)
        self.fortran_s6_spb.valueChanged.connect(self.PlotFortranBandStructure)

        # Advanced options tab
        self.adv_nm_layers_chkbx.stateChanged.connect(self.UpdateUnits)
        self.adv_save_gui_config.clicked.connect(self.SaveGUIConfigFile)

        # Actions from menus
        # File Menu
        # self.action_new.triggered.connect(self.NewSimulation)
        self.action_new.triggered.connect(self.OpenNewSimWindow)
        self.action_load.triggered.connect(self.LoadSimulation)
        self.action_save.triggered.connect(self.SaveSimulation)
        self.action_exit.triggered.connect(exit)
        # Help Menu
        self.action_about.triggered.connect(self.Sobre)

    def UpdateUnits(self):
        """
        This function is used to update the layer thickness spinboxes, making sure that
        the value in nm corresponds an integer multiple of the lattice parameter,
        defined by the monolayer's spinboxes, and vice versa.
        This routine is divided in two parts (by the try-except), things that can be
        determined before creating a simulation and the ones that depend on the lattice
        parameter.
        """
        # If the user prefers to use nanometers instead of monolayers:
        if hasattr(self, "auto_step_spb"):
            if self.adv_nm_layers_chkbx.checkState():
                self.auto_step_spb.setSuffix(" nm")
                self.auto_final_spb.setSuffix(" nm")
                self.auto_init_spb.setSuffix(" nm")
            else:
                self.auto_step_spb.setSuffix(" ML")
                self.auto_final_spb.setSuffix(" ML")
                self.auto_init_spb.setSuffix(" ML")

        # The lattice parameter is only defined after a simulation was created.
        try:  # Gets the current simulation and the value of the lattice parameter
            sim = self.sim_list[self.simulation_cbox.currentIndex()]
            ml = sim.latpar / 2.0
        except:
            return

        # Identifies which spinbox was modified (the one which called this function)
        op = self.sender()

        # If the user prefers to use nanometers instead of monolayers:
        if self.adv_nm_layers_chkbx.checkState():
            # Disable the ml spinboxes, so that the user cannot interact with them
            self.barrier_ml_spb.setEnabled(False)
            self.well_ml_spb.setEnabled(False)
            self.sim_dx_ml_spb.setEnabled(False)
            # Enable the nm spinboxes
            self.barrier_nm_spb.setEnabled(True)
            self.well_nm_spb.setEnabled(True)
            self.sim_dx_nm_spb.setEnabled(True)

            # Set the value of the monolayers spinboxes, base on the nm values (converted to meters)
            # if op == self.barrier_nm_spb:
            self.barrier_ml_spb.setValue(
                np.round(self.barrier_nm_spb.value() * NM / ml, 3)
            )
            # if op == self.well_nm_spb:
            self.well_ml_spb.setValue(np.round(self.well_nm_spb.value() * NM / ml, 3))
            self.sim_dx_ml_spb.setValue(
                np.round(self.sim_dx_nm_spb.value() * NM / ml, 3)
            )

        # If the user is using monolayers
        else:
            # Enable the ml spinboxes, so that the user can interact with them
            self.barrier_ml_spb.setEnabled(True)
            self.well_ml_spb.setEnabled(True)
            self.sim_dx_ml_spb.setEnabled(True)
            # Disable the nm spinboxes
            self.barrier_nm_spb.setEnabled(False)
            self.well_nm_spb.setEnabled(False)
            self.sim_dx_nm_spb.setEnabled(False)
            # if op == self.barrier_ml_spb:
            # lp is in meters, converts to nm
            self.barrier_nm_spb.setValue(self.barrier_ml_spb.value() * ml / NM)
            # if op == self.well_ml_spb:
            self.well_nm_spb.setValue(self.well_ml_spb.value() * ml / NM)
            self.sim_dx_nm_spb.setValue(self.sim_dx_ml_spb.value() * ml / NM)

    def CreatedNewSimulation(self):
        """
        Called after a new simulation is created, just to update the interface
        """
        self.simulation_cbox.setCurrentIndex(len(self.sim_list) - 1)
        # Defines the output folder based on the simulation title
        sim = self.sim_list[
            -1
        ]  # The new simulation was just appended to the end of the list
        sim.output_folder = str(self.output_folder_line.text())
        self.PlotStructure()
        self.UpdateStructureTable()
        self.UpdateSimList()
        self.UpdateInterface()
        self.UpdateLayerCount()
        self.UpdateUnits()

    def UpdateSimList(self):
        """
        If simulations are added, deleted or loaded, needs to update the list
        """
        # Clears the simulation list
        self.simulation_cbox.clear()
        # Fills the simulation combobox with every simulation from the list
        for sim in self.sim_list:
            self.simulation_cbox.addItem(sim.title)
        self.simulation_cbox.setCurrentIndex(len(self.sim_list) - 1)

    def UpdateStructure(self, op):
        """
        Function that adds, inserts or removes layers from the surface, based on the user choice on
        the GUI.
        op is the operation the user wants to perform, defined in self.ConnectSignals
        """
        sim = self.sim_list[self.simulation_cbox.currentIndex()]

        # Adding a new well
        if op == "AddWell":
            sim.AddWell(self.well_nm_spb.value())

        # Adding a new barrier
        elif op == "AddBarrier":
            sim.AddBarrier(self.barrier_nm_spb.value())

        # Replacing a well
        elif op == "ReplaceWell":
            index = self.struct_table.currentRow()  # Selected table line
            if index == -1:  # In case there is nothing to replace
                return
            sim.ReplaceWell(self.well_nm_spb.value(), index)

        # Replacing a barrier
        elif op == "ReplaceBarrier":
            index = self.struct_table.currentRow()  # Selected table line
            if index == -1:  # In case there is nothing to replace
                return
            sim.ReplaceBarrier(self.barrier_nm_spb.value(), index)

        # Inserting a well
        elif op == "InsertWell":
            index = self.struct_table.currentRow()  # Selected table line
            sim.InsertWell(self.well_nm_spb.value(), index)

        # Inserting a barrier
        elif op == "InsertBarrier":
            index = self.struct_table.currentRow()  # Selected table line
            sim.InsertBarrier(self.barrier_nm_spb.value(), index)

        elif op == "RemoveSelected":
            index = self.struct_table.currentRow()  # Linha da tabela selecionada
            if len(sim.estrutura) > 1:  # If there is a structure, delete the last item
                sim.RemoveSelected(index)
            else:
                op = "RemoveAll"  # Just to avoid repeating 5 lines of code

        elif op == "RemoveLast":
            if len(sim.estrutura) > 1:  # If there is a structure, delete the last item
                sim.RemoveSelected(-1)
            else:
                op = "RemoveAll"  # Just to avoid repeating 5 lines of code

        # This is not "elif" just so that the "else" from RemoveSelected and RemoveLast work
        if op == "RemoveAll":
            sim.RemoveAll()

        # Since the structure was modified, define this simulation as not ran
        sim.sim_ran = False
        sim.abs_ran = False

        # Updates table, graph and buttons
        self.UpdateStructureTable()
        self.PlotStructure(sim)
        self.UpdateInterface()
        self.UpdateLayerCount()

    def UpdateLayerCount(self):
        """
        Calculates the number of layers and updates the Spinbox on the advanced tab
        """

        try:  # If there is a simulation and this simulation has at least one layer
            sim = self.sim_list[self.simulation_cbox.currentIndex()]
            layers = len(sim.estrutura)
            self.adv_total_layers_spb.setValue(layers)
        except:  # If there is no simulation or it doesn't have any layers yet
            self.adv_total_layers_spb.setValue(0)

    def UpdateInterface(self):
        """
        Updates the buttons, list of simulations available on the simulations combobox.
        This simulation is called almost everytime after user interaction.
        """
        # Check whether there are simulations
        # The selected simulation defines whether some options on the interface are available
        # If there are no simulations, disable most buttons, except "load" and "new"
        if len(self.sim_list) == 0:
            # Lower bar
            self.save_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.copy_btn.setEnabled(False)
            self.rename_btn.setEnabled(False)
            self.output_folder_btn.setEnabled(False)
            # Structure tab
            self.add_well_btn.setEnabled(False)
            self.add_barrier_btn.setEnabled(False)
            self.replace_well_btn.setEnabled(False)
            self.replace_barrier_btn.setEnabled(False)
            self.insert_well_btn.setEnabled(False)
            self.insert_barrier_btn.setEnabled(False)
            self.remove_selected_btn.setEnabled(False)
            self.remove_last_btn.setEnabled(False)
            self.remove_all_btn.setEnabled(False)
            # Layer thickness spinboxes
            self.barrier_ml_spb.setEnabled(False)
            self.well_ml_spb.setEnabled(False)
            self.barrier_nm_spb.setEnabled(False)
            self.well_nm_spb.setEnabled(False)
            # Simulation tab
            self.sim_Efield_btn.setEnabled(False)
            self.sim_run_btn.setEnabled(False)
            self.sim_plot_results_btn.setEnabled(False)
            self.sim_plot_structure_btn.setEnabled(False)
            # Absorption tab
            self.abs_run_btn.setEnabled(False)
            self.abs_plot_btn.setEnabled(False)
            # Transmission tab
            self.tra_plot_btn.setEnabled(False)
            # Photocurrent tab
            self.pc_run_btn.setEnabled(False)
            # Genetic algorithm tab
            self.ga_run_btn.setEnabled(True)

            self.ClearStructureTable()
            self.ClearSimPlot()
            self.ClearAbsPlot()
            self.ClearMaterialData()

        else:
            # current simulation
            sim = self.sim_list[self.simulation_cbox.currentIndex()]
            # If there are available simulations, some buttons must to be enabled
            # Lower bar
            self.save_btn.setEnabled(True)
            self.delete_btn.setEnabled(True)
            self.copy_btn.setEnabled(True)
            self.rename_btn.setEnabled(True)
            self.output_folder_btn.setEnabled(True)
            # Structure tab
            self.add_well_btn.setEnabled(True)
            self.add_barrier_btn.setEnabled(True)
            self.insert_well_btn.setEnabled(True)
            self.insert_barrier_btn.setEnabled(True)
            # Layer thickness spinboxes
            if self.adv_nm_layers_chkbx.checkState():
                self.barrier_nm_spb.setEnabled(True)
                self.well_nm_spb.setEnabled(True)
            else:
                self.barrier_ml_spb.setEnabled(True)
                self.well_ml_spb.setEnabled(True)

            if (
                len(sim.estrutura) > 0
            ):  # If the structure is has layers, a simulation may be run
                # Simulation tab
                self.sim_run_btn.setEnabled(True)
                self.sim_plot_structure_btn.setEnabled(True)
                # Structure tab
                self.replace_well_btn.setEnabled(True)
                self.replace_barrier_btn.setEnabled(True)
                self.remove_selected_btn.setEnabled(True)
                self.remove_last_btn.setEnabled(True)
                self.remove_all_btn.setEnabled(True)
                # Genetic algorithm tab
                self.ga_run_btn.setEnabled(True)

                if sim.sim_ran is False:  # If the simulation has not yet been executed
                    # Simulation tab
                    self.sim_plot_results_btn.setEnabled(False)
                    # Absorption tab
                    self.abs_run_btn.setEnabled(False)
                    # Transmission tab
                    self.tra_plot_btn.setEnabled(False)
                    # Photocurrent tab
                    self.pc_run_btn.setEnabled(False)
                else:
                    # Simulation tab
                    self.sim_plot_results_btn.setEnabled(True)
                    # Absorption tab
                    self.abs_run_btn.setEnabled(True)
                    # Transmission tab
                    self.tra_plot_btn.setEnabled(True)
                    # Photocurrent tab
                    self.pc_run_btn.setEnabled(True)

                if (
                    sim.abs_ran is False
                ):  # If the absorption has not yet been calculated
                    self.abs_plot_btn.setEnabled(False)
                else:
                    self.abs_plot_btn.setEnabled(True)
            else:
                # Simulation tab
                self.sim_run_btn.setEnabled(False)
                # Structure tab
                self.replace_well_btn.setEnabled(False)
                self.replace_barrier_btn.setEnabled(False)
                self.remove_selected_btn.setEnabled(False)
                self.remove_last_btn.setEnabled(False)
                self.remove_all_btn.setEnabled(False)
                # Simulation tab
                self.sim_Efield_btn.setEnabled(False)
                self.sim_plot_results_btn.setEnabled(False)
                self.sim_plot_structure_btn.setEnabled(False)
                # Absorption tab
                self.abs_run_btn.setEnabled(False)
                # Genetic algorithm tab
                self.ga_run_btn.setEnabled(True)
            self.UpdateStructureTable()
            self.FillMaterialData()

    # Information about the materials used on the simulation, shown on the structure tab
    def ClearMaterialData(self):
        """
        Removes the information about the materials from the Structure Tab. This
        function is called when there is no simulation.
        """
        self.lbl_lattice_parameter_val.setText("")
        self.lbl_barrier_material_val.setText("")
        self.lbl_barrier_effective_mass_val.setText("")
        self.lbl_barrier_electronic_potential_val.setText("")
        self.lbl_barrier_non_parabolicity_val.setText("")
        self.lbl_well_material_val.setText("")
        self.lbl_well_effective_mass_val.setText("")
        self.lbl_well_electronic_potential_val.setText("")
        self.lbl_well_non_parabolicity_val.setText("")

    def FillMaterialData(self):
        """
        Fills the structure tab with data for the selected materials.
        """
        sim = self.sim_list[self.simulation_cbox.currentIndex()]
        self.lbl_lattice_parameter_val.setText(f"{1E10*sim.latpar:.3f} Ang")
        self.lbl_barrier_material_val.setText(f"{sim.barrier}")
        self.lbl_barrier_effective_mass_val.setText(f"{sim.m_eff_ct_barrier:.3e}")
        self.lbl_barrier_electronic_potential_val.setText(
            f"{1E3*sim.pot_barrier:.2f} meV"
        )
        self.lbl_barrier_non_parabolicity_val.setText(f"{sim.e_nonparab_barrier:.3e}")
        self.lbl_well_material_val.setText(f"{sim.well}")
        self.lbl_well_effective_mass_val.setText(f"{sim.m_eff_ct_well:.3e}")
        self.lbl_well_electronic_potential_val.setText(f"{1E3*sim.pot_well:.2f} meV")
        self.lbl_well_non_parabolicity_val.setText(f"{sim.e_nonparab_well:.3e}")

    # Tables ###########################################################################
    # Structure
    def InitializeStructTable(self):
        """
        Initializes the table presenting the structure
        """
        self.struct_table.setColumnCount(4)
        self.struct_table.setColumnWidth(0, 60)
        self.struct_table.setColumnWidth(1, 60)
        self.struct_table.setColumnWidth(2, 40)
        self.struct_table.setColumnWidth(3, 40)
        self.struct_table.move(0, 0)
        self.struct_table.setHorizontalHeaderLabels(["Material", "Feature", "ML", "nm"])
        self.struct_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.struct_table.setSelectionBehavior(QAbstractItemView.SelectRows)

    def UpdateStructureTable(self):
        """
        Function that updates the table at the structure tab. The table is erased every time and
        rewrites it all again usig information from sim.material and sim.estrutura.
        """
        sim = self.sim_list[self.simulation_cbox.currentIndex()]
        self.struct_table.setRowCount(len(sim.material))
        for i, material in enumerate(sim.material):
            col0 = QTableWidgetItem(material)  # Creates the item
            col0.setTextAlignment(0x0084)  # Align h center, v baseline
            col1 = QTableWidgetItem(sim.feature[i])
            col1.setTextAlignment(0x0084)  # Align h center, v baseline
            col2 = QTableWidgetItem(f"{sim.estrutura[i] / ( sim.latpar / 2):.0f}")
            col2.setTextAlignment(0x0082)  # Align h right, v baseline
            col3 = QTableWidgetItem(f"{sim.estrutura[i] / NM:.3f}")
            col3.setTextAlignment(0x0082)  # Align h right, v baseline
            self.struct_table.setItem(i, 0, col0)
            self.struct_table.setItem(i, 1, col1)
            self.struct_table.setItem(i, 2, col2)
            self.struct_table.setItem(i, 3, col3)
            # Adjusts the line heigth
            self.struct_table.setRowHeight(i, 18)
            # Corrects the line index (without this correction, it starts from 1, instead of 0)
            self.struct_table.setVerticalHeaderItem(i, QTableWidgetItem(f"{i}"))
        # Adjusts the column width
        self.struct_table.setColumnWidth(0, 60)
        self.struct_table.setColumnWidth(1, 60)
        self.struct_table.setColumnWidth(2, 40)
        self.struct_table.setColumnWidth(3, 40)

    def ClearStructureTable(self):
        self.struct_table.setRowCount(0)

    # Simulation data
    def InitializeDataTable(self):
        """
        Initializes the table presenting the results
        """
        self.data_table.setColumnCount(1)
        self.data_table.setColumnWidth(0, 120)
        # self.struct_table.setColumnWidth(1, 60)
        # self.struct_table.setColumnWidth(2, 40)
        # self.struct_table.setColumnWidth(3, 40)
        self.data_table.move(0, 0)
        # Definition of the header labels
        h_lbls = ["Energy (meV)"]
        self.data_table.setHorizontalHeaderLabels(h_lbls)
        self.data_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.data_table.setSelectionBehavior(QAbstractItemView.SelectRows)

    def UpdateDataTable(self):
        """
        Function that updates the table at the data tab. The table is erased every time and
        rewritten with data from the latest simulation
        """
        self.ClearDataTable()
        # Gets the current selected simulation
        sim = self.sim_list[self.simulation_cbox.currentIndex()]

        self.data_table.setRowCount(len(sim.sim_Energias))
        for i, energia in enumerate(sim.sim_Energias):

            col0 = QTableWidgetItem(f"{energia * 1000:.3f} meV")  # Creates the item
            col0.setTextAlignment(0x0084)  # Align h center, v baseline
            # col1 = QTableWidgetItem(sim.feature[i])
            # col1.setTextAlignment(0x0084)  # Align h center, v baseline
            # col2 = QTableWidgetItem(f"{sim.estrutura[i] / ( sim.latpar / 2):.0f}")
            # col2.setTextAlignment(0x0082)  # Align h right, v baseline
            # col3 = QTableWidgetItem(f"{sim.estrutura[i] / NM:.3f}")
            # col3.setTextAlignment(0x0082)  # Align h right, v baseline
            self.data_table.setItem(i, 0, col0)
            # self.struct_table.setItem(i, 1, col1)
            # self.struct_table.setItem(i, 2, col2)
            # self.struct_table.setItem(i, 3, col3)
            # Adjusts the line heigth
            self.data_table.setRowHeight(i, 18)
            # Corrects the line index (without this correction, it starts from 1, instead of 0)
            self.data_table.setVerticalHeaderItem(i, QTableWidgetItem(f"{i}"))
        # Adjusts the column width
        self.data_table.setColumnWidth(0, 120)
        # self.struct_table.setColumnWidth(1, 60)
        # self.struct_table.setColumnWidth(2, 40)
        # self.struct_table.setColumnWidth(3, 40)

    def ClearDataTable(self):
        self.data_table.setRowCount(0)

    # Plots ############################################################################
    def CreatePlots(self):
        """
        Initial configuration of the plots
        """
        # Simulation
        self.sim_fig = plt.figure()
        self.sim_canvas = FigureCanvas(self.sim_fig)
        self.sim_plot_layout.addWidget(self.sim_canvas)
        self.sim_nav = NavigationToolbar(self.sim_canvas, self.sim_tab)
        self.sim_plot_layout.addWidget(self.sim_nav)
        self.sim_subplot = self.sim_fig.add_subplot(111)
        self.sim_subplot.grid(True, axis="y")
        self.show()
        # Absorption
        self.abs_fig = plt.figure()
        self.abs_canvas = FigureCanvas(self.abs_fig)
        self.abs_plot_layout.addWidget(self.abs_canvas)
        self.abs_nav = NavigationToolbar(self.abs_canvas, self.abs_tab)
        self.abs_plot_layout.addWidget(self.abs_nav)
        self.abs_subplot = self.abs_fig.add_subplot(111)
        self.abs_subplot.grid(True, axis="both")
        self.show()
        # Transmission
        self.tra_fig = plt.figure()
        # self.tra_fig, self.tra_ax = plt.subplots(nrows=1, ncols=1)
        self.tra_canvas = FigureCanvas(self.tra_fig)
        self.tra_plot_layout.addWidget(self.tra_canvas)
        self.tra_nav = NavigationToolbar(self.tra_canvas, self.transmission_tab)
        self.tra_plot_layout.addWidget(self.tra_nav)
        self.tra_subplot = self.tra_fig.add_subplot(111)
        self.tra_subplot.grid(True, axis="both")
        self.show()
        # Photocurrent
        self.pc_fig = plt.figure()
        self.pc_canvas = FigureCanvas(self.pc_fig)
        self.pc_plot_layout.addWidget(self.pc_canvas)
        self.pc_nav = NavigationToolbar(self.pc_canvas, self.photocurrent_tab)
        self.pc_plot_layout.addWidget(self.pc_nav)
        self.pc_subplot = self.pc_fig.add_subplot(111)
        self.pc_subplot.grid(True, axis="both")
        self.show()
        # Genetic algorithm evolution plot
        self.ga_fig = plt.figure()
        self.ga_canvas = FigureCanvas(self.ga_fig)
        self.ga_plot_layout.addWidget(self.ga_canvas)
        self.ga_nav = NavigationToolbar(self.ga_canvas, self.ga_tab)
        self.ga_plot_layout.addWidget(self.ga_nav)
        self.ga_subplot = self.ga_fig.add_subplot(111)
        self.ga_subplot.grid(True, axis="both")
        self.show()

        # Genetic algorithm individual inspection plots (3 subplots in bottom panel)
        self.ga_ind_fig1 = plt.figure()
        self.ga_ind_canvas1 = FigureCanvas(self.ga_ind_fig1)
        self.ga_ind_plot_layout1.addWidget(self.ga_ind_canvas1)
        self.ga_ind_nav1 = NavigationToolbar(self.ga_ind_canvas1, self.ga_tab)
        self.ga_ind_plot_layout1.addWidget(self.ga_ind_nav1)
        self.ga_ind_subplot1 = self.ga_ind_fig1.add_subplot(111)
        self.ga_ind_subplot1.grid(True, axis="both")

        self.ga_ind_fig2 = plt.figure()
        self.ga_ind_canvas2 = FigureCanvas(self.ga_ind_fig2)
        self.ga_ind_plot_layout2.addWidget(self.ga_ind_canvas2)
        self.ga_ind_nav2 = NavigationToolbar(self.ga_ind_canvas2, self.ga_tab)
        self.ga_ind_plot_layout2.addWidget(self.ga_ind_nav2)
        self.ga_ind_subplot2 = self.ga_ind_fig2.add_subplot(111)
        self.ga_ind_subplot2.grid(True, axis="both")

        self.ga_ind_fig3 = plt.figure()
        self.ga_ind_canvas3 = FigureCanvas(self.ga_ind_fig3)
        self.ga_ind_plot_layout3.addWidget(self.ga_ind_canvas3)
        self.ga_ind_nav3 = NavigationToolbar(self.ga_ind_canvas3, self.ga_tab)
        self.ga_ind_plot_layout3.addWidget(self.ga_ind_nav3)
        self.ga_ind_subplot3 = self.ga_ind_fig3.add_subplot(111)
        self.ga_ind_subplot3.grid(True, axis="both")

        for nav in (self.ga_ind_nav1, self.ga_ind_nav2, self.ga_ind_nav3):
            nav.setIconSize(QSize(13, 13))
            nav.setMaximumHeight(26)
            nav.setStyleSheet("QToolBar { spacing: 1px; padding: 0px; border: none; background: transparent; } QToolButton { max-height: 20px; max-width: 20px; padding: 1px; }")

        # Connect click event on GA evolution plot for interactive individual inspection
        self.ga_canvas.mpl_connect("button_press_event", self.OnGAEvolutionPlotClicked)
        # Fortran simulation plots (3 graphs: Potencial+WF, OscStr, Photocurrent)
        self.fortran_fig1 = plt.figure()
        self.fortran_canvas1 = FigureCanvas(self.fortran_fig1)
        self.fortran_plot_layout1.addWidget(self.fortran_canvas1)
        self.fortran_nav1 = NavigationToolbar(self.fortran_canvas1, self.fortran_tab)
        self.fortran_plot_layout1.addWidget(self.fortran_nav1)
        self.fortran_subplot1 = self.fortran_fig1.add_subplot(111)
        self.fortran_subplot1.grid(True, axis="both")
        self.show()

        self.fortran_fig2 = plt.figure()
        self.fortran_canvas2 = FigureCanvas(self.fortran_fig2)
        self.fortran_plot_layout2.addWidget(self.fortran_canvas2)
        self.fortran_nav2 = NavigationToolbar(self.fortran_canvas2, self.fortran_tab)
        self.fortran_plot_layout2.addWidget(self.fortran_nav2)
        self.fortran_subplot2 = self.fortran_fig2.add_subplot(111)
        self.fortran_subplot2.grid(True, axis="both")
        self.show()

        self.fortran_fig3 = plt.figure()
        self.fortran_canvas3 = FigureCanvas(self.fortran_fig3)
        self.fortran_plot_layout3.addWidget(self.fortran_canvas3)
        self.fortran_nav3 = NavigationToolbar(self.fortran_canvas3, self.fortran_tab)
        self.fortran_plot_layout3.addWidget(self.fortran_nav3)
        self.fortran_subplot3 = self.fortran_fig3.add_subplot(111)
        self.fortran_subplot3.grid(True, axis="both")
        self.show()

        # Fortran table filtered samples plot
        if hasattr(self, "fortran_table_plot_layout"):
            self.fortran_table_fig = plt.figure()
            self.fortran_table_canvas = FigureCanvas(self.fortran_table_fig)
            self.fortran_table_plot_layout.addWidget(self.fortran_table_canvas)
            self.fortran_table_nav = NavigationToolbar(self.fortran_table_canvas, self.fortran_tab)
            self.fortran_table_nav.setIconSize(QSize(13, 13))
            self.fortran_table_nav.setMaximumHeight(26)
            self.fortran_table_nav.setStyleSheet("QToolBar { spacing: 1px; padding: 0px; border: none; background: transparent; } QToolButton { max-height: 20px; max-width: 20px; padding: 1px; }")
            self.fortran_table_plot_layout.addWidget(self.fortran_table_nav)
            self.fortran_table_subplot = self.fortran_table_fig.add_subplot(111)
            self.fortran_table_subplot.grid(True, linestyle=":", alpha=0.55)
            self.fortran_table_canvas.mpl_connect("button_press_event", self.OnFortranTablePlotClicked)

        self.PlotFortranBandStructure()

    # Structure and Wave Function
    def PlotStructure(self, sim=None):
        """
        Function that interprets the structure data and create arrays to plot the graph.
        It is necessary to analyze the position and potential arrays in order to correctly plot the
        structure. In the interface between two materials the position is the same, but the energy
        is different. If the material is repeated, but the position is different, there is no need
        to create another point.
        """
        # If this function is called without a specified sim, it gets the selected one from the
        # interface combobox
        if sim is None:
            try:  # Gets the current selected simulation
                sim = self.sim_list[self.simulation_cbox.currentIndex()]
            except:
                print("There is no simulation to choose from, create one first.")
                return

        self.ClearSimPlot()
        if len(sim.estrutura) < 1:  # Only continue if there is a structure
            return
        # Creating the x and energy arrays  just for the plot
        sim.x_graf = np.array([0])
        sim.v_graf = np.array([sim.pot[0]])

        # Going through the layers and creating the arrays iteratively
        for i, en in enumerate(sim.pot):
            if (
                en == sim.v_graf[-1]
            ):  # If this layer has the same energy as the previous one
                sim.x_graf = np.append(sim.x_graf, [sim.x_graf[-1] + sim.estrutura[i]])
                sim.v_graf = np.append(sim.v_graf, [en])
            else:
                sim.x_graf = np.append(
                    sim.x_graf, [sim.x_graf[-1], sim.x_graf[-1] + sim.estrutura[i]]
                )
                sim.v_graf = np.append(sim.v_graf, [en, en])

        # Changing x and y to [nm] and [meV]
        sim.x_graf = 1.0e9 * sim.x_graf
        sim.v_graf = 1.0e3 * sim.v_graf

        # Centering the structure around 0
        # sim.x_graf = sim.x_graf - (np.max(sim.x_graf) - np.min(sim.x_graf)) / 2.0

        layers = len(sim.estrutura)
        central_layer = self.sim_central_layer_spb.value()
        """
        This part of the code was used to put the 0 of the x-axis in the center of a 
        layer, but this doesn't work for Pedro's photocurrent calculations, therefore
        the 0 will be at the interfaces.
        # Creating the x-axis array - the x-axis 0 is centered in the target layer
        if central_layer < 0:
            x0 = 0.0
        else:
            if (
                central_layer > layers - 1
            ):  # In case the number of the layer exceeds the limit
                central_layer = layers - 1
            left_x_central = np.sum(
                sim.estrutura[0:central_layer]
            )  # Thickness of layers before
            central_thickness = sim.estrutura[
                central_layer
            ]  # Thickness of target layers
            x0 = -left_x_central - central_thickness / 2.0
        x0 = x0 / NM
        sim.x_graf = sim.x_graf + x0
        """
        if central_layer <= 0:
            x0 = 0.0
        else:
            # In case the number of the layer exceeds the limit
            if central_layer > layers:
                central_layer = layers
            # Thickness of layers on the left side of the target interface
            x0 = -np.sum(sim.estrutura[0:central_layer])
        x0 = x0 / NM
        sim.x_graf = sim.x_graf + x0

        """
        Plots the structure using arrays that were created by the UpdateStructure. Doesn't erase
        the graph, in order to allow comparison between two structures
        """
        # sim = self.sim_list[self.simulation_cbox.currentIndex()]
        self.sim_subplot.plot(sim.x_graf, sim.v_graf)  # x in [nm] and y in [meV]
        self.sim_subplot.set_xlabel("Length (nm)")
        self.sim_subplot.set_ylabel("Energy (meV)")
        self.sim_subplot.set_xbound(sim.x_graf[0] - 0.2, sim.x_graf[-1] + 0.2)
        self.sim_subplot.set_ybound(np.min(sim.v_graf) - 50, np.max(sim.v_graf) + 100)
        self.sim_fig.tight_layout()
        self.sim_canvas.draw()

    def PlotSimResults(self, sim=None):
        """
        Plost the results from ResultadoWF.
        """
        # If this function is called without a specified sim, it gets the selected one from the
        # interface combobox
        if sim is None:
            try:  # Gets the current selected simulation
                sim = self.sim_list[self.simulation_cbox.currentIndex()]
            except:
                print("There is no simulation to choose from, create one first.")
                return

        # Gets the selected simulation
        # sim = self.sim_list[self.simulation_cbox.currentIndex()]
        self.sim_subplot.grid(True, axis="y")
        # Plots the probability density
        for result in sim.sim_ResultadoWF:
            self.sim_subplot.plot(result[0, :] / NM, result[3, :] * 1.0e3)
            # self.subplot_sim.plot(result[:, 0] / 1.0E9, result[:, 3] * 1.0E0)
        self.sim_canvas.draw()

    def ClearSimPlot(self):
        """
        Clears the simulation plot.
        """
        self.sim_fig.clf()
        self.sim_fig.tight_layout()
        self.sim_subplot = self.sim_fig.add_subplot(111)
        self.sim_canvas.draw()

    # Absorption
    def PlotAbsorption(self, sim=None):
        """
        Plots the results from the absorption of the selected simulation.
        """
        # If this function is called without a specified sim, it gets the selected one
        # from the interface combobox
        if sim is None:
            try:  # Gets the current selected simulation
                sim = self.sim_list[self.simulation_cbox.currentIndex()]
            except:
                print("There is no simulation to choose from, create one first.")
                return

        self.abs_subplot.plot(sim.abs_energy_axis, sim.abs_result)
        self.abs_subplot.set_xlabel("Energy (eV)")
        self.abs_subplot.set_ylabel("Absorption (u.a.)")
        self.abs_fig.tight_layout()
        self.abs_subplot.grid(True, axis="both")
        self.abs_canvas.draw()

    def ClearAbsPlot(self):
        """
        Clears the absorption plot.
        """
        self.abs_fig.clf()
        self.abs_fig.tight_layout()
        self.abs_subplot = self.abs_fig.add_subplot(111)
        self.abs_canvas.draw()

    # Transmission
    def PlotTransmission(self, sim=None):
        """
        Plots the transmission. The transmission is only calculated after simulation was
        run.
        """
        # If this function is called without a specified sim, it gets the selected one
        # from the interface combobox
        if sim is None:
            try:  # Gets the current selected simulation
                sim = self.sim_list[self.simulation_cbox.currentIndex()]
            except:
                print("There is no simulation to choose from, create one first.")
                return

        self.tra_subplot.plot(sim.sim_VecEnergy * 1.0e3, sim.sim_Transmission)
        self.tra_subplot.set_xlabel("Energy (eV)")
        self.tra_subplot.set_ylabel("Transmission (u.a.)")
        self.tra_fig.tight_layout()
        self.tra_subplot.grid(True, axis="both")
        # self.tra_subplot.set_yscale("log")
        self.tra_canvas.draw()

    def ClearTransPlot(self):
        """
        Clears the Transmission plot.
        """
        self.tra_fig.clf()
        self.tra_fig.tight_layout()
        self.tra_subplot = self.tra_fig.add_subplot(111)
        self.tra_canvas.draw()

    # Photocurrent
    def PlotPhotocurrent(self, sim=None):
        """
        Plots the photocurrent. The photocurrent is only calculated after simulation was
        run.
        """
        # If this function is called without a specified sim, it gets the selected one
        # from the interface combobox
        if sim is None:
            try:  # Gets the current selected simulation
                sim = self.sim_list[self.simulation_cbox.currentIndex()]
            except:
                print("There is no simulation to choose from, create one first.")
                return

        self.pc_subplot.plot(sim.sim_Photocurrent[0], sim.sim_Photocurrent[1])
        self.pc_subplot.set_xlabel("Energy (eV)")
        self.pc_subplot.set_ylabel("Photocurrent (u.a.)")
        self.pc_fig.tight_layout()
        self.pc_subplot.grid(True, axis="both")
        # self.pc_subplot.set_yscale("log")
        self.pc_canvas.draw()

    def ClearPhotocurrentPlot(self):
        """
        Clears the photocurrent plot.
        """
        self.pc_fig.clf()
        self.pc_fig.tight_layout()
        self.pc_subplot = self.pc_fig.add_subplot(111)
        self.pc_canvas.draw()

    # File inputs and outputs ##########################################################
    def ChooseOutputFolder(self):
        """
        Prompts the user to choose the simulation data output folder. Inside this folder, there will
        be a folder with the simulation title.
        """
        import conf
        cur = getattr(conf, "output_fortran_folder", "")
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Saída", cur)
        if folder:
            clean = os.path.normpath(folder).replace("\\", "/")
            if not clean.endswith("/"):
                clean += "/"
            conf.save_conf_paths(new_output_folder=clean)
            self.SyncOutputFolderDisplays(clean)
            if hasattr(self, "sim_list") and hasattr(self, "simulation_cbox") and self.sim_list:
                sim = self.sim_list[self.simulation_cbox.currentIndex()]
                sim.output_folder = clean
            self.UpdateFortranSamplesTable()
            self.LogMessage(f"📁 Pasta de saída alterada para: {clean}", "info")

    def SaveSimulation(self):
        """
        Saves the structure, simulation and absorption parameters.
        """
        sim = self.sim_list[self.simulation_cbox.currentIndex()]
        try:
            save_dir = os.path.join(sim.output_folder, f"{sim.title}.qwsim")
            # print(f'save_dir: {save_dir}')
            file, _ = QFileDialog.getSaveFileName(
                self,
                caption="Save structure and simulation parameters",
                directory=save_dir,
                filter=self.tr("*.qwsim"),
            )

            output_file = open(file, "wb")
            pickle.dump(sim, output_file)
            output_file.close()
        except:
            print("Couldn't save the structure")
            # If the user doesn't choose a filename, closing the interface, do nothing
            return

    def LoadSimulation(self):
        """
        Loads structure and simulation data from a file
        """
        try:  # Returns a tuple
            file, _ = QFileDialog.getOpenFileName(
                self, "Loads the structure and simulation data", self.tr("*.qwsim")
            )
            input_file = open(file, "rb")
            sim = pickle.load(input_file)

            self.sim_list.append(sim)
            self.simulation_cbox.setCurrentIndex(len(self.sim_list) - 1)
            self.UpdateStructureTable()
            self.UpdateSimList()
            self.UpdateLayerCount()
            # If an structure was saved, plot it
            try:  # The try-except is to avoid errors in case sim.structure doesn't exist
                if len(sim.estrutura) > 0:
                    self.PlotStructure(sim)
            except:
                pass
            # If a simulation result was saved, plot it
            try:
                if sim.sim_ran:
                    self.PlotSimResults(sim)
            except:
                pass
            # If an absorption result was saved, plot it
            try:
                if sim.abs_ran:
                    self.PlotAbsorption(sim)
            except:
                pass
            self.UpdateInterface()

        except:
            # If an error happens upon opening the file, the function just returns False.
            print("Couldn't load the structure")
            return

    def CreateOutputFolder(self, sim):
        """
        Checks whether the selected output folder exists or must be created
        """
        # Defines the output folder based on the simulation title
        # output_folder = str(self.output_folder_line.text())
        output_folder = sim.output_folder
        # If the folder was not chosen, just get the file execution path
        if not output_folder:
            output_folder = os.path.dirname(os.path.abspath(__file__))
        # If the menu checkbox is checked (it is, by default) create a folder with the same name as
        # the simulation
        if self.adv_new_folder_chkbx.isChecked:
            output_folder = os.path.join(output_folder, sim.title)

        return output_folder

    def SaveSimOutput(self, sim):
        """
        Based on the interface's checkboxes, saves the simulation output
        """
        output_folder = self.CreateOutputFolder(sim)

        # To avoid repetition of the "all checkbox"
        save_all = self.sim_files_all_chkbx.isChecked()

        # After the end of the calculations, save the result in text files if desired
        if (
            save_all
            or self.sim_files_effm_chkbx.isChecked()
            or self.sim_files_energies_chkbx.isChecked()
            or self.sim_files_npe_chkbx.isChecked()
            or self.sim_files_pot_chkbx.isChecked()
            or self.sim_files_wf_chkbx.isChecked()
            or self.sim_files_x_chkbx.isChecked()
        ):
            data_folder = os.path.join(output_folder, "Data")

            # if the folder to save the data doesn't exist, create it
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)

        if save_all or self.sim_files_x_chkbx.isChecked():
            np.savetxt(os.path.join(data_folder, "X Axis.txt"), sim.sim_x, newline="\n")

        if save_all or self.sim_files_pot_chkbx.isChecked():
            np.savetxt(
                os.path.join(data_folder, "Electrical Potential.txt"),
                sim.sim_pot,
                newline="\n",
            )

        if save_all or self.sim_files_effm_chkbx.isChecked():
            np.savetxt(
                os.path.join(data_folder, "Effective Mass.txt"),
                sim.sim_effm_cte,
                newline="\n",
            )

        if save_all or self.sim_files_npe_chkbx.isChecked():
            np.savetxt(
                os.path.join(data_folder, "Non-Parabolicity.txt"),
                sim.sim_npe,
                newline="\n",
            )

        if save_all or self.sim_files_energies_chkbx.isChecked():
            np.savetxt(os.path.join(data_folder, "Autoenergias.txt"), sim.sim_Energias)

        if save_all or self.sim_files_wf_chkbx.isChecked():
            PastaWave = os.path.join(data_folder, "Wave Functions")
            if not os.path.exists(PastaWave):
                os.makedirs(PastaWave)
            for a in range(len(sim.sim_Energias)):
                fname = f"WF_E{a}_{sim.sim_Energias[a]:02.6f}.txt"
                # The transposed is print, so that the WF is in a column, not row
                np.savetxt(os.path.join(PastaWave, fname), sim.sim_ResultadoWF[a].T)

    def SaveTransmissionOutput(self, sim):
        """
        Based on the interface's checkboxes, saves the Transmission output
        """
        output_folder = self.CreateOutputFolder(sim)

        # After the end of the calculations, save the result in text files if desired
        if (
            self.sim_files_all_chkbx.isChecked()
            or self.sim_files_trans_chkbx.isChecked()
        ):
            data_folder = os.path.join(output_folder, "Data")

            # if the folder to save the data doesn't exist, create it
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)

            output = np.column_stack((sim.sim_VecEnergy, sim.sim_Transmission))
            output_file = os.path.join(data_folder, "Transmission Spectrum.txt")
            np.savetxt(output_file, output)

    def SavePhotocurrentOutput(self, sim):
        """
        Based on the interface's checkboxes, saves the Transmission output
        """
        output_folder = self.CreateOutputFolder(sim)

        # After the end of the calculations, save the result in text files if desired
        if self.sim_files_all_chkbx.isChecked() or self.sim_files_pc_chkbox.isChecked():
            data_folder = os.path.join(output_folder, "Data")

            # if the folder to save the data doesn't exist, create it
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)

            output = sim.sim_Photocurrent
            output_file = os.path.join(data_folder, "Photocurrent Spectrum.txt")
            np.savetxt(output_file, output)

    def SaveGUIConfigFile(self):
        """
        Saves all the values and settings on the interface to the "interface.cfg" file.
        This is meant to be a easy way of saving the default settings, so that the user doesn't need
        to change the values every time the software is opened, nor needs to manually edit the
        config file.
        The values must be in string format.
        """
        # Reads the configuration defaults from the "interface.cfg" file:
        cfg = configparser.ConfigParser()
        file = os.path.join(self.base_path, "interface.cfg")
        cfg.read(file)

        # Structure tab
        # Saves only the values of nm or ml, depending on the checkbox state
        if self.adv_nm_layers_chkbx.checkState():
            cfg["stru"]["barrier_nm"] = f"{self.barrier_nm_spb.value()}"
            cfg["stru"]["well_nm"] = f"{self.well_nm_spb.value()}"
        else:
            cfg["stru"]["barrier_ml"] = f"{self.barrier_ml_spb.value()}"
            cfg["stru"]["well_ml"] = f"{self.well_ml_spb.value()}"

        # Simulation tab
        cfg["simu"]["E0"] = f"{self.sim_E0_spb.value()}"
        cfg["simu"]["dE"] = f"{self.sim_dE_spb.value()}"
        cfg["simu"]["Ef"] = f"{self.sim_Ef_spb.value()}"
        if self.adv_nm_layers_chkbx.checkState():
            cfg["simu"]["dx_nm"] = f"{self.sim_dx_nm_spb.value()}"
        else:
            cfg["simu"]["dx_ml"] = f"{self.sim_dx_ml_spb.value()}"
        cfg["simu"]["Efield"] = f"{self.sim_Efield_spb.value()}"
        cfg["simu"]["central_layer"] = f"{self.sim_central_layer_spb.value()}"
        # Checkboxes
        cfg["simu"]["output_all"] = (
            "True" if self.sim_files_all_chkbx.checkState() else "False"
        )
        cfg["simu"]["output_effm"] = (
            "True" if self.sim_files_effm_chkbx.checkState() else "False"
        )
        cfg["simu"]["output_energies"] = (
            "True" if self.sim_files_energies_chkbx.checkState() else "False"
        )
        cfg["simu"]["output_npe"] = (
            "True" if self.sim_files_npe_chkbx.checkState() else "False"
        )
        cfg["simu"]["output_pot"] = (
            "True" if self.sim_files_pot_chkbx.checkState() else "False"
        )
        cfg["simu"]["output_trans"] = (
            "True" if self.sim_files_trans_chkbx.checkState() else "False"
        )
        cfg["simu"]["output_wf"] = (
            "True" if self.sim_files_wf_chkbx.checkState() else "False"
        )
        cfg["simu"]["output_x"] = (
            "True" if self.sim_files_x_chkbx.checkState() else "False"
        )

        # Absorption tab
        cfg["abso"]["initial_WF"] = f"{self.abs_init_wf_spb.value()}"
        cfg["abso"]["lorz_broad"] = f"{self.abs_broadening_spb.value()}"
        cfg["abso"]["E0"] = f"{self.abs_E0_spb.value()}"
        cfg["abso"]["dE"] = f"{self.abs_dE_spb.value()}"
        cfg["abso"]["Ef"] = f"{self.abs_Ef_spb.value()}"

        # Transmission tab

        # Photocurrent tab
        cfg["phot"]["E0"] = f"{self.pc_E0_spb.value()}"
        cfg["phot"]["dE"] = f"{self.pc_dE_spb.value()}"
        cfg["phot"]["Ef"] = f"{self.pc_Ef_spb.value()}"

        # Genetic Algorithm tab
        cfg["gene"]["iterations"] = f"{self.ga_iter_spb.value():d}"
        cfg["gene"]["population"] = f"{self.ga_pop_spb.value():d}"
        cfg["gene"]["target_E"] = f"{self.ga_tgt_en_spb.value()}"
        cfg["gene"]["target_E_margin"] = f"{self.ga_tgt_en_margin_spb.value()}"
        cfg["gene"]["opt_pc"] = "True" if (hasattr(self, "ga_opt_pc_chkbx") and self.ga_opt_pc_chkbx.isChecked()) else "False"
        cfg["gene"]["opt_os"] = "True" if (hasattr(self, "ga_opt_os_chkbx") and self.ga_opt_os_chkbx.isChecked()) else "False"
        cfg["gene"]["limit_energy"] = "True" if (hasattr(self, "ga_limit_energy_chkbx") and self.ga_limit_energy_chkbx.isChecked()) else "False"

        # Automation tab
        if hasattr(self, "auto_layer_spb"):
            cfg["auto"]["target_layer"] = f"{self.auto_layer_spb.value():d}"
            cfg["auto"]["thickness_initial"] = f"{self.auto_init_spb.value()}"
            cfg["auto"]["thickness_step"] = f"{self.auto_step_spb.value()}"
            cfg["auto"]["thickness_final"] = f"{self.auto_final_spb.value()}"

        # Data tab

        # Advanced tab
        cfg["adva"]["create_new_folder"] = (
            "True" if self.adv_new_folder_chkbx.checkState() else "False"
        )
        cfg["adva"]["use_nanometers"] = (
            "True" if self.adv_nm_layers_chkbx.checkState() else "False"
        )
        # cfg["adva"]["autorun_abs"] = (
        #     "True" if self.adv_autorun_abs_chkbx.checkState() else "False"
        # )
        # cfg["adva"]["autorun_abs_initial_WF"] = f"{self.adv_wf0_spb.value()}"
        # cfg["adva"]["autorun_trans"] = (
        #     "True" if self.adv_autorun_trans_chkbx.checkState() else "False"
        # )
        cfg["adva"]["interface_to_split"] = f"{self.adv_split_layer_spb.value()}"
        cfg["adva"]["method"] = f"{self.adv_method_cbox.currentIndex()}"

        # Saves the configuration file
        with open(file, "w") as configfile:
            cfg.write(configfile)

    # Run the calculations #############################################################
    # Absorption
    def RunAbsorption(self, sim=None):
        """
        Calculates the structure's absorption spectra, the dipole moment, oscilator strenght and
        delta energy between the absorption peaks and the reference wavefunction (wf_0).
        """
        # If this function is called without a specified sim, it gets the selected one from the
        # interface combobox
        if sim is None:
            try:  # Gets the current selected simulation
                sim = self.sim_list[self.simulation_cbox.currentIndex()]
            except:
                print("There is no simulation to choose from, create one first.")
                return
        # Checks whether there are at least two wavefunctions, in order to calculate the absorption.
        if len(sim.sim_Energias) < 2:
            print("Cannot calculate absorption if there are less than 2 wavefunctions.")
            return
        # Collecting the relevant data from the GUI
        wf_0_index = int(self.abs_init_wf_spb.value())
        E0 = self.abs_E0_spb.value()
        Ef = self.abs_Ef_spb.value()
        dE = self.abs_dE_spb.value()
        # Linewidth broadening of the lorentzian. From "Van Hove singularities in intersubband
        # transitions in multiquantum well photodetectors" doi.org/10.1016/j.infrared.2006.10.016
        broadening = self.abs_broadening_spb.value()

        # Gets the choosen simulation
        # sim = self.sim_list[self.simulation_cbox.currentIndex()]
        sim.CalcAbs(wf_0_index, E0, Ef, dE, broadening)

        # Saving the absorption output to files
        output_folder = (
            sim.output_folder
        )  # As defined when the sim was created or "change folder"
        # Defines the output folder based on the simulation title
        if self.adv_new_folder_chkbx.isChecked:
            output_folder = os.path.join(output_folder, sim.title)
        # Saves in a folder called "Absorption"
        output_folder = os.path.join(output_folder, "Absorption")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        try:
            data = np.column_stack(
                (sim.abs_delta_E, sim.abs_dipole, sim.abs_osc_strength)
            )
            filename = f"DeltaE_DipoloEletrico_ForcadeOscilador_E{wf_0_index:01d}.txt"
            np.savetxt(os.path.join(output_folder, filename), data)
        except:
            print("Could not save absorption results")

        try:
            data = np.column_stack((sim.abs_energy_axis, sim.abs_result))
            filename = f"Absorcao_E{wf_0_index:01d}.txt"
            np.savetxt(os.path.join(output_folder, filename), data)
        except:
            print("Could not save absorption results")

        # Plot the absorption and update the interface
        self.PlotAbsorption(sim)
        self.UpdateInterface()

    def RunAutomation(self):
        """
        Creates an automation changing the thickness of the target layer by the values
        in the range defined by the user in the GUI
        """
        if not getattr(self, "sim_list", None) or self.simulation_cbox.currentIndex() < 0:
            return
        if not hasattr(self, "auto_init_spb"):
            return
        # Gets the base simulation, which will be modified by the automated steps
        base_sim = self.sim_list[self.simulation_cbox.currentIndex()]

        # Obtaining the target range of thicknesses
        th_init = self.auto_init_spb.value()
        th_step = self.auto_step_spb.value()
        th_final = self.auto_final_spb.value()
        # List of thicknesses
        th_list = np.arange(th_init, th_final + th_step, th_step, dtype=float)

        # Defining whether the thickness is in nanometers or monolayers
        if (
            self.adv_nm_layers_chkbx.checkState()
        ):  # If user wants nanometers, use value as is
            th_unit = "nm"
        else:  # else, calculate dx as a multiple of monolayer = latpar/2
            # Correcting the units, if the value was entered in monolayers
            th_unit = "ml"
            th_list = th_list * base_sim.latpar / 2

        # Gets the index of the layer that must be modified
        tgt_layer = self.auto_layer_spb.value()
        # The target layer must exist. In case the structure doesn't have the defined layer, correct
        if tgt_layer < 0:
            tgt_layer = 0
        elif tgt_layer > len(base_sim.estrutura) - 1:
            tgt_layer = len(base_sim.estrutura) - 1

        slist = []
        # For each thickness
        for th in th_list:
            # Creates a copy of the original simulation
            new_sim = deepcopy(base_sim)
            # Puts it into the simulations list
            self.sim_list.append(new_sim)
            # Puts it into a list just for the multiprocessing
            slist.append(new_sim)
            # Changes the title based on the title of the base simulation and the modification
            new_sim.title = base_sim.title + f" {th:.3f}{th_unit}"
            # Changes the thickness of the target layer
            if new_sim.feature[tgt_layer] == "Well":
                new_sim.ReplaceWell(th, tgt_layer)
            else:
                new_sim.ReplaceBarrier(th, tgt_layer)
            self.RunSimulation(new_sim)

        # cores = cpu_count()
        # p = Pool(processes=cores)
        # p = Pool(processes=1)
        # p.map(self.RunASim, slist)
        # with Pool(processes=cores) as p:
        #     for i, _ in enumerate(p.imap_unordered(self.RunASim, slist)):
        #         print(i)

        self.UpdateStructureTable()
        self.UpdateSimList()

    # Simulation
    def RunSimulation(self, sim=None):
        """
        This function is a copy of "Run" but it takes the simulation as an argument
        instead of getting the selected simulation from the combobox.
        """
        # If this function is called without a specified sim, it gets the selected one
        # from the interface combobox
        if sim is None:
            try:  # Gets the current selected simulation
                sim = self.sim_list[self.simulation_cbox.currentIndex()]
            except:
                print("There is no simulation to choose from, create one first.")
                return

        # Obtaining the target range of energies
        E0 = self.sim_E0_spb.value()
        Ef = self.sim_Ef_spb.value()
        dE = self.sim_dE_spb.value()

        # Reads the value of dx from the interface
        if self.adv_nm_layers_chkbx.checkState():
            # If user wants nanometers, use value as is
            dx = self.sim_dx_nm_spb.value()
            dx_unit = "nm"
        else:  # else, calculate dx as a multiple of monolayer = latpar/2
            dx = self.sim_dx_ml_spb.value()
            dx_unit = "ml"

        # From the interface, defines the interface of the wavefunction split
        split_i = self.adv_split_layer_spb.value()

        # From the structure tab, gets the index of the central layer
        central_layer = self.sim_central_layer_spb.value()

        # The method chosen by the user to perform the calculations
        # Methods avaliable:
        # 0 - "Numerov - For"
        # 1 - "Numerov - Split"
        # 2 - "Numerov - Arrays"
        # 3 - "TMM"
        # 4 - "TMM - Split"
        wf_method = self.adv_method_cbox.currentIndex()

        # Timing
        t_start_run = time.time()

        # Runs the calculations with information gathered from the interface and stored
        # at a "sim".
        # Sim is the selected simulation, containing all relevant data
        sim.RunSim(wf_method, split_i, E0, Ef, dE, dx, dx_unit, central_layer)
        # Timing
        print(f"Total time: {time.time() - t_start_run:.3f} s")

        self.SaveSimOutput(sim)
        self.PlotSimResults(sim)
        self.UpdateInterface()
        self.UpdateDataTable()
        # If the user wants, automatically calculate the absorption, transmission or
        # photocurrent after the main simulation
        if self.sim_autorun_abs_chkbx.checkState():
            self.RunAbsorption(sim)
        if self.sim_autorun_tra_chkbx.checkState():
            self.RunTransmission(sim)
        if self.sim_autorun_pc_chkbx.checkState():
            self.RunPhotocurrent(sim)

    # Transmission
    def RunTransmission(self, sim=None):
        """
        Performs the calculation of the transmission and displays it on the graph.
        """
        # If this function is called without a specified sim, it gets the selected one
        # from the interface combobox
        if sim is None:
            try:  # Gets the current selected simulation
                sim = self.sim_list[self.simulation_cbox.currentIndex()]
            except:
                print("There is no simulation to choose from, create one first.")
                return

        sim.Transmission()

        self.PlotTransmission(sim)
        self.SaveTransmissionOutput(sim)

    def RunPhotocurrent(self, sim=None):
        """
        Performs the calculation of the Photocurrent and plots on the graph.
        """
        # If this function is called without a specified sim, it gets the selected one
        # from the interface combobox
        if sim is None:
            try:  # Gets the current selected simulation
                sim = self.sim_list[self.simulation_cbox.currentIndex()]
            except:
                print("There is no simulation to choose from, create one first.")
                return

        # Reads the value of dx from the interface
        if self.adv_nm_layers_chkbx.checkState():
            # If user wants nanometers, use value as is
            dx = self.sim_dx_nm_spb.value()
            dx_unit = "nm"
        else:  # else, calculate dx as a multiple of monolayer = latpar/2
            dx = self.sim_dx_ml_spb.value()
            dx_unit = "ml"

        # Obtaining the target range of energies and converting to eV
        E0 = self.pc_E0_spb.value() * 1.0e-3
        Ef = self.pc_Ef_spb.value() * 1.0e-3
        dE = self.pc_dE_spb.value() * 1.0e-3

        sim.RunPhotocurrent(dx, E0, Ef, dE)

        self.PlotPhotocurrent(sim)
        self.SavePhotocurrentOutput(sim)

    # Simulation instance functions
    def CreateSimulation(self, title, materials):
        """
        Function that creates a new simulation with the relevant information
        """
        # Creating the arrays necessary to the simulation
        array_data = dict()
        array_data["estrutura"] = np.array([], dtype=np.float64)
        array_data["massa_eff_const"] = np.array([], dtype=np.float64)
        array_data["pot"] = np.array([], dtype=np.float64)
        array_data["E_nonparab"] = np.array([], dtype=np.float64)
        # Loads the available materias from the file
        self.mat = configparser.ConfigParser()
        self.mat.read(os.path.join(self.base_path, "materials.data"))
        # Defining material properties from the materials.data file
        material_data = dict()
        material_data["latpar"] = self.mat[materials].getfloat("latpar")
        material_data["barrier"] = self.mat[materials]["barrier"]
        material_data["m_eff_ct_barrier"] = self.mat[materials].getfloat(
            "m_eff_ct_barrier"
        )
        material_data["e_nonparab_barrier"] = self.mat[materials].getfloat(
            "e_nonparab_barrier"
        )
        material_data["pot_barrier"] = self.mat[materials].getfloat("pot_barrier")
        material_data["well"] = self.mat[materials]["well"]
        material_data["m_eff_ct_well"] = self.mat[materials].getfloat("m_eff_ct_well")
        material_data["pot_well"] = self.mat[materials].getfloat("pot_well")
        material_data["e_nonparab_well"] = self.mat[materials].getfloat(
            "e_nonparab_well"
        )

        # Creates the simulation and puts it into a list of simulations
        self.sim_list.append(simdata.SimData(title, array_data, material_data))
        self.current_number += 1
        self.new_sim_window.close()
        self.CreatedNewSimulation()

    def DeleteSimulation(self):
        """
        Deletes the simulation selected in the combobox.
        """
        # Gets the index of the selected simulation
        current_index = self.simulation_cbox.currentIndex()
        list_len = len(self.sim_list)

        # Removes it from the simulations list, if not empty
        if list_len == 0:
            return
        self.sim_list.pop(current_index)

        # if it was the last item in the combobox, select the previous
        if current_index == list_len - 1:
            self.simulation_cbox.setCurrentIndex(current_index - 1)
        # else if just selects the next item
        else:
            self.simulation_cbox.setCurrentIndex(current_index)

        # Calls the function that updates the interface with the simulation selected from the
        # combobox
        # Updates the combobox to reflect the change made to the simulation list
        self.UpdateInterface()
        self.UpdateSimList()
        self.UpdateLayerCount()
        # self.ChangedSimulation()

    def RenameSimulation(self):
        # Gets the current selected simulation
        try:
            sim = self.sim_list[self.simulation_cbox.currentIndex()]
        except:
            print("There is no simulation to choose from, create one first.")
            return
        # Creates a new window pasing the selected simulation
        self.rename_window = RenameSimWindow(sim)
        # Shows the window
        self.rename_window.show()
        # Connects the signal to the function used to close the window
        self.rename_window.signal_renamed.connect(self.CloseRenameWindow)

    def CopySimulation(self):
        """
        Copies the current simulation creating a new instance with the same attributes and opens the
        title window so that the user can change the title.
        """
        # Gets the current selected simulation
        try:
            current_sim = self.sim_list[self.simulation_cbox.currentIndex()]
        except:
            print("There is no simulation to choose from, create one first.")
            return

        # Copies the current simulation and adds it to the list
        new_sim = deepcopy(current_sim)
        self.sim_list.append(new_sim)

        # Creates a new window
        self.rename_window = RenameSimWindow(new_sim)
        # Shows the window
        self.rename_window.show()
        # Connects the signal to the function used to close the window
        # This window will be closed by another function, which is called when the user
        # confirms the name of the new simulation. This is done by a signal emitted from
        # the TitleWindow.
        self.rename_window.signal_renamed.connect(self.CloseRenameWindow)

    # Functions related to other windows ###############################################
    def CloseRenameWindow(self):
        """
        Closes the window that was opened to change the simulation name.
        """
        self.UpdateSimList()
        self.rename_window.close()

    def OpenNewSimWindow(self):
        """
        Creates and opens the window that will create the title for the new simulation.
        """
        self.new_sim_window = NewSimWindow(self.current_number)
        # Shows the window
        self.new_sim_window.show()
        # Connects the signal to the function used to close the window
        self.new_sim_window.signal_updated_current_number.connect(
            self.UpdateCurrentNumber
        )
        # This window will be closed by another function, which is called when the user
        # confirms the name of the new simulation. This is done by a signal emitted from
        # the TitleWindow.
        self.new_sim_window.signal_new_title.connect(self.CreateSimulation)

    def UpdateCurrentNumber(self, cnum):
        """
        Updates the current number, in order to keep track of how many simulations were run.
        """
        self.current_number = int(cnum)

    def Sobre(self):
        """
        Opens the "About" window
        """
        # chamando a nova classe SobreWindow que cria uma nova janela
        # se nao colocar o self, garbage collection will remove that object as soon as setupUi
        # method finishes.
        self.Sobre = SobreWindow()
        # mostrando na tela a classe criada para a segunda janela
        self.Sobre.show()

    def PlotFortranBandStructure(self):
        """
        Plots 3 figures for Fortran simulation tab:
        1. Potencial with wavefunctions (from simulation data if available, or structural profile).
        2. Oscillator Strength (OscStr).
        3. Photocurrent (PC).
        """
        struct = [
            int(self.fortran_s0_spb.value()),
            round(self.fortran_s1_spb.value() * 10, 1),
            round(self.fortran_s2_spb.value() * 10, 1),
            round(self.fortran_s3_spb.value() * 10, 1),
            int(self.fortran_s4_spb.value()),
            round(self.fortran_s5_spb.value() * 10, 1),
            round(self.fortran_s6_spb.value() * 10, 1),
        ]

        import conf
        output_fortran_folder = getattr(conf, "output_fortran_folder", "")
        sim_options = SimulationOptions(
            force_parser=False,
            force_simulation=False,
            reference_os=None,
            reference_os_e=None,
            reference_pc=None,
            reference_pc_e=None
        )
        simulator = FortranSimulator(struct, sim_options=sim_options, output_folder=output_fortran_folder)

        # Clear subplots
        self.fortran_subplot1.clear()
        self.fortran_subplot2.clear()
        self.fortran_subplot3.clear()

        # Check if simulation exists (all output text files exist)
        sim_exists = (
            os.path.exists(simulator.potencial_file) and
            os.path.exists(simulator.wavefunction_file) and
            os.path.exists(simulator.oscstr_file) and
            os.path.exists(simulator.photocurrent_file)
        )

        if sim_exists:
            try:
                sample_data = simulator.simulate()

                # Plot 1: Potencial + Wavefunctions
                Plotter.plot_structure(
                    self.fortran_subplot1,
                    sample_data.x_potencial,
                    sample_data.y_potencial,
                    sample_data.autoenergias,
                    sample_data.wavefunctions,
                    sample_data.max_e_oscstr_index,
                    -50, 700
                )
                self.fortran_subplot1.set_title("Potencial e Funções de Onda")

                # Plot 2: Oscillator Strength
                Plotter.plot_osc(
                    self.fortran_subplot2,
                    oscstr=sample_data.oscstr,
                    oscstr_e=sample_data.oscstr_e,
                    E0=sample_data.E0,
                    lim_E_min=-50,
                    lim_E_max=700,
                    max_e_oscstr_index=sample_data.max_e_oscstr_index
                )
                self.fortran_subplot2.set_title("Força de Oscilador")

                # Plot 3: Photocurrent
                Plotter.plot_pc(
                    self.fortran_subplot3,
                    pc=sample_data.pc,
                    pc_e=sample_data.pc_e,
                    E0=sample_data.E0,
                    lim_E_min=-50,
                    lim_E_max=700,
                    max_abs_photocurrent=sample_data.max_abs_photocurrent,
                    max_e_abs_photocurrent=sample_data.max_e_abs_photocurrent,
                    min_abs_photocurrent=sample_data.min_abs_photocurrent,
                    min_e_abs_photocurrent=sample_data.min_e_abs_photocurrent
                )
                self.fortran_subplot3.set_title("Photocurrent")

                # Display summary text in results text box
                _structure = sample_data.structure
                _pc = sample_data.max_abs_photocurrent
                _pc_e = sample_data.max_e_abs_photocurrent
                _os = sample_data.max_e_oscstr
                _os_e = sample_data.max_e_transition
                output_lines = [
                    "=" * 50,
                    f"Amostra simulada encontrada: {sample_data.sample_id}",
                    f"Fitness: {sample_data.fitness:.4f}",
                    f"Estructura: {_structure}",
                    f"PC: {_pc:.4e}, PC_e: {_pc_e:.1f} meV",
                    f"OS: {_os:.4f}, OS_e: {_os_e:.1f} meV",
                    "=" * 50,
                ]
                self.fortran_output_txt.setText("\n".join(output_lines))

            except Exception as e:
                self.fortran_output_txt.setText(f"Erro ao carregar dados da simulação: {str(e)}")
                self._plot_fallback_band_profile(struct)
        else:
            self._plot_fallback_band_profile(struct)
            self.fortran_output_txt.setText("Amostra ainda não simulada. Clique em 'Run Fortran Simulation' para ejecutar.")

        self.fortran_fig1.tight_layout()
        self.fortran_fig2.tight_layout()
        self.fortran_fig3.tight_layout()
        self.fortran_canvas1.draw()
        self.fortran_canvas2.draw()
        self.fortran_canvas3.draw()

    def _plot_fallback_band_profile(self, struct):
        """Fallback method to plot simple conduction band profile when simulation files do not exist yet."""
        w_l = struct[0]
        qw_l = struct[1] / 10.0
        qb_l = struct[2] / 10.0
        qw_c = struct[3] / 10.0
        w_r = struct[4]
        qw_r = struct[5] / 10.0
        qb_r = struct[6] / 10.0

        barrier_energy = 500.0
        well_energy = 0.0

        thicknesses = [50.0]
        energies = [barrier_energy]

        for _ in range(w_l):
            thicknesses.extend([qw_l, qb_l])
            energies.extend([well_energy, barrier_energy])

        thicknesses.append(qw_c)
        energies.append(well_energy)

        for _ in range(w_r):
            thicknesses.extend([qb_r, qw_r])
            energies.extend([barrier_energy, well_energy])

        thicknesses.append(50.0)
        energies.append(barrier_energy)

        left_total_thickness = 50.0 + w_l * (qw_l + qb_l)
        current_x = -left_total_thickness

        x_pts, y_pts = [], []
        for th, en in zip(thicknesses, energies):
            x_pts.extend([current_x, current_x + th])
            y_pts.extend([en, en])
            current_x += th

        self.fortran_subplot1.plot(x_pts, y_pts, color="#1e3a8a", linewidth=1.4)
        self.fortran_subplot1.fill_between(x_pts, y_pts, color="#dbeafe", alpha=0.45)
        self.fortran_subplot1.set_xlabel("Thickness (nm)")
        self.fortran_subplot1.set_ylabel("Energy (meV)")
        self.fortran_subplot1.set_title("Potencial e Funções de Onda (Perfil)", fontsize=9, fontweight='bold', color='#1e293b')
        self.fortran_subplot1.grid(True, linestyle=":", alpha=0.55)
        self.fortran_subplot1.set_ylim(-50, 700)

        self.fortran_subplot2.set_xlabel("Oscillator Strength", color="#be123c", fontweight='bold')
        self.fortran_subplot2.set_ylabel("ΔE (meV)")
        self.fortran_subplot2.set_title("Força de Oscilador", fontsize=9, fontweight='bold', color='#1e293b')
        self.fortran_subplot2.grid(True, linestyle=":", alpha=0.55)
        self.fortran_subplot2.set_ylim(-50, 700)

        self.fortran_subplot3.set_xlabel("Photocurrent (a.u)", color="#0369a1", fontweight='bold')
        self.fortran_subplot3.set_ylabel("ΔE (meV)")
        self.fortran_subplot3.set_title("Photocurrent", fontsize=9, fontweight='bold', color='#1e293b')
        self.fortran_subplot3.grid(True, linestyle=":", alpha=0.55)
        self.fortran_subplot3.set_ylim(-50, 700)

    def RunFortranSimulation(self):
        """
        Executes Fortran simulation asynchronously in a background QThread.
        Does not freeze the GUI and updates the real-time status and log bar.
        """
        if self.fortran_worker and self.fortran_worker.isRunning():
            QMessageBox.warning(self, "Aviso", "Já existe uma simulação Fortran em execução. Aguarde a conclusão.")
            return

        try:
            struct = [
                int(self.fortran_s0_spb.value()),
                round(self.fortran_s1_spb.value() * 10, 1),
                round(self.fortran_s2_spb.value() * 10, 1),
                round(self.fortran_s3_spb.value() * 10, 1),
                int(self.fortran_s4_spb.value()),
                round(self.fortran_s5_spb.value() * 10, 1),
                round(self.fortran_s6_spb.value() * 10, 1),
            ]

            force_parser = self.fortran_force_parser_chkbx.isChecked()
            force_sim = self.fortran_force_sim_chkbx.isChecked()
            sim_options = SimulationOptions(
                force_parser=force_parser,
                force_simulation=force_sim,
                reference_os=None,
                reference_os_e=None,
                reference_pc=None,
                reference_pc_e=None,
            )

            w_l, qw_l, qb_l, mqw, w_r, qw_r, qb_r = set_structure_values(struct)
            desc = f"{int(w_l)}x{qw_l:.1f}_{qb_l:.1f} __{mqw:.1f}__ {int(w_r)}x{qw_r:.1f}_{qb_r:.1f}"

            self.fortran_run_btn.setEnabled(False)
            self.fortran_run_btn.setText("⏳ Simulando...")
            self.SetSimulatorState("running", "Fortran Sim")
            self.LogMessage(f"⏳ Simulação Fortran iniciada: {desc}... Aguarde, executando Fortran...", "running")
            self.fortran_output_txt.setText(f"Iniciando simulação Fortran...\nEstrutura: {desc}\nStatus: Processando no Fortran em segundo plano...")

            import conf
            output_fortran_folder = getattr(conf, "output_fortran_folder", "")
            self.fortran_worker = FortranSimWorker(struct, sim_options, output_fortran_folder)
            self.fortran_worker.sig_finished.connect(self.OnFortranSimFinished)
            self.fortran_worker.start()
        except Exception as e:
            self.SetSimulatorState("idle")
            self.fortran_run_btn.setEnabled(True)
            self.fortran_run_btn.setText("Run Fortran")
            err_msg = f"Erro ao iniciar simulação Fortran: {str(e)}"
            self.LogMessage(err_msg, "error")
            self.fortran_output_txt.setText(err_msg)

    def OnFortranSimFinished(self, success: bool, sample_data, error_msg: str, elapsed: float):
        """Slot called when the background FortranSimWorker finishes."""
        self.fortran_run_btn.setEnabled(True)
        self.fortran_run_btn.setText("Run Fortran")
        self.SetSimulatorState("idle")

        if success and sample_data:
            pc = float(sample_data.max_abs_photocurrent)
            pc_e = float(sample_data.max_e_abs_photocurrent)
            os_val = float(sample_data.max_e_oscstr)
            os_e = float(sample_data.max_e_transition)

            output_lines = [
                "=" * 50,
                f"Simulação completada com sucesso em {elapsed:.1f}s.",
                f"Amostra ID: {sample_data.sample_id}",
                f"Estrutura: {sample_data.structure}",
                f"PC Máx: {pc:.4e} (Pico em {pc_e:.1f} meV)",
                f"OS Máx: {os_val:.4f} (Transição em {os_e:.1f} meV)",
                "=" * 50,
            ]
            text_output = "\n".join(output_lines)
            self.fortran_output_txt.setText(text_output)
            self.LogMessage(
                f"✅ Simulação concluída com sucesso em {elapsed:.1f}s! (PC Máx: {pc:.2e}, OS Máx: {os_val:.4f})",
                "success"
            )
            self.PlotFortranBandStructure()
            self.UpdateFortranSamplesTable()
        else:
            err = f"Erro na simulação Fortran ({elapsed:.1f}s): {error_msg}"
            self.fortran_output_txt.setText(err)
            self.LogMessage(f"❌ {err}", "error")
            QMessageBox.critical(self, "Erro na Simulação", f"Ocorreu um erro ao executar a simulação Fortran:\n\n{error_msg}")

    def InitializeFortranSamplesTable(self):
        """
        Initializes the columns for simulated Fortran samples table and configures the resizable QSplitters.
        Columns: s0, s1, s2, s3, s4, s5, s6, PC Max, PC Energy (meV), OS Max, OS Energy (meV)
        """
        if hasattr(self, "fortran_splitter"):
            # Set relative stretch factors for main panels: Controls (1), Plots (3), Table (3)
            self.fortran_splitter.setStretchFactor(0, 1)
            self.fortran_splitter.setStretchFactor(1, 3)
            self.fortran_splitter.setStretchFactor(2, 3)

        if hasattr(self, "fortran_plots_splitter"):
            # Set 2:1:1 ratio (2/4, 1/4, 1/4) for the 3 plot panels
            self.fortran_plots_splitter.setStretchFactor(0, 2)
            self.fortran_plots_splitter.setStretchFactor(1, 1)
            self.fortran_plots_splitter.setStretchFactor(2, 1)

        if hasattr(self, "fortran_table_vsplitter"):
            self.fortran_table_vsplitter.setStretchFactor(0, 3)
            self.fortran_table_vsplitter.setStretchFactor(1, 2)
            self.fortran_table_vsplitter.setSizes([260, 180])

        self.fortran_column_names = [
            "LLQW\n(Nº Left)",
            "LQW\n(Left nm)",
            "LQB\n(Left Bar nm)",
            "MQW\n(Main nm)",
            "RLQW\n(Nº Right)",
            "RQW\n(Right nm)",
            "RQB\n(Right Bar nm)",
            "PC Max\n(A)",
            "PC Peak E\n(meV)",
            "OS Max\n(Força)",
            "OS Trans E\n(meV)"
        ]

        # Barra de seleção de pasta de simulações (Output Folder) integrada
        if not hasattr(self, "output_folder_bar_widget"):
            import conf
            self.output_folder_bar_widget = QWidget()
            folder_bar_layout = QHBoxLayout(self.output_folder_bar_widget)
            folder_bar_layout.setContentsMargins(0, 2, 0, 4)
            folder_bar_layout.setSpacing(6)

            lbl_folder = QLabel("📁 Pasta de Simulações:")
            lbl_folder.setStyleSheet("font-weight: bold; font-size: 11px; color: #1e293b;")

            self.output_folder_line_edit = QLineEdit(conf.output_fortran_folder)
            self.output_folder_line_edit.setStyleSheet(
                "padding: 4px 8px; border: 1px solid #cbd5e1; border-radius: 4px; background: #ffffff; font-size: 11px; color: #334155;"
            )
            self.output_folder_line_edit.setToolTip("Caminho do diretório onde as simulações Fortran são salvas e lidas.")
            self.output_folder_line_edit.returnPressed.connect(self.OnOutputFolderReturnPressed)

            self.btn_browse_output_folder = QPushButton("📁 Procurar...")
            self.btn_browse_output_folder.setStyleSheet(
                "padding: 4px 10px; background-color: #2563eb; color: white; font-weight: bold; border-radius: 4px; font-size: 11px;"
            )
            self.btn_browse_output_folder.setToolTip("Selecionar pasta onde as simulações estão salvas.")
            self.btn_browse_output_folder.clicked.connect(self.OnBrowseOutputFolder)

            self.btn_reload_fortran_table = QPushButton("🔄 Atualizar")
            self.btn_reload_fortran_table.setStyleSheet(
                "padding: 4px 10px; background-color: #f1f5f9; color: #1e293b; border: 1px solid #cbd5e1; font-weight: bold; border-radius: 4px; font-size: 11px;"
            )
            self.btn_reload_fortran_table.setToolTip("Recarrega as simulações existentes na pasta atual.")
            self.btn_reload_fortran_table.clicked.connect(self.UpdateFortranSamplesTable)

            folder_bar_layout.addWidget(lbl_folder)
            folder_bar_layout.addWidget(self.output_folder_line_edit, stretch=1)
            folder_bar_layout.addWidget(self.btn_browse_output_folder)
            folder_bar_layout.addWidget(self.btn_reload_fortran_table)

            if hasattr(self, "fortran_table_top_layout"):
                self.fortran_table_top_layout.insertWidget(0, self.output_folder_bar_widget)
        hdr = ExcelFilterHeaderView(Qt.Horizontal, self.fortran_samples_table)
        hdr.setMinimumHeight(44)
        hdr.setSortIndicatorShown(True)
        hdr.setSectionsClickable(True)
        hdr.setFilterCallback(self.OpenExcelFilterDialogForCol)
        self.fortran_samples_table.setHorizontalHeader(hdr)
        self.fortran_samples_table.setColumnCount(len(self.fortran_column_names))
        self.fortran_samples_table.setHorizontalHeaderLabels(self.fortran_column_names)
        self.fortran_samples_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.fortran_samples_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.fortran_samples_table.setSortingEnabled(True)
        self.fortran_samples_table.horizontalHeader().setToolTip(
            "Clique para ordenar | Clique na seta [▼] ou com o botão direito para menu de filtros tipo Excel."
        )
        self.fortran_samples_table.horizontalHeader().sectionDoubleClicked.connect(self.OpenExcelFilterDialogForCol)
        self.fortran_samples_table.itemSelectionChanged.connect(self.OnFortranSampleSelected)

        if hasattr(self, "fortran_plot_x_cbox") and self.fortran_plot_x_cbox.count() == 0:
            for name in self.fortran_column_names:
                clean_name = name.replace("\n", " ")
                self.fortran_plot_x_cbox.addItem(clean_name)
                self.fortran_plot_y_cbox.addItem(clean_name)
            self.fortran_plot_x_cbox.setCurrentIndex(3)  # MQW
            self.fortran_plot_y_cbox.setCurrentIndex(7)  # PC Max
            self.fortran_plot_x_cbox.currentIndexChanged.connect(self.PlotFortranTableData)
            self.fortran_plot_y_cbox.currentIndexChanged.connect(self.PlotFortranTableData)

        if hasattr(self, "btn_fortran_refresh_plot"):
            self.btn_fortran_refresh_plot.clicked.connect(self.PlotFortranTableData)
        if hasattr(self, "btn_fortran_filter"):
            self.btn_fortran_filter.clicked.connect(self.OpenExcelFilterDialog)
        if hasattr(self, "btn_fortran_clear_filters"):
            self.btn_fortran_clear_filters.clicked.connect(self.ClearFortranFilters)

        # Popula a tabela de simulações imediatamente na inicialização
        self.UpdateFortranSamplesTable()
        self._fortran_table_loaded = True

    def OpenExcelFilterDialog(self):
        """Opens Excel-style filter dropdown for selected column (or MQW col 3 by default)."""
        sel_cols = self.fortran_samples_table.selectionModel().selectedColumns()
        col = sel_cols[0].column() if sel_cols else 3
        self.OpenExcelFilterDialogForCol(col)

    def OpenExcelFilterDialogForCol(self, col: int, global_pt: QPoint = None):
        """Opens Excel-style column dropdown filter popup for fortran_samples_table at col."""
        if col < 0 or col >= len(self.fortran_column_names):
            return

        col_name = self.fortran_column_names[col]
        popup = ExcelColumnFilterPopup(
            parent_win=self,
            table=self.fortran_samples_table,
            col_idx=col,
            col_name=col_name,
            active_filters=self.fortran_table_filters,
            on_applied_callback=self.ApplyFortranFilters
        )

        hdr = self.fortran_samples_table.horizontalHeader()
        if global_pt is None:
            vx = hdr.sectionViewportPosition(col)
            global_pt = hdr.viewport().mapToGlobal(QPoint(vx, hdr.height()))

        screen = QApplication.desktop().availableGeometry(global_pt)
        popup.adjustSize()
        pw = max(270, popup.sizeHint().width())
        ph = max(400, popup.sizeHint().height())
        x = global_pt.x()
        y = global_pt.y()
        if x + pw > screen.right():
            x = max(screen.left(), screen.right() - pw - 10)
        if y + ph > screen.bottom():
            y = max(screen.top(), y - ph - hdr.height() - 10)

        popup.move(x, y)
        popup.exec_()

    def ApplyFortranFilters(self):
        """Applies active filters to fortran_samples_table rows and updates status and plot."""
        total = self.fortran_samples_table.rowCount()
        visible_cnt = 0
        for r in range(total):
            visible = True
            for col_idx, cond in self.fortran_table_filters.items():
                item = self.fortran_samples_table.item(r, col_idx)
                if not item:
                    continue
                txt = item.text().strip()
                raw = getattr(item, "sort_val", None)

                # Check numeric comparison filter
                num_op = cond.get("num_op", "none")
                if num_op != "none" and raw is not None:
                    try:
                        val = float(raw)
                        target = cond.get("num_val", 0.0)
                        if num_op == ">" and not (val > target):
                            visible = False; break
                        elif num_op == ">=" and not (val >= target):
                            visible = False; break
                        elif num_op == "<" and not (val < target):
                            visible = False; break
                        elif num_op == "<=" and not (val <= target):
                            visible = False; break
                        elif num_op == "==" and not (abs(val - target) < 1e-4):
                            visible = False; break
                        elif num_op == "range":
                            v_min = cond.get("min", -1e9)
                            v_max = cond.get("max", 1e9)
                            if not (v_min <= val <= v_max):
                                visible = False; break
                    except (ValueError, TypeError):
                        pass

                # Legacy min/max checks if any
                if cond.get("min") is not None and num_op == "none":
                    if raw is None or float(raw) < cond["min"]:
                        visible = False; break
                if cond.get("max") is not None and num_op == "none":
                    if raw is None or float(raw) > cond["max"]:
                        visible = False; break

                # Checkbox list filter
                if cond.get("checked") is not None:
                    if txt not in cond["checked"]:
                        visible = False; break

            self.fortran_samples_table.setRowHidden(r, not visible)
            if visible:
                visible_cnt += 1

        n_filters = len(self.fortran_table_filters)
        if hasattr(self, "lbl_fortran_filter_status"):
            if n_filters == 0:
                self.lbl_fortran_filter_status.setText(f"({total} visíveis)")
            else:
                self.lbl_fortran_filter_status.setText(f"({visible_cnt}/{total} visíveis - {n_filters} filtros)")

        if hasattr(self, "btn_fortran_clear_filters"):
            self.btn_fortran_clear_filters.setEnabled(n_filters > 0)

        # Notify custom header view to repaint active filter indicator badges
        hdr = self.fortran_samples_table.horizontalHeader()
        if hasattr(hdr, "setActiveFilters"):
            hdr.setActiveFilters(self.fortran_table_filters)

        self.PlotFortranTableData()

    def ClearFortranFilters(self):
        """Clears all active filters on fortran_samples_table."""
        self.fortran_table_filters.clear()
        self.ApplyFortranFilters()

    def PlotFortranTableData(self):
        """Plots scatter of currently filtered rows in fortran_table_subplot."""
        if not hasattr(self, "fortran_table_subplot"):
            return

        col_x = self.fortran_plot_x_cbox.currentIndex() if hasattr(self, "fortran_plot_x_cbox") else 3
        col_y = self.fortran_plot_y_cbox.currentIndex() if hasattr(self, "fortran_plot_y_cbox") else 7

        if col_x < 0: col_x = 3
        if col_y < 0: col_y = 7

        xs, ys, rows = [], [], []
        total = self.fortran_samples_table.rowCount()
        for r in range(total):
            if self.fortran_samples_table.isRowHidden(r):
                continue
            item_x = self.fortran_samples_table.item(r, col_x)
            item_y = self.fortran_samples_table.item(r, col_y)
            if item_x and item_y:
                vx = getattr(item_x, "sort_val", None)
                vy = getattr(item_y, "sort_val", None)
                if vx is not None and vy is not None:
                    try:
                        xs.append(float(vx))
                        ys.append(float(vy))
                        rows.append(r)
                    except (ValueError, TypeError):
                        pass

        self._fortran_table_plot_data = {"xs": np.array(xs), "ys": np.array(ys), "rows": np.array(rows)}

        self.fortran_table_subplot.clear()
        name_x = self.fortran_column_names[col_x].replace("\n", " ") if col_x < len(self.fortran_column_names) else f"Col {col_x}"
        name_y = self.fortran_column_names[col_y].replace("\n", " ") if col_y < len(self.fortran_column_names) else f"Col {col_y}"

        if len(xs) > 0:
            self.fortran_table_subplot.scatter(
                xs, ys,
                color="#2563eb", edgecolor="#1e40af", s=55, alpha=0.85, zorder=3,
                label=f"Amostras ({len(xs)})"
            )

            # Highlight selected row if visible
            sel_rows = self.fortran_samples_table.selectionModel().selectedRows()
            if sel_rows:
                sel_r = sel_rows[0].row()
                if sel_r in rows:
                    idx = list(rows).index(sel_r)
                    self.fortran_table_subplot.scatter(
                        [xs[idx]], [ys[idx]],
                        s=130, facecolors='none', edgecolors='#ef4444', linewidths=2.5, zorder=5
                    )

        self.fortran_table_subplot.set_xlabel(name_x, fontsize=9, fontweight='bold', color="#1e293b")
        self.fortran_table_subplot.set_ylabel(name_y, fontsize=9, fontweight='bold', color="#1e293b")
        self.fortran_table_subplot.set_title(f"{name_y} vs {name_x} ({len(xs)} pontos)", fontsize=9, fontweight='bold', color="#1e293b")
        self.fortran_table_subplot.grid(True, linestyle=":", alpha=0.55)
        self.fortran_table_fig.tight_layout()
        self.fortran_table_canvas.draw_idle()

    def OnFortranTablePlotClicked(self, event):
        """Finds closest point in fortran_table_subplot and selects that row in fortran_samples_table."""
        if hasattr(self, "fortran_table_nav") and self.fortran_table_nav.mode != "":
            return
        if event.inaxes != self.fortran_table_subplot or event.xdata is None or event.ydata is None:
            return
        data = getattr(self, "_fortran_table_plot_data", None)
        if not data or len(data["xs"]) == 0:
            return

        xs, ys, rows = data["xs"], data["ys"], data["rows"]
        xlim = self.fortran_table_subplot.get_xlim()
        ylim = self.fortran_table_subplot.get_ylim()
        span_x = max(1e-6, xlim[1] - xlim[0])
        span_y = max(1e-6, ylim[1] - ylim[0])

        norm_dx = (xs - event.xdata) / span_x
        norm_dy = (ys - event.ydata) / span_y
        dist_sq = norm_dx ** 2 + norm_dy ** 2
        min_idx = int(np.argmin(dist_sq))
        if np.sqrt(dist_sq[min_idx]) <= 0.12:
            target_row = int(rows[min_idx])
            self.fortran_samples_table.selectRow(target_row)
            self.PlotFortranTableData()

    def OnMainTabChanged(self, index: int):
        """Lazy-loads the Fortran samples table when the user switches to the Fortran Sim tab."""
        if hasattr(self, "fortran_tab") and self.tabWidget.widget(index) == self.fortran_tab:
            if not getattr(self, "_fortran_table_loaded", False):
                self.UpdateFortranSamplesTable()
                self._fortran_table_loaded = True

    def UpdateFortranSamplesTable(self):
        """
        Scans temp_database/temp folder for existing simulations and populates fortran_samples_table.
        Uses persistent JSON caching (samples_cache.json) to eliminate startup and disk parsing lag.
        """
        import conf
        from conf import resolve_simulation_paths
        output_fortran_folder = getattr(conf, "output_fortran_folder", "")
        db_dir, sim_dir, pkl_dir, cache_file = resolve_simulation_paths(output_fortran_folder)

        self.SyncOutputFolderDisplays(output_fortran_folder)

        if not sim_dir or not os.path.exists(sim_dir):
            self.fortran_samples_table.setRowCount(0)
            self.fortran_samples_data = []
            if hasattr(self, "lbl_fortran_filter_status"):
                self.lbl_fortran_filter_status.setText("(Pasta de simulações não encontrada)")
            return

        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cache = json.load(f)
            except Exception:
                cache = {}
        cache_dirty = False

        # Disable sorting while inserting rows to prevent row order shifts during population
        self.fortran_samples_table.setSortingEnabled(False)
        self.fortran_samples_table.setRowCount(0)
        self.fortran_samples_data = []

        folder_names = sorted(os.listdir(sim_dir))
        row = 0
        for folder in folder_names:
            folder_path = os.path.join(sim_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            if folder in cache:
                cached_data = cache[folder]
                s0 = cached_data["s0"]
                s1 = cached_data["s1"]
                s2 = cached_data["s2"]
                s3 = cached_data["s3"]
                s4 = cached_data["s4"]
                s5 = cached_data["s5"]
                s6 = cached_data["s6"]
                pc_max = cached_data["pc_max"]
                # Caso legado: converter se valor guardado estava normalizado (> 0.001)
                if pc_max > 0.001:
                    pc_max = pc_max * 1.551237337323166e-11
                    cached_data["pc_max"] = pc_max
                    cache_dirty = True
                pc_e = cached_data["pc_e"]
                os_max = cached_data["os_max"]
                os_e = cached_data["os_e"]
            else:
                # Check if simulation output files exist
                pot_file = os.path.join(folder_path, "Potencial_SL.txt")
                wf_file = os.path.join(folder_path, "wavefunction_SL.txt")
                osc_file = os.path.join(folder_path, "OscStr_SL.txt")
                pc_file = os.path.join(folder_path, "Photocurrent_SL.txt")

                if not (os.path.exists(pot_file) and os.path.exists(wf_file) and os.path.exists(osc_file) and os.path.exists(pc_file)):
                    continue

                try:
                    # Parse structure params from folder name format: 05x02.0_07.0__02.5__01x02.0_07.0
                    parts = folder.split("__")
                    left_parts = parts[0].split("x")
                    s0 = int(left_parts[0])
                    s1_s2 = left_parts[1].split("_")
                    s1 = float(s1_s2[0])
                    s2 = float(s1_s2[1])

                    s3 = float(parts[1])

                    right_parts = parts[2].split("x")
                    s4 = int(right_parts[0])
                    s5_s6 = right_parts[1].split("_")
                    s5 = float(s5_s6[0])
                    s6 = float(s5_s6[1])

                    struct = [s0, round(s1 * 10, 1), round(s2 * 10, 1), round(s3 * 10, 1), s4, round(s5 * 10, 1), round(s6 * 10, 1)]

                    sim_options = SimulationOptions(
                        force_parser=False,
                        force_simulation=False,
                        reference_os=None,
                        reference_os_e=None,
                        reference_pc=None,
                        reference_pc_e=None
                    )
                    simulator = FortranSimulator(struct, sim_options=sim_options, output_folder=db_dir)
                    sample_data = simulator.simulate()

                    pc_max = sample_data.max_abs_photocurrent
                    pc_e = sample_data.max_e_abs_photocurrent
                    os_max = sample_data.max_e_oscstr
                    os_e = sample_data.max_e_transition

                    cache[folder] = {
                        "s0": s0, "s1": s1, "s2": s2, "s3": s3,
                        "s4": s4, "s5": s5, "s6": s6,
                        "pc_max": pc_max, "pc_e": pc_e,
                        "os_max": os_max, "os_e": os_e
                    }
                    cache_dirty = True
                except Exception as e:
                    print(f"Error reading sample folder {folder}: {e}")
                    continue

            self.fortran_samples_table.insertRow(row)

            display_vals = [
                f"{s0}", f"{s1:.1f}", f"{s2:.1f}", f"{s3:.1f}", f"{s4}", f"{s5:.1f}", f"{s6:.1f}",
                f"{pc_max:.2e}", f"{pc_e:.1f}", f"{os_max:.4f}", f"{os_e:.1f}"
            ]
            raw_vals = [
                s0, s1, s2, s3, s4, s5, s6,
                pc_max, pc_e, os_max, os_e
            ]

            sample_dict = {
                "s0": s0, "s1": s1, "s2": s2, "s3": s3,
                "s4": s4, "s5": s5, "s6": s6
            }
            self.fortran_samples_data.append(sample_dict)

            for col, (val, raw) in enumerate(zip(display_vals, raw_vals)):
                item = NumericTableWidgetItem(val, raw)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 0:
                    item.setData(Qt.UserRole, sample_dict)
                self.fortran_samples_table.setItem(row, col, item)

            row += 1

        self.fortran_samples_table.resizeColumnsToContents()
        self.fortran_samples_table.setSortingEnabled(True)
        if hasattr(self, "lbl_fortran_filter_status"):
            self.lbl_fortran_filter_status.setText(f"Mostrando {row} de {row} simulações (Sem filtros ativos)")

        if cache_dirty:
            try:
                with open(cache_file, "w") as f:
                    json.dump(cache, f, indent=2)
            except Exception as e:
                print(f"Error saving samples cache: {e}")

    def OnFortranSampleSelected(self):
        """
        Triggered when a row in fortran_samples_table is clicked.
        Loads the selected sample parameters into spinboxes and updates the 3 plots.
        """
        selected_rows = self.fortran_samples_table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        item0 = self.fortran_samples_table.item(row, 0)
        sample = item0.data(Qt.UserRole) if item0 else None
        if not sample and row < len(self.fortran_samples_data):
            sample = self.fortran_samples_data[row]

        if sample:
            # Block signals while setting values to avoid redundant redraw calls
            self.fortran_s0_spb.blockSignals(True)
            self.fortran_s1_spb.blockSignals(True)
            self.fortran_s2_spb.blockSignals(True)
            self.fortran_s3_spb.blockSignals(True)
            self.fortran_s4_spb.blockSignals(True)
            self.fortran_s5_spb.blockSignals(True)
            self.fortran_s6_spb.blockSignals(True)

            self.fortran_s0_spb.setValue(sample["s0"])
            self.fortran_s1_spb.setValue(sample["s1"])
            self.fortran_s2_spb.setValue(sample["s2"])
            self.fortran_s3_spb.setValue(sample["s3"])
            self.fortran_s4_spb.setValue(sample["s4"])
            self.fortran_s5_spb.setValue(sample["s5"])
            self.fortran_s6_spb.setValue(sample["s6"])

            self.fortran_s0_spb.blockSignals(False)
            self.fortran_s1_spb.blockSignals(False)
            self.fortran_s2_spb.blockSignals(False)
            self.fortran_s3_spb.blockSignals(False)
            self.fortran_s4_spb.blockSignals(False)
            self.fortran_s5_spb.blockSignals(False)
            self.fortran_s6_spb.blockSignals(False)

            self.PlotFortranBandStructure()
            self.PlotFortranTableData()

    # GA tab methods ##################################################################
    def InitializeGATab(self):
        """Initializes the Genetic Algorithm tab controls, tooltips, help buttons, and signals."""
        if hasattr(self, "ga_splitter"):
            self.ga_splitter.setStretchFactor(0, 1)
            self.ga_splitter.setStretchFactor(1, 2)
        if hasattr(self, "ga_right_splitter"):
            self.ga_right_splitter.setStretchFactor(0, 5)
            self.ga_right_splitter.setStretchFactor(1, 5)
        if hasattr(self, "ga_ind_plots_splitter"):
            # Second row: 1/3 (Text), 1/3 (Structure), 1/6 (Oscillator Strength), 1/6 (Photocurrent) -> 2 : 2 : 1 : 1
            self.ga_ind_plots_splitter.setStretchFactor(0, 2)
            self.ga_ind_plots_splitter.setStretchFactor(1, 2)
            self.ga_ind_plots_splitter.setStretchFactor(2, 1)
            self.ga_ind_plots_splitter.setStretchFactor(3, 1)
            self.ga_ind_plots_splitter.setSizes([380, 380, 190, 190])

        # Auto-detect CPU cores and configure parallel controls
        total_cores = os.cpu_count() or 4
        self.ga_parallel_cores_spb.setMinimum(1)
        self.ga_parallel_cores_spb.setMaximum(total_cores)
        default_cores = max(1, total_cores - 2) if total_cores > 4 else total_cores
        self.ga_parallel_cores_spb.setValue(default_cores)
        if hasattr(self, "lbl_ga_cores"):
            self.lbl_ga_cores.setText(f"Núcleos de CPU (Máx: {total_cores}):")

        # Set Portuguese tooltips
        self.ga_num_params_cbox.setToolTip(GA_PARAM_INFO["num_params"][1])
        self.ga_pop_spb.setToolTip(GA_PARAM_INFO["pop_size"][1])
        self.ga_iter_spb.setToolTip(GA_PARAM_INFO["generations"][1])
        self.ga_elit_spb.setToolTip(GA_PARAM_INFO["elitism"][1])
        self.ga_cross_prob_spb.setToolTip(GA_PARAM_INFO["crossover_prob"][1])
        self.ga_cross_cbox.setToolTip(GA_PARAM_INFO["crossover_type"][1])
        self.ga_mut_prob_spb.setToolTip(GA_PARAM_INFO["mutation_prob"][1])
        self.ga_parallel_chkbx.setToolTip(GA_PARAM_INFO["parallel"][1])
        self.ga_parallel_cores_spb.setToolTip(GA_PARAM_INFO["parallel"][1])
        self.ga_use_seeds_chkbx.setToolTip(GA_PARAM_INFO["seeds"][1])
        self.ga_seeds_txt.setToolTip(GA_PARAM_INFO["seeds"][1])
        self.ga_add_current_seed_btn.setToolTip("Adiciona a estrutura atual da aba Fortran Sim como indivíduo semente.")
        self.ga_clear_seeds_btn.setToolTip("Limpa o campo de texto de sementes.")
        if hasattr(self, "ga_opt_pc_chkbx"):
            self.ga_opt_pc_chkbx.setToolTip("Maximiza a intensidade do pico de fotocorrente (PC) gerado pela estrutura.")
        if hasattr(self, "ga_opt_os_chkbx"):
            self.ga_opt_os_chkbx.setToolTip("Maximiza a magnitude da força de oscilador óptica (OS) calculada na simulação.")
        if hasattr(self, "ga_limit_energy_chkbx"):
            self.ga_limit_energy_chkbx.setToolTip(
                "Restringe a busca à faixa [Alvo ± Margem] com penalização contínua proporcional para picos fora da borda."
            )
        self.ga_tgt_en_spb.setToolTip(GA_PARAM_INFO["target_energy"][1])
        self.ga_tgt_en_margin_spb.setToolTip(GA_PARAM_INFO["target_margin"][1])
        self.ga_resume_chkbx.setToolTip(GA_PARAM_INFO["resume"][1])

        # Valores padrão para a interface do GA: Energia 300 meV, Margem de tolerância 20 meV
        self.ga_tgt_en_spb.setValue(300.0)
        self.ga_tgt_en_margin_spb.setValue(20.0)
        if hasattr(self, "ga_limit_energy_chkbx"):
            self.ga_limit_energy_chkbx.setChecked(True)

        # Connect help buttons
        help_map = [
            ("btn_help_params", "num_params"),
            ("btn_help_pop", "pop_size"),
            ("btn_help_iter", "generations"),
            ("btn_help_elit", "elitism"),
            ("btn_help_cross_prob", "crossover_prob"),
            ("btn_help_cross_type", "crossover_type"),
            ("btn_help_mut", "mutation_prob"),
            ("btn_help_parallel", "parallel"),
            ("btn_help_seeds", "seeds"),
            ("btn_help_goal", "goal"),
            ("btn_help_tgt_en", "target_energy"),
            ("btn_help_tgt_margin", "target_margin"),
            ("btn_help_resume", "resume"),
        ]
        for btn_name, key in help_map:
            if hasattr(self, btn_name):
                btn = getattr(self, btn_name)
                btn.clicked.connect(lambda checked, k=key: self.ShowGAHelp(k))

        # Connect operational buttons
        self.ga_run_btn.clicked.connect(self.OnGARunClicked)
        self.ga_stop_btn.clicked.connect(self.OnGAStopClicked)
        self.ga_select_csv_btn.clicked.connect(self.OnGASelectCSV)
        self.ga_plot_csv_btn.clicked.connect(self.OnGAPlotCSV)
        self.ga_apply_to_fortran_btn.clicked.connect(self.OnGAApplyToFortran)
        if hasattr(self, "ga_copy_text_btn"):
            self.ga_copy_text_btn.clicked.connect(self.OnGACopyTextClicked)
            self.ga_copy_text_btn.setToolTip("Copia as informações do indivíduo para a área de transferência.")
        if hasattr(self, "ga_show_best_btn"):
            self.ga_show_best_btn.clicked.connect(self.OnGAShowBestClicked)
            self.ga_show_best_btn.setToolTip("Retorna a inspeção e os gráficos para o melhor indivíduo global.")
        self.ga_resume_chkbx.toggled.connect(self.OnGAResumeToggled)
        self.ga_parallel_chkbx.toggled.connect(self.OnGAParallelToggled)
        self.ga_use_seeds_chkbx.toggled.connect(self.OnGAUseSeedsToggled)
        if hasattr(self, "ga_limit_energy_chkbx"):
            self.ga_limit_energy_chkbx.toggled.connect(self.OnGALimitEnergyToggled)
            self.OnGALimitEnergyToggled(self.ga_limit_energy_chkbx.isChecked())
        self.ga_add_current_seed_btn.clicked.connect(self.OnGAAddCurrentSeed)
        self.ga_clear_seeds_btn.clicked.connect(self.ga_seeds_txt.clear)

        self.ga_run_btn.setEnabled(True)
        self.ga_stop_btn.setEnabled(False)

        # Initialize evolution plot
        self.ga_subplot.clear()
        self.ga_subplot.set_xlabel("Geração (Generation)", fontsize=10, fontweight='bold', color="#1e293b")
        self.ga_subplot.set_ylabel("Aptidão (Fitness)", fontsize=10, fontweight='bold', color="#1e293b")
        self.ga_subplot.set_title("Evolução do Algoritmo Genético (Clique em um indivíduo para inspecionar)", fontsize=10, fontweight='bold', color="#1e293b")
        self.ga_subplot.grid(True, linestyle=":", alpha=0.55)
        self.ga_fig.tight_layout()
        self.ga_canvas.draw()

        # Initialize individual inspection subplots
        if hasattr(self, "ga_ind_subplot1"):
            self.ga_ind_subplot1.clear()
            self.ga_ind_subplot1.set_title("Potencial e Funções de Onda", fontsize=9, fontweight='bold', color="#1e293b")
            self.ga_ind_subplot1.grid(True, linestyle=":", alpha=0.55)
            self.ga_ind_subplot2.clear()
            self.ga_ind_subplot2.set_title("Força de Oscilador", fontsize=9, fontweight='bold', color="#1e293b")
            self.ga_ind_subplot2.yaxis.tick_left()
            self.ga_ind_subplot2.yaxis.set_label_position("left")
            self.ga_ind_subplot2.grid(True, linestyle=":", alpha=0.55)
            self.ga_ind_subplot3.clear()
            self.ga_ind_subplot3.set_title("Photocurrent", fontsize=9, fontweight='bold', color="#1e293b")
            self.ga_ind_subplot3.yaxis.tick_right()
            self.ga_ind_subplot3.yaxis.set_label_position("right")
            self.ga_ind_subplot3.set_ylabel('ΔE (meV)', fontdict=None, labelpad=0)
            self.ga_ind_subplot3.grid(True, linestyle=":", alpha=0.55)
            self.ga_ind_fig1.tight_layout()
            self.ga_ind_fig2.tight_layout()
            self.ga_ind_fig3.tight_layout()
            self.ga_ind_canvas1.draw()
            self.ga_ind_canvas2.draw()
            self.ga_ind_canvas3.draw()

    def PlotGAIndividual(self, ind: list, title_suffix: str = ""):
        """Plots the 3 Fortran simulation curves for an individual in the bottom GA panels."""
        if not hasattr(self, "ga_ind_subplot1"):
            return

        from conf import output_fortran_folder
        sim_options = SimulationOptions(force_parser=False, force_simulation=False)
        simulator = FortranSimulator(ind, sim_options=sim_options, output_folder=output_fortran_folder)

        self.ga_ind_subplot1.clear()
        self.ga_ind_subplot2.clear()
        self.ga_ind_subplot3.clear()

        sim_exists = (
            os.path.exists(simulator.potencial_file) and
            os.path.exists(simulator.wavefunction_file) and
            os.path.exists(simulator.oscstr_file) and
            os.path.exists(simulator.photocurrent_file)
        )

        if sim_exists:
            try:
                sample_data = simulator.simulate()
                Plotter.plot_structure(
                    self.ga_ind_subplot1,
                    sample_data.x_potencial,
                    sample_data.y_potencial,
                    sample_data.autoenergias,
                    sample_data.wavefunctions,
                    sample_data.max_e_oscstr_index,
                    -50, 700
                )
                self.ga_ind_subplot1.set_title(f"Potencial e Funções de Onda{title_suffix}", fontsize=9, fontweight='bold', color='#1e293b')

                Plotter.plot_osc(
                    self.ga_ind_subplot2,
                    oscstr=sample_data.oscstr,
                    oscstr_e=sample_data.oscstr_e,
                    E0=sample_data.E0,
                    lim_E_min=-50,
                    lim_E_max=700,
                    max_e_oscstr_index=sample_data.max_e_oscstr_index
                )
                self.ga_ind_subplot2.yaxis.tick_left()
                self.ga_ind_subplot2.yaxis.set_label_position("left")
                self.ga_ind_subplot2.set_title(f"Força de Oscilador{title_suffix}", fontsize=9, fontweight='bold', color='#1e293b')

                Plotter.plot_pc(
                    self.ga_ind_subplot3,
                    pc=sample_data.pc,
                    pc_e=sample_data.pc_e,
                    E0=sample_data.E0,
                    lim_E_min=-50,
                    lim_E_max=700,
                    max_abs_photocurrent=sample_data.max_abs_photocurrent,
                    max_e_abs_photocurrent=sample_data.max_e_abs_photocurrent,
                    min_abs_photocurrent=sample_data.min_abs_photocurrent,
                    min_e_abs_photocurrent=sample_data.min_e_abs_photocurrent
                )
                self.ga_ind_subplot3.yaxis.tick_right()
                self.ga_ind_subplot3.yaxis.set_label_position("right")
                self.ga_ind_subplot3.set_ylabel('ΔE (meV)', fontdict=None, labelpad=0)
                self.ga_ind_subplot3.set_title(f"Photocurrent{title_suffix}", fontsize=9, fontweight='bold', color='#1e293b')
            except Exception as e:
                print(f"Erro ao renderizar gráficos do indivíduo GA: {e}")
                struct = set_structure_values(ind)
                self._plot_fallback_ga_band_profile(struct)
        else:
            try:
                sample_data = simulator.simulate()
                Plotter.plot_structure(
                    self.ga_ind_subplot1,
                    sample_data.x_potencial,
                    sample_data.y_potencial,
                    sample_data.autoenergias,
                    sample_data.wavefunctions,
                    sample_data.max_e_oscstr_index,
                    -50, 700
                )
                self.ga_ind_subplot1.set_title(f"Potencial e Funções de Onda{title_suffix}", fontsize=9, fontweight='bold', color='#1e293b')

                Plotter.plot_osc(
                    self.ga_ind_subplot2,
                    oscstr=sample_data.oscstr,
                    oscstr_e=sample_data.oscstr_e,
                    E0=sample_data.E0,
                    lim_E_min=-50,
                    lim_E_max=700,
                    max_e_oscstr_index=sample_data.max_e_oscstr_index
                )
                self.ga_ind_subplot2.yaxis.tick_left()
                self.ga_ind_subplot2.yaxis.set_label_position("left")
                self.ga_ind_subplot2.set_title(f"Força de Oscilador{title_suffix}", fontsize=9, fontweight='bold', color='#1e293b')

                Plotter.plot_pc(
                    self.ga_ind_subplot3,
                    pc=sample_data.pc,
                    pc_e=sample_data.pc_e,
                    E0=sample_data.E0,
                    lim_E_min=-50,
                    lim_E_max=700,
                    max_abs_photocurrent=sample_data.max_abs_photocurrent,
                    max_e_abs_photocurrent=sample_data.max_e_abs_photocurrent,
                    min_abs_photocurrent=sample_data.min_abs_photocurrent,
                    min_e_abs_photocurrent=sample_data.min_e_abs_photocurrent
                )
                self.ga_ind_subplot3.yaxis.tick_right()
                self.ga_ind_subplot3.yaxis.set_label_position("right")
                self.ga_ind_subplot3.set_ylabel('ΔE (meV)', fontdict=None, labelpad=0)
                self.ga_ind_subplot3.set_title(f"Photocurrent{title_suffix}", fontsize=9, fontweight='bold', color='#1e293b')
            except Exception:
                struct = set_structure_values(ind)
                self._plot_fallback_ga_band_profile(struct)

        self.ga_ind_fig1.tight_layout()
        self.ga_ind_fig2.tight_layout()
        self.ga_ind_fig3.tight_layout()
        self.ga_ind_canvas1.draw_idle()
        self.ga_ind_canvas2.draw_idle()
        self.ga_ind_canvas3.draw_idle()

    def _plot_fallback_ga_band_profile(self, struct):
        """Fallback method to plot conduction band profile in GA tab."""
        w_l = struct[0]
        qw_l = struct[1] / 10.0 if struct[1] > 20 else struct[1]
        qb_l = struct[2] / 10.0 if struct[2] > 20 else struct[2]
        qw_c = struct[3] / 10.0 if struct[3] > 20 else struct[3]
        w_r = struct[4]
        qw_r = struct[5] / 10.0 if struct[5] > 20 else struct[5]
        qb_r = struct[6] / 10.0 if struct[6] > 20 else struct[6]

        barrier_energy = 500.0
        well_energy = 0.0

        thicknesses = [50.0]
        energies = [barrier_energy]

        for _ in range(int(w_l)):
            thicknesses.extend([qw_l, qb_l])
            energies.extend([well_energy, barrier_energy])

        thicknesses.append(qw_c)
        energies.append(well_energy)

        for _ in range(int(w_r)):
            thicknesses.extend([qb_r, qw_r])
            energies.extend([barrier_energy, well_energy])

        thicknesses.append(50.0)
        energies.append(barrier_energy)

        left_total_thickness = 50.0 + w_l * (qw_l + qb_l)
        current_x = -left_total_thickness

        x_pts, y_pts = [], []
        for th, en in zip(thicknesses, energies):
            x_pts.extend([current_x, current_x + th])
            y_pts.extend([en, en])
            current_x += th

        self.ga_ind_subplot1.plot(x_pts, y_pts, color="#1e3a8a", linewidth=1.4)
        self.ga_ind_subplot1.fill_between(x_pts, y_pts, color="#dbeafe", alpha=0.45)
        self.ga_ind_subplot1.set_xlabel("Thickness (nm)")
        self.ga_ind_subplot1.set_ylabel("Energy (meV)")
        self.ga_ind_subplot1.set_title("Potencial da Estrutura", fontsize=9, fontweight='bold', color='#1e293b')
        self.ga_ind_subplot1.grid(True, linestyle=":", alpha=0.55)
        self.ga_ind_subplot1.set_ylim(-50, 700)

    def OnGAEvolutionPlotClicked(self, event):
        """
        Interactive click on the GA evolution scatter plot.
        Finds the nearest individual in generation vs fitness space, highlights it,
        renders its 3 simulation plots (Band Structure, OS, PC) in the lower panel,
        and updates the information display while preserving the best individual summary.
        """
        if hasattr(self, "ga_nav") and self.ga_nav.mode != "":
            return

        if event.inaxes != self.ga_subplot or event.xdata is None or event.ydata is None:
            return

        if not hasattr(self, "ga_last_df") or self.ga_last_df is None or self.ga_last_df.empty:
            return

        df = self.ga_last_df
        try:
            x_min, x_max = self.ga_subplot.get_xlim()
            y_min, y_max = self.ga_subplot.get_ylim()
            span_x = max(1e-5, (x_max - x_min))
            span_y = max(1e-5, (y_max - y_min))

            norm_dx = (df["generation"].values - event.xdata) / span_x
            norm_dy = (df["fitness"].values - event.ydata) / span_y
            dist_sq = norm_dx ** 2 + norm_dy ** 2
            nearest_idx = int(np.argmin(dist_sq))
            min_dist = float(np.sqrt(dist_sq[nearest_idx]))

            if min_dist > 0.12:
                return

            row = df.iloc[nearest_idx]
            gen = int(row["generation"])
            ind_raw = ast.literal_eval(str(row["individual"]))

            self.ga_selected_individual = ind_raw
            self.ga_selected_row = row

            if hasattr(self, "_ga_selected_scatter") and self._ga_selected_scatter:
                try:
                    self._ga_selected_scatter.remove()
                except Exception:
                    pass
            self._ga_selected_scatter = self.ga_subplot.scatter(
                [row["generation"]], [row["fitness"]],
                s=130, facecolors='none', edgecolors='#ef4444', linewidths=2.5, zorder=10
            )
            self.ga_canvas.draw_idle()

            self.PlotGAIndividual(ind_raw, title_suffix=f" (Geração {gen})")
            self._UpdateGASummaryDisplay(selected_row=row)

            if hasattr(self, "ga_apply_to_fortran_btn"):
                self.ga_apply_to_fortran_btn.setEnabled(True)
            if hasattr(self, "ga_show_best_btn"):
                self.ga_show_best_btn.setEnabled(True)

        except Exception as e:
            print(f"Erro ao selecionar indivíduo no gráfico: {e}")

    def _UpdateGASummaryDisplay(self, selected_row=None):
        """Updates ga_best_summary_lbl combining preserved best individual and selected individual."""
        best_txt = getattr(self, "ga_best_summary_text", "")
        if not best_txt:
            best_txt = self.ga_best_summary_lbl.text()

        parts = []
        if best_txt and "Nenhuma simulação" not in best_txt:
            parts.append(f"🏆 MELHOR INDIVÍDUO GLOBAL:\n{best_txt}")
        else:
            parts.append("🏆 MELHOR INDIVÍDUO GLOBAL: (Aguardando simulações)")

        if selected_row is not None:
            gen = int(selected_row.get("generation", 0))
            fit = float(selected_row.get("fitness", 0.0))
            ind_genes = selected_row.get("individual", "")
            struct = selected_row.get("structure", "")
            pc1 = selected_row.get("PC max 1", "")
            pc1e = selected_row.get("PC max 1 energy (meV)", "")
            osc = selected_row.get("OscStr max", "")
            osce = selected_row.get("OscStr energy (meV)", "")

            sel_text = (
                f"🔍 INDIVÍDUO SELECIONADO NO GRÁFICO (Geração {gen}):\n"
                f"Aptidão (Fitness): {fit:.4f} | Genes: {ind_genes}\n"
                f"Estrutura: {struct}\n"
                f"Fotocorrente (PC Máx): {pc1} (Pico em {pc1e} meV)\n"
                f"Força de Oscilador (OS Máx): {osc} (Transição em {osce} meV)"
            )
            parts.append(sel_text)
        else:
            parts.append("💡 Dica: Clique em qualquer ponto do gráfico de evolução acima para inspecionar seus 3 gráficos e parâmetros.")

        full_display = "\n\n".join(parts)
        self.ga_best_summary_lbl.setText(full_display)
        self._ga_current_display_text = full_display

    def OnGACopyTextClicked(self):
        """Copies the displayed GA summary text to system clipboard."""
        text = getattr(self, "_ga_current_display_text", "") or self.ga_best_summary_lbl.text()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            self.ga_copy_text_btn.setText("✓ Copiado com Sucesso!")
            QTimer.singleShot(2000, lambda: self.ga_copy_text_btn.setText("📋 Copiar Texto"))

    def OnGAShowBestClicked(self):
        """Re-displays and plots the best individual found."""
        if not hasattr(self, "ga_best_individual") or not self.ga_best_individual:
            return

        self.ga_selected_individual = self.ga_best_individual
        best_row = None
        if hasattr(self, "ga_last_df") and self.ga_last_df is not None and not self.ga_last_df.empty:
            df = self.ga_last_df
            best_row = df.loc[df["fitness"].idxmax()]
            if hasattr(self, "_ga_selected_scatter") and self._ga_selected_scatter:
                try:
                    self._ga_selected_scatter.remove()
                except Exception:
                    pass
            self._ga_selected_scatter = self.ga_subplot.scatter(
                [best_row["generation"]], [best_row["fitness"]],
                s=130, facecolors='none', edgecolors='#059669', linewidths=2.5, zorder=10
            )
            self.ga_canvas.draw_idle()

        self.PlotGAIndividual(self.ga_best_individual, title_suffix=" (Melhor Global)")
        self._UpdateGASummaryDisplay(selected_row=best_row)

    def OnGALimitEnergyToggled(self, checked: bool):
        """Enables or disables target energy and margin spinboxes according to checkbox state."""
        self.ga_tgt_en_spb.setEnabled(checked)
        self.ga_tgt_en_margin_spb.setEnabled(checked)
        if hasattr(self, "lbl_ga_tgt_en"):
            self.lbl_ga_tgt_en.setEnabled(checked)
        if hasattr(self, "lbl_ga_tgt_margin"):
            self.lbl_ga_tgt_margin.setEnabled(checked)

    def OnGAParallelToggled(self, checked: bool):
        """Enables or disables parallel cores spinbox."""
        self.ga_parallel_cores_spb.setEnabled(checked)

    def OnGAUseSeedsToggled(self, checked: bool):
        """Enables or disables seeds input and buttons."""
        self.ga_seeds_txt.setEnabled(checked)
        self.ga_add_current_seed_btn.setEnabled(checked)
        self.ga_clear_seeds_btn.setEnabled(checked)

    def OnGAAddCurrentSeed(self):
        """Converts current Fortran Sim tab spinbox parameters into a seed structure."""
        param_text = self.ga_num_params_cbox.currentText()
        if "3" in param_text:
            n_params = 3
            seed = [
                int(round(self.fortran_s1_spb.value() * 10)),
                int(round(self.fortran_s2_spb.value() * 10)),
                int(round(self.fortran_s3_spb.value() * 10)),
            ]
        elif "5" in param_text:
            n_params = 5
            seed = [
                int(round(self.fortran_s0_spb.value())),
                int(round(self.fortran_s1_spb.value() * 10)),
                int(round(self.fortran_s2_spb.value() * 10)),
                int(round(self.fortran_s3_spb.value() * 10)),
                int(round(self.fortran_s4_spb.value())),
            ]
        else:
            n_params = 7
            seed = [
                int(round(self.fortran_s0_spb.value())),
                int(round(self.fortran_s1_spb.value() * 10)),
                int(round(self.fortran_s2_spb.value() * 10)),
                int(round(self.fortran_s3_spb.value() * 10)),
                int(round(self.fortran_s4_spb.value())),
                int(round(self.fortran_s5_spb.value() * 10)),
                int(round(self.fortran_s6_spb.value() * 10)),
            ]

        # Enforce physical constraints (MQW >= max(LQW, RQW) + 5)
        bounds = DEFAULT_BOUNDS.get(n_params, DEFAULT_BOUNDS[3])
        clamp_individual(seed, bounds)

        current = self.ga_seeds_txt.toPlainText().strip()
        new_text = f"{current}\n{seed}" if current else f"{seed}"
        self.ga_seeds_txt.setPlainText(new_text)

        if not self.ga_use_seeds_chkbx.isChecked():
            self.ga_use_seeds_chkbx.setChecked(True)

        self.ga_status_lbl.setText(f"Semente adicionada: {seed}")

    def ShowGAHelp(self, key: str):
        """Displays informative help dialog for GA parameters in Portuguese."""
        if key in GA_PARAM_INFO:
            title, desc = GA_PARAM_INFO[key]
            QMessageBox.information(self, f"Ajuda: {title}", f"{title}\n\n{desc}")

    def OnGAResumeToggled(self, checked: bool):
        """Handles toggling of the resume optimization checkbox."""
        if checked:
            if not self.ga_csv_path_txt.text().strip():
                self.OnGASelectCSV()
            self.ga_run_btn.setText("Continuar Otimização")
        else:
            self.ga_run_btn.setText("Iniciar Otimização")

    def OnGASelectCSV(self):
        """Opens file dialog for user to select an existing optimization CSV file."""
        opt_folder = os.path.join(output_fortran_folder, "optimizations")
        if not os.path.exists(opt_folder):
            opt_folder = output_fortran_folder

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar Arquivo CSV de Otimização",
            opt_folder,
            "Arquivos CSV (*.csv);;Todos os Arquivos (*)"
        )
        if filename:
            self.ga_csv_path_txt.setText(filename)
            self.ga_resume_chkbx.setChecked(True)
            self.ga_run_btn.setText("Continuar Otimização")

            # Try to inspect CSV and set parameter count
            try:
                df = pd.read_csv(filename)
                if not df.empty and "individual" in df.columns:
                    sample_ind = ast.literal_eval(str(df.iloc[0]["individual"]))
                    n_p = len(sample_ind)
                    if n_p == 3:
                        self.ga_num_params_cbox.setCurrentIndex(0)
                    elif n_p == 5:
                        self.ga_num_params_cbox.setCurrentIndex(1)
                    elif n_p == 7:
                        self.ga_num_params_cbox.setCurrentIndex(2)

                    max_g = int(df["generation"].max())
                    best_f = float(df["fitness"].max())
                    best_row = df.loc[df["fitness"].idxmax()]
                    pc1 = best_row.get("PC max 1", "")
                    pc1e = best_row.get("PC max 1 energy (meV)", "")
                    oscstr = best_row.get("OscStr max", "")
                    oscstre = best_row.get("OscStr energy (meV)", "")
                    self.ga_best_summary_text = (
                        f"CSV Carregado: {os.path.basename(filename)}\n"
                        f"Última Geração: {max_g} | Melhor Fitness Histórico: {best_f:.4f}\n"
                        f"Melhor Estrutura: {best_row.get('structure', '')}\n"
                        f"Fotocorrente (PC Máx): {pc1} (Pico em {pc1e} meV)\n"
                        f"Força de Oscilador (OS Máx): {oscstr} (Transição em {oscstre} meV)\n"
                        f"Indivíduo (Genes): {best_row.get('individual', '')}"
                    )
                    self._UpdateGASummaryDisplay(selected_row=best_row)
                    try:
                        self.ga_best_individual = ast.literal_eval(str(best_row["individual"]))
                        self.ga_selected_individual = self.ga_best_individual
                        self.ga_apply_to_fortran_btn.setEnabled(True)
                        if hasattr(self, "ga_show_best_btn"):
                            self.ga_show_best_btn.setEnabled(True)
                        self.PlotGAIndividual(self.ga_best_individual, title_suffix=" (Melhor Histórico)")
                    except Exception:
                        pass
            except Exception as e:
                print(f"Aviso ao analisar CSV: {e}")

            self.PlotGAProgress(filename)

    def OnGAPlotCSV(self):
        """Plots the progress from the CSV specified in the line edit."""
        csv_path = self.ga_csv_path_txt.text().strip()
        if not csv_path or not os.path.exists(csv_path):
            QMessageBox.warning(self, "Aviso", "Por favor, selecione um arquivo CSV existente primeiro.")
            return
        self.PlotGAProgress(csv_path)

    def PlotGAProgress(self, csv_path: str = None):
        """Reads the optimization CSV and renders generation vs fitness on ga_subplot."""
        if not csv_path:
            csv_path = self.ga_csv_path_txt.text().strip()
        if not csv_path or not os.path.exists(csv_path):
            return

        try:
            df = pd.read_csv(csv_path)
            if df.empty or "generation" not in df.columns or "fitness" not in df.columns:
                return

            df["generation"] = pd.to_numeric(df["generation"], errors="coerce")
            df["fitness"] = pd.to_numeric(df["fitness"], errors="coerce")
            df = df.dropna(subset=["generation", "fitness"])

            self.ga_last_df = df

            stats = df.groupby("generation")["fitness"].agg(["max", "mean", "min"]).reset_index()
            x_gen = stats["generation"].values
            y_max = stats["max"].values
            y_mean = stats["mean"].values

            self.ga_subplot.clear()

            # Plot ALL individuals as slate gray dots with soft alpha
            self.ga_subplot.scatter(
                df["generation"].values,
                df["fitness"].values,
                color="#64748b",
                alpha=0.40,
                s=24,
                edgecolors="none",
                label="Indivíduos (Todos)"
            )

            # Plot Best (max) fitness line in vibrant emerald green
            self.ga_subplot.plot(
                x_gen, y_max,
                color="#059669", marker="o", markersize=6, linewidth=2.2,
                label="Melhor Aptidão (Max)"
            )

            # Plot Mean fitness line in sapphire blue
            self.ga_subplot.plot(
                x_gen, y_mean,
                color="#2563eb", marker="s", markersize=5, linewidth=1.8,
                linestyle="--", label="Aptidão Média (Mean)"
            )

            # If an individual was selected, re-draw the highlight marker
            if hasattr(self, "ga_selected_row") and self.ga_selected_row is not None:
                sel_row = self.ga_selected_row
                self._ga_selected_scatter = self.ga_subplot.scatter(
                    [sel_row["generation"]], [sel_row["fitness"]],
                    s=130, facecolors='none', edgecolors='#ef4444', linewidths=2.5, zorder=10
                )

            max_g = int(x_gen.max()) if len(x_gen) > 0 else 1
            best_f = float(y_max.max()) if len(y_max) > 0 else 0.0

            n_params_text = ""
            if "individual" in df.columns and len(df) > 0:
                try:
                    s_ind = ast.literal_eval(str(df.iloc[0]["individual"]))
                    n_params_text = f" ({len(s_ind)} Parâmetros)"
                except Exception:
                    pass

            self.ga_subplot.set_xlabel("Geração (Generation)", fontsize=10, fontweight='bold', color="#1e293b")
            self.ga_subplot.set_ylabel("Aptidão (Fitness)", fontsize=10, fontweight='bold', color="#1e293b")
            self.ga_subplot.set_title(
                f"Evolução do Algoritmo Genético{n_params_text} - Geração {max_g} | Melhor: {best_f:.4f} (Clique em um indivíduo para inspecionar)",
                fontsize=10, fontweight='bold', color="#1e293b"
            )
            self.ga_subplot.grid(True, linestyle=":", alpha=0.55)
            self.ga_subplot.legend(loc="lower right")

            self.ga_fig.tight_layout()
            self.ga_canvas.draw_idle()
        except Exception as e:
            print(f"Erro ao plotar progresso do GA: {e}")

    def OnGARunClicked(self):
        """Starts or continues DEAP genetic algorithm execution in background thread."""
        if self.ga_worker and self.ga_worker.isRunning():
            return

        param_text = self.ga_num_params_cbox.currentText()
        if "3" in param_text:
            n_params = 3
        elif "5" in param_text:
            n_params = 5
        else:
            n_params = 7

        pop_size = self.ga_pop_spb.value()
        generations = self.ga_iter_spb.value()
        elitism = self.ga_elit_spb.value()
        cx_prob = self.ga_cross_prob_spb.value()
        cx_type = self.ga_cross_cbox.currentText()
        mut_prob = self.ga_mut_prob_spb.value()

        opt_pc = self.ga_opt_pc_chkbx.isChecked() if hasattr(self, "ga_opt_pc_chkbx") else True
        opt_os = self.ga_opt_os_chkbx.isChecked() if hasattr(self, "ga_opt_os_chkbx") else True
        limit_energy = self.ga_limit_energy_chkbx.isChecked() if hasattr(self, "ga_limit_energy_chkbx") else True

        if not opt_pc and not opt_os:
            QMessageBox.warning(
                self,
                "Aviso - Objetivo de Otimização",
                "Por favor, selecione pelo menos um objetivo a ser otimizado:\n"
                "• Otimizar Fotocorrente (PC)\n"
                "• Otimizar Força de Oscilador (OS)"
            )
            return

        if opt_pc and opt_os:
            objective = "ospc"
        elif opt_pc:
            objective = "pc"
        else:
            objective = "os"

        target_energy = self.ga_tgt_en_spb.value()
        target_margin = self.ga_tgt_en_margin_spb.value()

        resume = self.ga_resume_chkbx.isChecked()
        csv_path = self.ga_csv_path_txt.text().strip()

        if resume:
            if not csv_path or not os.path.exists(csv_path):
                QMessageBox.warning(
                    self,
                    "Aviso",
                    "Por favor, selecione um arquivo CSV válido antes de continuar uma otimização."
                )
                return

        parallel = self.ga_parallel_chkbx.isChecked()
        n_workers = self.ga_parallel_cores_spb.value() if parallel else 1

        seeds = []
        if self.ga_use_seeds_chkbx.isChecked():
            raw_seeds = self.ga_seeds_txt.toPlainText().strip()
            if raw_seeds:
                seeds = parse_seeds_text(raw_seeds, n_params)
                if not seeds:
                    QMessageBox.warning(
                        self,
                        "Aviso - Formato de Sementes",
                        f"Nenhuma semente válida foi detectada para {n_params} parâmetros.\n\n"
                        f"Certifique-se de que cada semente contenha exatamente {n_params} valores numéricos "
                        f"(exemplo: [20, 70, 25] ou um indivíduo por linha)."
                    )
                    return

        optimizer_kwargs = {
            "n_params": n_params,
            "pop_size": pop_size,
            "generations": generations,
            "elitism_count": elitism,
            "cx_prob": cx_prob,
            "mut_prob": mut_prob,
            "cx_type": cx_type,
            "objective": objective,
            "opt_pc": opt_pc,
            "opt_os": opt_os,
            "limit_energy": limit_energy,
            "target_energy": target_energy,
            "target_margin": target_margin,
            "parallel": parallel,
            "n_workers": n_workers,
            "seeds": seeds,
            "csv_path": csv_path if resume else None,
            "resume": resume,
        }

        self.ga_run_btn.setEnabled(False)
        self.ga_stop_btn.setEnabled(True)
        self.ga_progress_bar.setValue(0)
        self.ga_status_lbl.setText("Iniciando GA (DEAP) e preparando simulações Fortran...")
        self.SetSimulatorState("running", "GA DEAP")
        self.LogMessage(f"🚀 Otimização GA iniciada: {pop_size} indivíduos, {generations} gerações ({n_workers} núcleos).", "running")

        self.ga_worker = GAWorker(optimizer_kwargs)
        self.ga_worker.sig_generation_finished.connect(self.OnGAGenerationFinished)
        self.ga_worker.sig_individual_evaluated.connect(self.OnGAIndividualEvaluated)
        self.ga_worker.sig_status.connect(self.OnGAStatus)
        self.ga_worker.sig_finished.connect(self.OnGAFinished)
        self.ga_worker.start()

    def OnGAStopClicked(self):
        """Requests cancellation of ongoing GA execution."""
        if self.ga_worker and self.ga_worker.isRunning():
            self.ga_worker.stop()
            self.ga_status_lbl.setText("Parando otimização... Aguardando a simulação atual terminar.")
            self.LogMessage("🛑 Cancelamento do GA solicitado. Finalizando simulação atual...", "warning")
            self.ga_stop_btn.setEnabled(False)

    def OnGAGenerationFinished(self, gen: int, total_gens: int, best_ind: list, best_fit: float, mean_fit: float, csv_path: str, best_res: dict = None):
        """Slot called whenever a generation finishes in the background thread."""
        pct = int((gen / max(1, total_gens)) * 100)
        self.ga_progress_bar.setValue(min(100, pct))
        self.sim_status_progress.setValue(min(100, pct))
        self.ga_csv_path_txt.setText(csv_path)

        self.ga_best_individual = best_ind
        self.ga_best_fitness = best_fit

        # If best_res is empty, try to get from CSV
        if not best_res and csv_path and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if not df.empty and "fitness" in df.columns:
                    best_row = df.loc[df["fitness"].idxmax()]
                    best_res = {
                        "pc1": best_row.get("PC max 1", 0.0),
                        "pc1e": best_row.get("PC max 1 energy (meV)", 0.0),
                        "oscstr": best_row.get("OscStr max", 0.0),
                        "oscstre": best_row.get("OscStr energy (meV)", 0.0),
                    }
            except Exception:
                pass

        # Format structure representation
        struct = set_structure_values(best_ind)
        w_l, qw_l, qb_l, mqw, w_r, qw_r, qb_r = struct

        pc_str = ""
        os_str = ""
        if best_res:
            try:
                pc1 = float(best_res.get("pc1", 0.0))
                pc1e = float(best_res.get("pc1e", 0.0))
                pc_str = f"Fotocorrente (PC Máx): {pc1:.4e} (Pico em {pc1e:.1f} meV)"
            except Exception:
                pc_str = f"Fotocorrente (PC Máx): {best_res.get('pc1', '')} (Pico em {best_res.get('pc1e', '')} meV)"
            try:
                oscstr = float(best_res.get("oscstr", 0.0))
                oscstre = float(best_res.get("oscstre", 0.0))
                os_str = f"Força de Oscilador (OS Máx): {oscstr:.4f} (Transição em {oscstre:.1f} meV)"
            except Exception:
                os_str = f"Força de Oscilador (OS Máx): {best_res.get('oscstr', '')} (Transição em {best_res.get('oscstre', '')} meV)"

        summary_text = (
            f"Geração {gen}/{total_gens} Concluída | Melhor Fitness: {best_fit:.4f} | Média: {mean_fit:.4f}\n"
            f"Estrutura: {w_l:.0f}x{qw_l:.1f}_{qb_l:.1f} __{mqw:.1f}__ {w_r:.0f}x{qw_r:.1f}_{qb_r:.1f}\n"
            f"{pc_str}\n"
            f"{os_str}\n"
            f"Indivíduo (Genes): {best_ind}"
        )
        self.ga_best_summary_text = summary_text
        self.ga_best_individual = best_ind
        self.ga_best_fitness = best_fit

        if getattr(self, "ga_selected_individual", None) is None:
            self._UpdateGASummaryDisplay(selected_row=None)
            self.PlotGAIndividual(best_ind, title_suffix=" (Melhor da Geração)")
        else:
            self._UpdateGASummaryDisplay(selected_row=getattr(self, "ga_selected_row", None))

        self.ga_apply_to_fortran_btn.setEnabled(True)
        if hasattr(self, "ga_show_best_btn"):
            self.ga_show_best_btn.setEnabled(True)

        self.PlotGAProgress(csv_path)
        self.UpdateFortranSamplesTable()
        self.LogMessage(f"🧬 GA Geração {gen}/{total_gens} concluída | Melhor Aptidão: {best_fit:.4f} | Média: {mean_fit:.4f}", "info")

    def OnGAIndividualEvaluated(self, ind_idx: int, total_inds: int, ind: list, fit: float):
        """Slot called whenever an individual evaluation finishes."""
        msg = f"Avaliando indivíduo {ind_idx}/{total_inds} | Estrutura {ind} | Aptidão: {fit:.3f}..."
        self.ga_status_lbl.setText(msg)
        self.sim_active_op_lbl.setText(f"[GA: Ind {ind_idx}/{total_inds}]")

    def OnGAStatus(self, msg: str):
        """Slot called for general status messages."""
        self.ga_status_lbl.setText(msg)
        self.LogMessage(msg, "info")

    def OnGAFinished(self, success: bool, message: str, best_ind: list, best_fit: float, csv_path: str, best_res: dict = None):
        """Slot called when GAWorker finishes."""
        self.ga_run_btn.setEnabled(True)
        self.ga_stop_btn.setEnabled(False)
        self.SetSimulatorState("idle")
        if success:
            self.ga_progress_bar.setValue(100)
            self.sim_status_progress.setValue(100)

        # Build detailed status message
        extra_status = ""
        if best_res:
            try:
                pc1 = float(best_res.get("pc1", 0.0))
                pc1e = float(best_res.get("pc1e", 0.0))
                oscstr = float(best_res.get("oscstr", 0.0))
                oscstre = float(best_res.get("oscstre", 0.0))
                extra_status = f" | PC Máx: {pc1:.4e} ({pc1e:.1f} meV) | OS Máx: {oscstr:.4f} ({oscstre:.1f} meV)"
            except Exception:
                pass

        full_msg = message + extra_status
        self.ga_status_lbl.setText(full_msg)
        self.LogMessage(f"🎉 {full_msg}", "success" if success else "warning")
        if csv_path:
            self.PlotGAProgress(csv_path)
            self.ga_csv_path_txt.setText(csv_path)

        if best_ind:
            self.ga_best_individual = best_ind
            self.ga_best_fitness = best_fit
            self.ga_apply_to_fortran_btn.setEnabled(True)
            if hasattr(self, "ga_show_best_btn"):
                self.ga_show_best_btn.setEnabled(True)
            if getattr(self, "ga_selected_individual", None) is None:
                self.PlotGAIndividual(best_ind, title_suffix=" (Melhor Global)")

        self.UpdateFortranSamplesTable()

    def OnGAApplyToFortran(self):
        """Applies selected (or best) GA individual to Fortran Sim tab spinboxes and updates plots."""
        ind = getattr(self, "ga_selected_individual", None) or getattr(self, "ga_best_individual", None)
        if not ind:
            return

        struct = set_structure_values(ind)
        w_l, qw_l, qb_l, mqw, w_r, qw_r, qb_r = struct

        self.fortran_s0_spb.blockSignals(True)
        self.fortran_s1_spb.blockSignals(True)
        self.fortran_s2_spb.blockSignals(True)
        self.fortran_s3_spb.blockSignals(True)
        self.fortran_s4_spb.blockSignals(True)
        self.fortran_s5_spb.blockSignals(True)
        self.fortran_s6_spb.blockSignals(True)

        self.fortran_s0_spb.setValue(int(w_l))
        self.fortran_s1_spb.setValue(float(qw_l))
        self.fortran_s2_spb.setValue(float(qb_l))
        self.fortran_s3_spb.setValue(float(mqw))
        self.fortran_s4_spb.setValue(int(w_r))
        self.fortran_s5_spb.setValue(float(qw_r))
        self.fortran_s6_spb.setValue(float(qb_r))

        self.fortran_s0_spb.blockSignals(False)
        self.fortran_s1_spb.blockSignals(False)
        self.fortran_s2_spb.blockSignals(False)
        self.fortran_s3_spb.blockSignals(False)
        self.fortran_s4_spb.blockSignals(False)
        self.fortran_s5_spb.blockSignals(False)
        self.fortran_s6_spb.blockSignals(False)

        if hasattr(self, "fortran_tab"):
            self.tabWidget.setCurrentWidget(self.fortran_tab)
        self.PlotFortranBandStructure()
        QMessageBox.information(
            self,
            "Sucesso",
            f"Estrutura aplicada na aba Fortran Sim com sucesso:\n"
            f"{int(w_l)}x{qw_l:.1f}_{qb_l:.1f} __{mqw:.1f}__ {int(w_r)}x{qw_r:.1f}_{qb_r:.1f}"
        )

    # Automate Tab Methods ############################################################
    def InitializeAutomateTab(self):
        """Initializes the Automate tab for batch/sweep simulations with multiprocessing."""
        if hasattr(self, "auto_splitter"):
            self.auto_splitter.setStretchFactor(0, 2)
            self.auto_splitter.setStretchFactor(1, 3)

        # Configure CPU cores spinbox
        total_cores = os.cpu_count() or 4
        if hasattr(self, "auto_cores_spb"):
            self.auto_cores_spb.setMinimum(1)
            self.auto_cores_spb.setMaximum(total_cores)
            default_cores = max(1, total_cores - 2) if total_cores > 4 else total_cores
            self.auto_cores_spb.setValue(default_cores)
            self.auto_cores_spb.setToolTip(f"Número de núcleos para simulações simultâneas (Total detectado: {total_cores})")

        # Configure results table
        self.auto_column_names = [
            "LLQW\n(Nº Left)",
            "LQW\n(Left nm)",
            "LQB\n(Left Bar nm)",
            "MQW\n(Main nm)",
            "RLQW\n(Nº Right)",
            "RQW\n(Right nm)",
            "RQB\n(Right Bar nm)",
            "PC Max\n(Pico)",
            "PC Peak E\n(meV)",
            "OS Max\n(Força)",
            "OS Trans E\n(meV)"
        ]
        if hasattr(self, "auto_results_table"):
            self.auto_table_filters = {}
            auto_hdr = ExcelFilterHeaderView(Qt.Horizontal, self.auto_results_table)
            auto_hdr.setMinimumHeight(44)
            auto_hdr.setSortIndicatorShown(True)
            auto_hdr.setSectionsClickable(True)
            auto_hdr.setFilterCallback(self.OpenAutoExcelFilterDialogForCol)
            self.auto_results_table.setHorizontalHeader(auto_hdr)
            self.auto_results_table.setColumnCount(len(self.auto_column_names))
            self.auto_results_table.setHorizontalHeaderLabels(self.auto_column_names)
            self.auto_results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.auto_results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.auto_results_table.setSortingEnabled(True)
            self.auto_results_table.horizontalHeader().setToolTip(
                "Clique para ordenar | Clique na seta [▼] ou com o botão direito para menu de filtros tipo Excel."
            )
            self.auto_results_table.horizontalHeader().sectionDoubleClicked.connect(self.OpenAutoExcelFilterDialogForCol)
            self.auto_results_table.itemSelectionChanged.connect(self.OnAutoTableSelectionChanged)

        # Set descriptive labels and connect signals for all 7 parameters
        param_labels = [
            "s0: LLQW (Nº Poços Esquerdos)",
            "s1: LQW (Largura Poço Esq. - nm)",
            "s2: LQB (Barreira Esquerda - nm)",
            "s3: MQW (Poço Central/Main - nm)",
            "s4: RLQW (Nº Poços Direitos)",
            "s5: RQW (Largura Poço Dir. - nm)",
            "s6: RQB (Barreira Direita - nm)",
        ]
        for i in range(7):
            lbl = getattr(self, f"auto_lbl_{i}", None)
            if lbl:
                lbl.setText(param_labels[i])
            cbox = getattr(self, f"auto_mode_{i}_cbox", None)
            if cbox:
                try:
                    cbox.currentIndexChanged.disconnect()
                except Exception:
                    pass
                cbox.currentIndexChanged.connect(self._MakeAutoModeHandler(i))

            for spb_name in [f"auto_fixed_{i}_spb", f"auto_start_{i}_spb", f"auto_end_{i}_spb", f"auto_step_{i}_spb"]:
                spb = getattr(self, spb_name, None)
                if spb:
                    spb.valueChanged.connect(self.UpdateAutoEstimate)

            self.OnAutoParamModeChanged(i)

        # Connect action buttons
        if hasattr(self, "auto_copy_fortran_btn"):
            try:
                self.auto_copy_fortran_btn.clicked.disconnect()
            except Exception:
                pass
            self.auto_copy_fortran_btn.clicked.connect(self.OnAutoCopyFortranClicked)
        if hasattr(self, "auto_run_btn"):
            try:
                self.auto_run_btn.clicked.disconnect()
            except Exception:
                pass
            self.auto_run_btn.clicked.connect(self.OnAutoRunClicked)
            self.auto_run_btn.setEnabled(True)
        if hasattr(self, "auto_stop_btn"):
            try:
                self.auto_stop_btn.clicked.disconnect()
            except Exception:
                pass
            self.auto_stop_btn.clicked.connect(self.OnAutoStopClicked)
            self.auto_stop_btn.setEnabled(False)
        if hasattr(self, "auto_apply_to_fortran_btn"):
            try:
                self.auto_apply_to_fortran_btn.clicked.disconnect()
            except Exception:
                pass
            self.auto_apply_to_fortran_btn.clicked.connect(self.OnAutoApplyToFortranClicked)
        if hasattr(self, "auto_export_csv_btn"):
            try:
                self.auto_export_csv_btn.clicked.disconnect()
            except Exception:
                pass
            self.auto_export_csv_btn.clicked.connect(self.OnAutoExportCSVClicked)
        if hasattr(self, "auto_parallel_chkbx"):
            self.auto_parallel_chkbx.toggled.connect(
                lambda chk: self.auto_cores_spb.setEnabled(chk) if hasattr(self, "auto_cores_spb") else None
            )

        self.UpdateAutoEstimate()

    def _MakeAutoModeHandler(self, idx: int):
        return lambda _: self.OnAutoParamModeChanged(idx)

    def OnAutoParamModeChanged(self, i: int):
        """Shows/hides controls depending on whether parameter i is fixed or sweep (varredura)."""
        cbox = getattr(self, f"auto_mode_{i}_cbox", None)
        if not cbox:
            return
        is_sweep = (cbox.currentIndex() == 1)
        fix_lbl = getattr(self, f"auto_fix_lbl_{i}", None)
        fix_spb = getattr(self, f"auto_fixed_{i}_spb", None)
        swp_lbl = getattr(self, f"auto_swp_lbl_{i}", None)
        start_spb = getattr(self, f"auto_start_{i}_spb", None)
        end_spb = getattr(self, f"auto_end_{i}_spb", None)
        step_spb = getattr(self, f"auto_step_{i}_spb", None)

        if fix_lbl: fix_lbl.setVisible(not is_sweep)
        if fix_spb: fix_spb.setVisible(not is_sweep)
        if swp_lbl: swp_lbl.setVisible(is_sweep)
        if start_spb: start_spb.setVisible(is_sweep)
        if end_spb: end_spb.setVisible(is_sweep)
        if step_spb: step_spb.setVisible(is_sweep)

        self.UpdateAutoEstimate()

    def _GetAutoParamValues(self, i: int):
        """Returns the list of values to simulate for parameter i."""
        cbox = getattr(self, f"auto_mode_{i}_cbox", None)
        is_sweep = cbox and (cbox.currentIndex() == 1)
        is_int = (i in (0, 4))
        if not is_sweep:
            spb = getattr(self, f"auto_fixed_{i}_spb", None)
            val = int(round(spb.value())) if is_int else round(float(spb.value()), 2) if spb else (5 if is_int else 6.0)
            return [val]
        else:
            s_spb = getattr(self, f"auto_start_{i}_spb", None)
            e_spb = getattr(self, f"auto_end_{i}_spb", None)
            st_spb = getattr(self, f"auto_step_{i}_spb", None)
            start = float(s_spb.value()) if s_spb else (1.0 if is_int else 4.0)
            end = float(e_spb.value()) if e_spb else (5.0 if is_int else 10.0)
            step = float(st_spb.value()) if st_spb else (1.0 if is_int else 0.5)

            if step <= 0:
                step = 1.0 if is_int else 0.1

            if is_int:
                istart, iend, istep = int(round(start)), int(round(end)), max(1, int(round(step)))
                if iend < istart:
                    vals = list(range(istart, iend - 1, -istep))
                else:
                    vals = list(range(istart, iend + 1, istep))
                return vals if vals else [istart]
            else:
                if abs(end - start) < 1e-6:
                    return [round(start, 2)]
                direction = 1.0 if end >= start else -1.0
                vals = []
                cur = start
                for _ in range(10000):
                    vals.append(round(cur, 2))
                    cur += direction * step
                    if direction > 0 and cur > end + 1e-6:
                        break
                    elif direction < 0 and cur < end - 1e-6:
                        break
                return vals if vals else [round(start, 2)]

    def UpdateAutoEstimate(self):
        """Recalculates points for each parameter and updates the total estimate badge."""
        total_sims = 1
        for i in range(7):
            vals = self._GetAutoParamValues(i)
            pts_lbl = getattr(self, f"auto_pts_{i}_lbl", None)
            if pts_lbl:
                cbox = getattr(self, f"auto_mode_{i}_cbox", None)
                if cbox and cbox.currentIndex() == 0:
                    pts_lbl.setText("1 valor fixo")
                else:
                    pts_lbl.setText(f"{len(vals)} valores")
            total_sims *= max(1, len(vals))

        if hasattr(self, "auto_estimate_lbl"):
            if total_sims > 5000:
                self.auto_estimate_lbl.setText(f"Total estimado: {total_sims:,} simulações (Atenção: lote grande)")
                self.auto_estimate_lbl.setStyleSheet("color: #b45309; font-weight: bold;")
            else:
                self.auto_estimate_lbl.setText(f"Total estimado: {total_sims:,} simulações")
                self.auto_estimate_lbl.setStyleSheet("color: #1e40af; font-weight: bold;")

    def OnAutoCopyFortranClicked(self):
        """Copies the current 7 structural parameters from the Fortran Sim tab spinboxes into Automate."""
        for i in range(7):
            spb_src = getattr(self, f"fortran_s{i}_spb", None)
            if not spb_src:
                continue
            val = spb_src.value()
            fixed_spb = getattr(self, f"auto_fixed_{i}_spb", None)
            start_spb = getattr(self, f"auto_start_{i}_spb", None)
            if fixed_spb:
                fixed_spb.setValue(val)
            if start_spb:
                start_spb.setValue(val)
        self.UpdateAutoEstimate()
        if hasattr(self, "auto_status_lbl"):
            self.auto_status_lbl.setText("Parâmetros copiados da aba Fortran Sim com sucesso.")

    def OnAutoTableSelectionChanged(self):
        """Enables the apply button when a row is selected in the batch results table."""
        if hasattr(self, "auto_results_table") and hasattr(self, "auto_apply_to_fortran_btn"):
            sel = self.auto_results_table.selectionModel().selectedRows()
            self.auto_apply_to_fortran_btn.setEnabled(len(sel) > 0)

    def OpenAutoExcelFilterDialogForCol(self, col: int, global_pt: QPoint = None):
        """Opens Excel-style filter dropdown for auto_results_table at col."""
        if col < 0 or col >= len(self.auto_column_names):
            return

        col_name = self.auto_column_names[col]
        popup = ExcelColumnFilterPopup(
            parent_win=self,
            table=self.auto_results_table,
            col_idx=col,
            col_name=col_name,
            active_filters=self.auto_table_filters,
            on_applied_callback=self.ApplyAutoFilters
        )

        hdr = self.auto_results_table.horizontalHeader()
        if global_pt is None:
            vx = hdr.sectionViewportPosition(col)
            global_pt = hdr.viewport().mapToGlobal(QPoint(vx, hdr.height()))

        screen = QApplication.desktop().availableGeometry(global_pt)
        popup.adjustSize()
        pw = max(270, popup.sizeHint().width())
        ph = max(400, popup.sizeHint().height())
        x = global_pt.x()
        y = global_pt.y()
        if x + pw > screen.right():
            x = max(screen.left(), screen.right() - pw - 10)
        if y + ph > screen.bottom():
            y = max(screen.top(), y - ph - hdr.height() - 10)

        popup.move(x, y)
        popup.exec_()

    def ApplyAutoFilters(self):
        """Applies active filters to auto_results_table rows."""
        total = self.auto_results_table.rowCount()
        for r in range(total):
            visible = True
            for col_idx, cond in self.auto_table_filters.items():
                item = self.auto_results_table.item(r, col_idx)
                if not item:
                    continue
                txt = item.text().strip()
                raw = getattr(item, "sort_val", None)

                num_op = cond.get("num_op", "none")
                if num_op != "none" and raw is not None:
                    try:
                        val = float(raw)
                        target = cond.get("num_val", 0.0)
                        if num_op == ">" and not (val > target): visible = False; break
                        elif num_op == ">=" and not (val >= target): visible = False; break
                        elif num_op == "<" and not (val < target): visible = False; break
                        elif num_op == "<=" and not (val <= target): visible = False; break
                        elif num_op == "==" and not (abs(val - target) < 1e-4): visible = False; break
                        elif num_op == "range":
                            v_min = cond.get("min", -1e9)
                            v_max = cond.get("max", 1e9)
                            if not (v_min <= val <= v_max): visible = False; break
                    except (ValueError, TypeError): pass

                if cond.get("checked") is not None and txt not in cond["checked"]:
                    visible = False; break

            self.auto_results_table.setRowHidden(r, not visible)

        hdr = self.auto_results_table.horizontalHeader()
        if hasattr(hdr, "setActiveFilters"):
            hdr.setActiveFilters(self.auto_table_filters)

    def OnAutoRunClicked(self):
        """Starts batch simulation sweep in background worker thread."""
        if self.auto_worker and self.auto_worker.isRunning():
            QMessageBox.warning(self, "Aviso", "Já existe uma varredura em execução.")
            return

        grid = [self._GetAutoParamValues(i) for i in range(7)]
        total = 1
        for g in grid:
            total *= len(g)

        if total == 0:
            QMessageBox.warning(self, "Aviso", "Nenhuma simulação a executar.")
            return

        if total > 2000:
            reply = QMessageBox.question(
                self,
                "Confirmar Execução de Lote",
                f"Você está prestes a executar {total:,} simulações em lote.\n"
                "Isso pode levar bastante tempo. Deseja prosseguir?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        parallel = self.auto_parallel_chkbx.isChecked() if hasattr(self, "auto_parallel_chkbx") else True
        n_cores = self.auto_cores_spb.value() if hasattr(self, "auto_cores_spb") else 4

        self.auto_run_btn.setEnabled(False)
        self.auto_stop_btn.setEnabled(True)
        self.auto_progress_bar.setValue(0)
        self.sim_status_progress.setValue(0)
        self.auto_results_data = []
        if hasattr(self, "auto_results_table"):
            self.auto_results_table.setSortingEnabled(False)
            self.auto_results_table.setRowCount(0)

        self.SetSimulatorState("running", f"Varredura: 0/{total}")
        self.LogMessage(f"🚀 Varredura iniciada! {total:,} simulações agendadas ({n_cores if parallel else 1} núcleos).", "running")
        self.auto_status_lbl.setText(f"Iniciando {total:,} simulações com {n_cores if parallel else 1} núcleo(s)...")

        self.auto_worker = AutomateWorker(
            param_grid=grid,
            parallel=parallel,
            n_cores=n_cores,
            output_folder=output_fortran_folder
        )
        self.auto_worker.sig_sample_started.connect(self.OnAutoSampleStarted)
        self.auto_worker.sig_progress.connect(self.OnAutoProgress)
        self.auto_worker.sig_status.connect(lambda msg: self.LogMessage(msg, "info"))
        self.auto_worker.sig_error.connect(lambda err: self.LogMessage(err, "error"))
        self.auto_worker.sig_finished.connect(self.OnAutoFinished)
        self.auto_worker.start()

    def OnAutoSampleStarted(self, idx: int, total: int, desc: str):
        """Slot called immediately before a simulation starts running."""
        self.sim_active_op_lbl.setText(f"[Varredura: {idx}/{total}]")
        self.LogMessage(f"⏳ Executando simulação {idx}/{total}: {desc}... Aguarde (processando no Fortran)", "running")
        self.auto_status_lbl.setText(f"⏳ Executando simulação {idx}/{total}: {desc}...")

    def OnAutoStopClicked(self):
        """Requests cancellation of ongoing batch simulations."""
        if self.auto_worker and self.auto_worker.isRunning():
            self.auto_worker.stop()
            self.auto_status_lbl.setText("Parando simulações... Aguardando a simulação em andamento concluir.")
            self.LogMessage("🛑 Solicitação de parada recebida. Finalizando simulação em andamento...", "warning")
            self.auto_stop_btn.setEnabled(False)

    def OnAutoProgress(self, completed: int, total: int, res: dict, msg: str):
        """Slot called whenever a simulation finishes in the background thread."""
        pct = int(min(100, (completed / max(1, total)) * 100))
        self.auto_progress_bar.setValue(pct)
        self.sim_status_progress.setValue(pct)
        self.sim_active_op_lbl.setText(f"[Varredura: {completed}/{total}]")
        self.auto_status_lbl.setText(f"Progresso: {completed}/{total} simulações ({pct}%) | {msg}")
        self.auto_results_data.append(res)

        self.LogMessage(
            f"✅ [{completed}/{total}] {res.get('desc', '')} concluída em {res.get('time', 0.0):.1f}s | "
            f"PC: {res.get('pc_max', 0.0):.2e} | OS: {res.get('os_max', 0.0):.4f}",
            "success"
        )

        if hasattr(self, "auto_results_table"):
            row = self.auto_results_table.rowCount()
            self.auto_results_table.insertRow(row)

            cols = [
                res["s0"], res["s1"], res["s2"], res["s3"],
                res["s4"], res["s5"], res["s6"],
                res.get("pc_max", 0.0), res.get("pc_e", 0.0),
                res.get("os_max", 0.0), res.get("os_e", 0.0)
            ]
            for c, val in enumerate(cols):
                if c in (0, 4):
                    txt = str(int(val))
                elif c in (1, 2, 3, 5, 6, 8, 10):
                    txt = f"{float(val):.1f}"
                elif c == 7:  # PC Max
                    txt = f"{float(val):.4e}"
                else:  # OS Max
                    txt = f"{float(val):.4f}"

                item = NumericTableWidgetItem(txt, sort_val=val)
                item.setTextAlignment(Qt.AlignCenter)
                item.setData(Qt.UserRole, res)
                self.auto_results_table.setItem(row, c, item)

            self.auto_results_table.scrollToBottom()

    def OnAutoFinished(self, completed: int, total_elapsed: float = 0.0):
        """Slot called when batch simulation finishes."""
        self.auto_run_btn.setEnabled(True)
        self.auto_stop_btn.setEnabled(False)
        self.SetSimulatorState("idle")
        if hasattr(self, "auto_results_table"):
            self.auto_results_table.setSortingEnabled(True)
            self.auto_results_table.resizeColumnsToContents()
        self.auto_status_lbl.setText(f"Varredura concluída! {completed} simulações foram executadas em {total_elapsed:.1f}s.")
        self.LogMessage(f"🎉 Varredura finalizada! {completed} simulações concluídas com sucesso em {total_elapsed:.1f}s.", "success")
        self.UpdateFortranSamplesTable()
        QMessageBox.information(
            self,
            "Varredura Concluída",
            f"Foram concluídas {completed} simulações em {total_elapsed:.1f}s com sucesso!\n"
            "Os resultados estão listados na tabela e integrados ao banco de simulações."
        )

    def OnAutoApplyToFortranClicked(self):
        """Applies parameters of the selected simulation in batch table to Fortran Sim tab."""
        if not hasattr(self, "auto_results_table"):
            return
        sel = self.auto_results_table.selectionModel().selectedRows()
        if not sel:
            QMessageBox.warning(self, "Aviso", "Selecione uma simulação na tabela de resultados para aplicar.")
            return
        r = sel[0].row()
        item0 = self.auto_results_table.item(r, 0)
        res = item0.data(Qt.UserRole) if item0 else None
        if not res and r < len(self.auto_results_data):
            res = self.auto_results_data[r]
        if res:
            self.fortran_s0_spb.setValue(int(res["s0"]))
            self.fortran_s1_spb.setValue(float(res["s1"]))
            self.fortran_s2_spb.setValue(float(res["s2"]))
            self.fortran_s3_spb.setValue(float(res["s3"]))
            self.fortran_s4_spb.setValue(int(res["s4"]))
            self.fortran_s5_spb.setValue(float(res["s5"]))
            self.fortran_s6_spb.setValue(float(res["s6"]))
            if hasattr(self, "fortran_tab"):
                self.tabWidget.setCurrentWidget(self.fortran_tab)
            self.PlotFortranBandStructure()
            QMessageBox.information(
                self,
                "Estrutura Aplicada",
                f"Estrutura aplicada na aba Fortran Sim com sucesso:\n"
                f"{int(res['s0'])}x{res['s1']:.1f}_{res['s2']:.1f} __{res['s3']:.1f}__ {int(res['s4'])}x{res['s5']:.1f}_{res['s6']:.1f}"
            )

    def OnAutoExportCSVClicked(self):
        """Exports the batch results table to a CSV file."""
        if not self.auto_results_data:
            QMessageBox.warning(self, "Aviso", "Não há resultados para exportar.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exportar Resultados da Varredura", "", "Arquivos CSV (*.csv);;Todos os Arquivos (*)")
        if path:
            try:
                df = pd.DataFrame(self.auto_results_data)
                df.to_csv(path, index=False)
                QMessageBox.information(self, "Sucesso", f"Resultados exportados com sucesso para:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao exportar CSV:\n{str(e)}")

    # Legacy functions #################################################################
    def Campo(self):
        """
        Função que aplica o campo elétrico à estrutura no momento que o usuário aperta o botão
        apply.
        """
        l = self.sim.self.sim.x_graf
        # trazendo o grafico para o centro da estrutura e colocando em nm
        l = l - l[-1] / 2  # / 1E-9
        self.sim.v_graf = (
            self.sim.v_graf - self.sim_Efield_spb.value() * 1.0e-4 * l
        )  # 1E5 * NM
        self.sim.e_field = self.sim.e_field + self.sim_Efield_spb.value()
        self.label_campo.setText(str(self.sim.e_field) + " kV/cm")
        # So altero o vetor posicao ao clicar a primeira vez (quando x0 = 0). Depois disso, o x ja
        # esta definido
        if self.sim.x0 == 0:
            self.sim.x0 = self.sim.x0 - np.sum(self.sim.estrutura) / 2
            # este valor vai ser usado no calculo quando usarmos a funcao run.

        v = self.sim.v_graf
        self.sim.self.sim.x_graf = l

        self.UpdateSimGraph()

    # Configuration and Documentation Handlers #######################################
    def SetupConfigMenu(self):
        """
        Creates 'Configurações' menu in the main menubar placed after 'File' and before 'Ajuda/Help'.
        """
        if hasattr(self, "menubar"):
            self.menu_config = QMenu("Configurações", self)
            if hasattr(self, "menu_help"):
                self.menubar.insertMenu(self.menu_help.menuAction(), self.menu_config)
            else:
                self.menubar.addMenu(self.menu_config)

            act_paths = QAction("⚙️ Configurar Pastas e Executável Fortran...", self)
            act_paths.setStatusTip("Define a pasta das simulações e o executável Fortran (.exe)")
            act_paths.triggered.connect(self.OpenConfigPathsDialog)
            self.menu_config.addAction(act_paths)

            self.menu_config.addSeparator()

            act_folder = QAction("📁 Selecionar Pasta de Simulações (Output Folder)...", self)
            act_folder.setStatusTip("Altera a pasta onde as simulações Fortran são salvas e lidas")
            act_folder.triggered.connect(self.OnBrowseOutputFolder)
            self.menu_config.addAction(act_folder)

            act_exe = QAction("⚡ Selecionar Executável Fortran (.exe)...", self)
            act_exe.setStatusTip("Seleciona o executável de simulação (.exe)")
            act_exe.triggered.connect(self.OnBrowseFortranExe)
            self.menu_config.addAction(act_exe)

            self.menu_config.addSeparator()

            act_reset = QAction("🔄 Restaurar Caminhos Padrão do Repositório", self)
            act_reset.setStatusTip("Restaura os caminhos relativos ao diretório do repositório")
            act_reset.triggered.connect(self.ResetPathsToRepoDefaults)
            self.menu_config.addAction(act_reset)

    def SetupHelpMenu(self):
        """
        Configures Help menu with rich Portuguese documentation.
        """
        if hasattr(self, "menu_help"):
            self.action_documentation = QAction("📖 Manual e Documentação do E-mulate...", self)
            self.action_documentation.setShortcut(QKeySequence("F1"))
            self.action_documentation.triggered.connect(self.OpenHelpDocumentationDialog)
            if hasattr(self, "action_about"):
                self.menu_help.insertAction(self.action_about, self.action_documentation)
                self.menu_help.insertSeparator(self.action_about)
            else:
                self.menu_help.addAction(self.action_documentation)

    def SyncOutputFolderDisplays(self, folder_path=None):
        """
        Sincroniza todos os controles da interface que exibem a pasta de saída
        (tanto a barra inferior 'output_folder_line' quanto a barra da aba Fortran 'output_folder_line_edit')
        com o caminho atual configurado.
        """
        import conf
        if folder_path is None:
            folder_path = getattr(conf, "output_fortran_folder", "")
        clean = os.path.normpath(str(folder_path).strip().strip('"').strip("'")).replace("\\", "/")
        if not clean.endswith("/"):
            clean += "/"

        if hasattr(self, "output_folder_line") and self.output_folder_line is not None:
            self.output_folder_line.setText(clean)
        if hasattr(self, "output_folder_line_edit") and self.output_folder_line_edit is not None:
            self.output_folder_line_edit.setText(clean)

    def OnBrowseOutputFolder(self):
        """Opens folder picker to select Fortran simulation database directory."""
        import conf
        cur = getattr(conf, "output_fortran_folder", "")
        chosen = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Simulações Fortran", cur)
        if chosen:
            clean = os.path.normpath(chosen).replace("\\", "/")
            if not clean.endswith("/"):
                clean += "/"
            conf.save_conf_paths(new_output_folder=clean)
            self.SyncOutputFolderDisplays(clean)
            self.UpdateFortranSamplesTable()
            self.LogMessage(f"📁 Pasta de simulações alterada para: {clean}", "info")

    def OnOutputFolderReturnPressed(self):
        """Triggered when the user presses Enter in output_folder_line_edit."""
        import conf
        txt = self.output_folder_line_edit.text().strip().strip('"').strip("'")
        norm_txt = os.path.normpath(txt)
        if os.path.isdir(norm_txt):
            clean = norm_txt.replace("\\", "/")
            if not clean.endswith("/"):
                clean += "/"
            conf.save_conf_paths(new_output_folder=clean)
            self.SyncOutputFolderDisplays(clean)
            self.UpdateFortranSamplesTable()
            self.LogMessage(f"📁 Pasta de simulações atualizada: {clean}", "info")
        else:
            QMessageBox.warning(self, "Pasta Inválida", f"O diretório especificado não existe:\n{txt}")
            self.SyncOutputFolderDisplays(getattr(conf, "output_fortran_folder", ""))

    def OnBrowseFortranExe(self):
        """Opens file picker to select Fortran .exe executable."""
        import conf
        cur = getattr(conf, "file_ed_exe", "")
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Selecionar Executável Fortran (.exe)", cur, "Executáveis (*.exe);;Todos os arquivos (*.*)"
        )
        if chosen:
            clean = os.path.normpath(chosen).replace("\\", "/")
            conf.save_conf_paths(new_exe_path=clean)
            self.LogMessage(f"⚡ Executável Fortran alterado para: {clean}", "info")
            QMessageBox.information(self, "Executável Configurado", f"Executável Fortran atualizado para:\n{clean}")

    def ResetPathsToRepoDefaults(self):
        """Resets both output_fortran_folder and file_ed_exe to relative repository defaults."""
        import conf
        base_dir = os.path.dirname(os.path.abspath(conf.__file__))
        default_folder = os.path.abspath(os.path.join(base_dir, "windows_executable", "temp_database")).replace("\\", "/") + "/"
        default_exe = os.path.abspath(os.path.join(base_dir, "windows_executable", "photocurrent_sim_windows.exe")).replace("\\", "/")
        conf.save_conf_paths(new_output_folder=default_folder, new_exe_path=default_exe)
        self.SyncOutputFolderDisplays(default_folder)
        self.UpdateFortranSamplesTable()
        self.LogMessage("🔄 Caminhos restaurados para os padrões relativos ao repositório.", "success")
        QMessageBox.information(
            self, "Configurações Restauradas",
            f"Os caminhos foram restaurados para o repositório local:\n\n"
            f"• Pasta de simulações:\n  {default_folder}\n\n"
            f"• Executável Fortran:\n  {default_exe}"
        )


    def OpenConfigPathsDialog(self):
        """Opens ConfigPathsDialog to configure folders and executable."""
        dlg = ConfigPathsDialog(self)
        dlg.exec_()

    def OpenHelpDocumentationDialog(self):
        """Opens HelpDocumentationDialog with rich Portuguese documentation."""
        dlg = HelpDocumentationDialog(self)
        dlg.exec_()

    def closeEvent(self, event):
        """Cleans up background workers upon closing the application."""
        if getattr(self, "auto_worker", None) and self.auto_worker.isRunning():
            self.auto_worker.stop()
            self.auto_worker.wait(1000)
        if getattr(self, "ga_worker", None) and self.ga_worker.isRunning():
            self.ga_worker.stop()
            self.ga_worker.wait(1000)
        if getattr(self, "fortran_worker", None) and self.fortran_worker.isRunning():
            self.fortran_worker.wait(1000)
        event.accept()


class ConfigPathsDialog(QDialog):
    """
    Dialog to configure simulation output directory and Fortran .exe executable.
    Persists selections to conf.py and updates runtime state immediately.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configurações - Pastas e Executável Fortran")
        self.resize(720, 440)
        self.setStyleSheet("""
            QDialog {
                background-color: #f8fafc;
                font-family: 'Segoe UI', system-ui, sans-serif;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                color: #1e293b;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 14px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
            }
            QLineEdit {
                border: 1px solid #cbd5e1;
                border-radius: 5px;
                padding: 6px 10px;
                background-color: #f8fafc;
                font-size: 12px;
                color: #334155;
            }
            QLineEdit:focus {
                border-color: #3b82f6;
                background-color: #ffffff;
            }
            QPushButton {
                border-radius: 5px;
                padding: 6px 14px;
                font-weight: 600;
                font-size: 11px;
            }
        """)

        import conf
        self.conf = conf
        self.parent_window = parent

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(18, 18, 18, 18)

        # Header Title
        title_box = QHBoxLayout()
        icon_lbl = QLabel("⚙️")
        icon_lbl.setStyleSheet("font-size: 24px;")
        header_text = QVBoxLayout()
        h_title = QLabel("Configuração de Ambiente e Executável Fortran")
        h_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #0f172a;")
        h_desc = QLabel("Defina a pasta onde ficam salvas as simulações e o caminho para o arquivo .exe do simulador.")
        h_desc.setStyleSheet("font-size: 11px; color: #64748b;")
        header_text.addWidget(h_title)
        header_text.addWidget(h_desc)
        title_box.addWidget(icon_lbl)
        title_box.addLayout(header_text)
        title_box.addStretch()
        layout.addLayout(title_box)

        # Group 1: Output Folder
        grp_folder = QGroupBox("Pasta de Simulações (Output Folder)")
        g_fol_lay = QVBoxLayout(grp_folder)
        g_fol_lay.setSpacing(6)

        fol_h = QHBoxLayout()
        self.txt_folder = QLineEdit(self.conf.output_fortran_folder)
        self.txt_folder.textChanged.connect(self.ValidatePaths)
        btn_browse_folder = QPushButton("📁 Procurar...")
        btn_browse_folder.setStyleSheet("background-color: #2563eb; color: white; border: none;")
        btn_browse_folder.clicked.connect(self.OnBrowseFolder)
        fol_h.addWidget(self.txt_folder, stretch=1)
        fol_h.addWidget(btn_browse_folder)
        g_fol_lay.addLayout(fol_h)

        self.lbl_folder_status = QLabel("")
        self.lbl_folder_status.setStyleSheet("font-size: 11px;")
        g_fol_lay.addWidget(self.lbl_folder_status)
        layout.addWidget(grp_folder)

        # Group 2: Executable File
        grp_exe = QGroupBox("Executável do Simulador Fortran (.exe)")
        g_exe_lay = QVBoxLayout(grp_exe)
        g_exe_lay.setSpacing(6)

        exe_h = QHBoxLayout()
        self.txt_exe = QLineEdit(self.conf.file_ed_exe)
        self.txt_exe.textChanged.connect(self.ValidatePaths)
        btn_browse_exe = QPushButton("⚡ Procurar...")
        btn_browse_exe.setStyleSheet("background-color: #2563eb; color: white; border: none;")
        btn_browse_exe.clicked.connect(self.OnBrowseExe)
        btn_repo_exe = QPushButton("↺ Usar do Repositório")
        btn_repo_exe.setStyleSheet("background-color: #e2e8f0; color: #1e293b; border: 1px solid #cbd5e1;")
        btn_repo_exe.setToolTip("Restaura o caminho do executável para photocurrent_sim_windows.exe na pasta do repositório.")
        btn_repo_exe.clicked.connect(self.OnUseRepoExe)

        exe_h.addWidget(self.txt_exe, stretch=1)
        exe_h.addWidget(btn_browse_exe)
        exe_h.addWidget(btn_repo_exe)
        g_exe_lay.addLayout(exe_h)

        self.lbl_exe_status = QLabel("")
        self.lbl_exe_status.setStyleSheet("font-size: 11px;")
        g_exe_lay.addWidget(self.lbl_exe_status)
        layout.addWidget(grp_exe)

        # Bottom Buttons
        btn_box = QHBoxLayout()
        btn_reset_all = QPushButton("🔄 Restaurar Padrões do Repositório")
        btn_reset_all.setStyleSheet("background-color: #f1f5f9; color: #475569; border: 1px solid #cbd5e1;")
        btn_reset_all.clicked.connect(self.OnResetAll)

        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setStyleSheet("background-color: #f1f5f9; color: #1e293b; border: 1px solid #cbd5e1;")
        btn_cancel.clicked.connect(self.reject)

        btn_save = QPushButton("💾 Salvar e Atualizar")
        btn_save.setStyleSheet("background-color: #16a34a; color: white; border: none; padding: 6px 18px;")
        btn_save.clicked.connect(self.OnSave)

        btn_box.addWidget(btn_reset_all)
        btn_box.addStretch()
        btn_box.addWidget(btn_cancel)
        btn_box.addWidget(btn_save)
        layout.addLayout(btn_box)

        self.ValidatePaths()

    def ValidatePaths(self):
        from conf import resolve_simulation_paths
        f_path = self.txt_folder.text().strip().strip('"').strip("'")
        if f_path and os.path.isdir(os.path.normpath(f_path)):
            db_dir, sim_dir, pkl_dir, _ = resolve_simulation_paths(f_path)
            if sim_dir and os.path.isdir(sim_dir):
                cnt = len([d for d in os.listdir(sim_dir) if os.path.isdir(os.path.join(sim_dir, d))])
                sub_label = os.path.basename(sim_dir)
                self.lbl_folder_status.setText(f"✓ Pasta válida: {cnt} simulações detectadas (em '{sub_label}').")
                self.lbl_folder_status.setStyleSheet("color: #16a34a; font-weight: 600; font-size: 11px;")
            else:
                self.lbl_folder_status.setText("⚠ Pasta existe, mas ainda não possui subpastas de simulação.")
                self.lbl_folder_status.setStyleSheet("color: #d97706; font-weight: 600; font-size: 11px;")
        else:
            self.lbl_folder_status.setText("❌ Pasta não encontrada no sistema de arquivos.")
            self.lbl_folder_status.setStyleSheet("color: #dc2626; font-weight: 600; font-size: 11px;")

        e_path = os.path.normpath(self.txt_exe.text().strip().strip('"').strip("'"))
        if os.path.isfile(e_path):
            self.lbl_exe_status.setText("✓ Arquivo executável (.exe) válido e pronto para uso.")
            self.lbl_exe_status.setStyleSheet("color: #16a34a; font-weight: 600; font-size: 11px;")
        else:
            self.lbl_exe_status.setText("❌ Arquivo executável não encontrado no caminho especificado.")
            self.lbl_exe_status.setStyleSheet("color: #dc2626; font-weight: 600; font-size: 11px;")

    def OnBrowseFolder(self):
        chosen = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Simulações", self.txt_folder.text().strip())
        if chosen:
            clean = os.path.normpath(chosen).replace("\\", "/")
            if not clean.endswith("/"): clean += "/"
            self.txt_folder.setText(clean)

    def OnBrowseExe(self):
        chosen, _ = QFileDialog.getOpenFileName(self, "Selecionar Executável Fortran", self.txt_exe.text().strip(), "Executáveis (*.exe);;Todos (*.*)")
        if chosen:
            clean = os.path.normpath(chosen).replace("\\", "/")
            self.txt_exe.setText(clean)

    def OnUseRepoExe(self):
        base_dir = os.path.dirname(os.path.abspath(self.conf.__file__))
        repo_exe = os.path.abspath(os.path.join(base_dir, "windows_executable", "photocurrent_sim_windows.exe")).replace("\\", "/")
        self.txt_exe.setText(repo_exe)

    def OnResetAll(self):
        base_dir = os.path.dirname(os.path.abspath(self.conf.__file__))
        default_folder = os.path.abspath(os.path.join(base_dir, "windows_executable", "temp_database")).replace("\\", "/") + "/"
        default_exe = os.path.abspath(os.path.join(base_dir, "windows_executable", "photocurrent_sim_windows.exe")).replace("\\", "/")
        self.txt_folder.setText(default_folder)
        self.txt_exe.setText(default_exe)

    def OnSave(self):
        f_path = self.txt_folder.text().strip().strip('"').strip("'")
        e_path = self.txt_exe.text().strip().strip('"').strip("'")

        clean_f = os.path.normpath(f_path).replace("\\", "/")
        if clean_f and not clean_f.endswith("/"):
            clean_f += "/"
        clean_e = os.path.normpath(e_path).replace("\\", "/")

        self.conf.save_conf_paths(new_output_folder=clean_f, new_exe_path=clean_e)

        if self.parent_window:
            self.parent_window.SyncOutputFolderDisplays(clean_f)
            self.parent_window.UpdateFortranSamplesTable()
            self.parent_window.LogMessage(f"Configurações atualizadas: Pasta='{clean_f}', Executável='{clean_e}'", "success")

        QMessageBox.information(
            self, "Sucesso",
            f"Configurações gravadas com sucesso no conf.py e atualizadas em memória!\n\n"
            f"• Pasta de simulações:\n  {clean_f}\n\n"
            f"• Executável:\n  {clean_e}"
        )
        self.accept()



class HelpDocumentationDialog(QDialog):
    """
    Rich documentation manual window in Portuguese explaining all software features:
    Run Fortran, Genetic Algorithm (GA), Automate parameter sweeps, Excel filters, and interactive plots.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📖 Manual e Documentação de Funcionalidades - E-mulate")
        self.resize(1000, 720)
        self.setStyleSheet("""
            QDialog {
                background-color: #f8fafc;
                font-family: 'Segoe UI', system-ui, sans-serif;
            }
            QListWidget {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 6px;
                font-size: 13px;
                color: #1e293b;
            }
            QListWidget::item {
                padding: 10px 12px;
                border-radius: 6px;
                margin-bottom: 4px;
            }
            QListWidget::item:hover {
                background-color: #f1f5f9;
            }
            QListWidget::item:selected {
                background-color: #2563eb;
                color: #ffffff;
                font-weight: bold;
            }
            QTextBrowser {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 16px;
                color: #1e293b;
                font-size: 13px;
                line-height: 1.6;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header bar with search
        top_bar = QHBoxLayout()
        header_lbl = QLabel("📖 Manual de Instruções e Documentação do E-mulate")
        header_lbl.setStyleSheet("font-size: 17px; font-weight: bold; color: #0f172a;")
        top_bar.addWidget(header_lbl)
        top_bar.addStretch()

        btn_close = QPushButton("Fechar")
        btn_close.setStyleSheet("background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 16px; font-weight: bold;")
        btn_close.clicked.connect(self.close)
        top_bar.addWidget(btn_close)
        layout.addLayout(top_bar)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(6)

        # Navigation list
        self.nav_list = QListWidget()
        self.nav_list.setFixedWidth(260)
        nav_items = [
            ("📘 1. Visão Geral", "intro"),
            ("⚡ 2. Run Fortran (Simulação)", "fortran"),
            ("🧬 3. Algoritmo Genético (GA)", "ga"),
            ("🔄 4. Automate (Varredura)", "automate"),
            ("🔍 5. Filtros e Gráficos", "filters"),
            ("⚙️ 6. Pastas e Portabilidade", "config"),
        ]
        for label, tag in nav_items:
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, tag)
            self.nav_list.addItem(item)

        splitter.addWidget(self.nav_list)

        # Browser
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        splitter.addWidget(self.browser)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        layout.addWidget(splitter, stretch=1)

        self.BuildDocumentationContent()
        self.nav_list.currentRowChanged.connect(self.OnSectionChanged)
        self.nav_list.setCurrentRow(0)

    def BuildDocumentationContent(self):
        css_style = """
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; color: #1e293b; line-height: 1.6; }
            h1 { color: #1e40af; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; font-size: 20px; margin-top: 0px; }
            h2 { color: #0f172a; font-size: 16px; margin-top: 18px; margin-bottom: 6px; border-left: 4px solid #3b82f6; padding-left: 8px; }
            h3 { color: #334155; font-size: 14px; margin-top: 12px; margin-bottom: 4px; }
            p { margin-top: 4px; margin-bottom: 8px; font-size: 13px; }
            ul, ol { margin-top: 4px; margin-bottom: 8px; padding-left: 20px; }
            li { margin-bottom: 4px; }
            table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 12px; }
            th, td { border: 1px solid #cbd5e1; padding: 6px 10px; text-align: left; }
            th { background-color: #f1f5f9; font-weight: bold; color: #0f172a; }
            tr:nth-child(even) { background-color: #f8fafc; }
            .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; background-color: #dbeafe; color: #1e40af; }
            .badge-green { background-color: #dcfce7; color: #15803d; }
            .badge-amber { background-color: #fef3c7; color: #b45309; }
            .note { background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 10px 14px; margin: 10px 0; border-radius: 4px; }
            .tip { background-color: #f0fdf4; border-left: 4px solid #22c55e; padding: 10px 14px; margin: 10px 0; border-radius: 4px; }
            .warn { background-color: #fffbeb; border-left: 4px solid #f59e0b; padding: 10px 14px; margin: 10px 0; border-radius: 4px; }
            code { background-color: #f1f5f9; padding: 2px 5px; border-radius: 3px; font-family: Consolas, monospace; font-size: 12px; color: #0f172a; }
        </style>
        """

        self.sections = {
            "intro": css_style + """
            <h1>📘 1. Visão Geral do E-mulate</h1>
            <p>O <b>E-mulate</b> é um ambiente integrado para simulação, análise e otimização bioinspirada de <b>heteroestruturas quânticas de semicondutores</b> (poços quânticos acoplados e super-redes com defeitos), voltado para dispositivos optoeletrônicos de infravermelho médio e distante (tais como <i>QWIPs</i> - Quantum Well Infrared Photodetectors e <i>QCDs</i> - Quantum Cascade Detectors).</p>

            <h2>Principais Funcionalidades</h2>
            <ul>
                <li><b>Simulador Fortran Individual (Run Fortran):</b> Cálculo de autoestados quânticos, funções de onda de envelope, força de oscilador intersubbanda e espectro de fotocorrente física absoluta (em Amperes).</li>
                <li><b>Algoritmo Genético (GA):</b> Otimização evolutiva multivariada (3, 5 ou 7 parâmetros estruturais) com seleção contínua de aptidão (fitness), restrição física estrutural e foco em janelas de energia de transição óptica (padrão 300 ± 20 meV).</li>
                <li><b>Varredura Automatizada (Automate):</b> Avaliação sistemática em grade (grid search) combinando parâmetros fixos e variáveis com aceleração por multiprocessamento paralelo.</li>
                <li><b>Tabelas Inteligentes com Filtros tipo Excel:</b> Ordenação A-Z/Z-A, filtragem numérica avançada (&gt;, &lt;, ==, faixa de valores) e seleção por caixas de seleção (checkboxes) com busca instantânea.</li>
                <li><b>Gráficos Interativos Integrados:</b> Dispersão X vs. Y interativa sincronizada com a tabela (o clique em qualquer ponto do gráfico seleciona e destaca a linha na tabela).</li>
                <li><b>Configuração Dinâmica e Portabilidade:</b> Configuração rápida da pasta de dados e binário <code>.exe</code>, permitindo copiar a pasta do projeto para qualquer máquina sem perder os caminhos.</li>
            </ul>
            <div class="note"><b>Dica de Navegação:</b> Utilize o menu lateral esquerdo para navegar pelas seções e detalhamentos de cada ferramenta.</div>
            """,

            "fortran": css_style + r"""
            <h1>⚡ 2. Run Fortran (Simulação de Super-rede)</h1>
            <p>A aba <b>Fortran Sim</b> realiza a simulação detalhada de uma heteroestrutura quântica completa através da execução do módulo compilado Fortran (<code>photocurrent_sim_windows.exe</code>).</p>

            <h2>Parâmetros Estruturais da Super-rede</h2>
            <p>A heteroestrutura é modelada através de 7 dimensões geométricas principais:</p>
            <table>
                <tr><th>Parâmetro</th><th>Nome</th><th>Descrição</th><th>Unidade / Faixa</th></tr>
                <tr><td><b>LLQW</b></td><td>Lateral Left QWs</td><td>Número de poços quânticos no bloco periódico esquerdo</td><td>Inteiro (ex: 5)</td></tr>
                <tr><td><b>LQW</b></td><td>Left Quantum Well</td><td>Espessura de cada poço quântico do bloco esquerdo</td><td>nm (ex: 2.0 nm)</td></tr>
                <tr><td><b>LQB</b></td><td>Left Quantum Barrier</td><td>Espessura de cada barreira do bloco esquerdo</td><td>nm (ex: 7.0 nm)</td></tr>
                <tr><td><b>MQW</b></td><td>Main Quantum Well</td><td>Espessura do poço quântico central / poço de defeito</td><td>nm (ex: 2.5 nm)</td></tr>
                <tr><td><b>RLQW</b></td><td>Lateral Right QWs</td><td>Número de poços quânticos no bloco periódico direito</td><td>Inteiro (ex: 1)</td></tr>
                <tr><td><b>RQW</b></td><td>Right Quantum Well</td><td>Espessura de cada poço quântico do bloco direito</td><td>nm (ex: 2.0 nm)</td></tr>
                <tr><td><b>RQB</b></td><td>Right Quantum Barrier</td><td>Espessura de cada barreira do bloco direito</td><td>nm (ex: 7.0 nm)</td></tr>
            </table>

            <h2>Opções de Execução</h2>
            <ul>
                <li><b>Force Parser:</b> Quando marcado, força o simulador a reler e reprocessar os arquivos de texto do disco mesmo se o programa já possuir os dados parseados.</li>
                <li><b>Force Simulation:</b> Quando desmarcado, se a pasta correspondente à estrutura já possuir o arquivo <code>FimPrograma.txt</code>, o resultado existente é carregado instantaneamente do disco sem reexecutar o cálculo numérico. Se marcado, reexecuta o binário do zero.</li>
                <li><b>Run Fortran Simulation:</b> Executa a simulação em segundo plano via thread dedicada sem travar a interface, exibindo tempo decorrido e registrando eventos na barra de status.</li>
            </ul>

            <h2>Os 3 Gráficos da Simulação</h2>
            <ol>
                <li><b>Potencial e Funções de Onda:</b> Exibe o perfil do potencial eletrostático ao longo do eixo de crescimento $z$, com as energias dos autoestados confinados e o quadrado das funções de onda de envelope $|\psi_i(z)|^2$.</li>
                <li><b>Força de Oscilador (OscStr):</b> Mostra a intensidade do acoplamento dipolar óptico para cada nível de transição a partir do estado fundamental ($0 \to i$).</li>
                <li><b>Fotocorrente (PC):</b> Espectro contínuo de fotocorrente física absoluta em <b>Amperes (A)</b> em função da energia do fóton incidente (meV), sem divisão por referências relativas arbitrárias.</li>
            </ol>

            <h2>Tabela de Amostras Simuladas</h2>
            <p>Localizada no painel direito da aba, lista todas as simulações encontradas no banco de dados com valores de pico físico absoluto (<code>PC Max</code>, <code>PC Peak E</code>, <code>OS Max</code>, <code>OS Trans E</code>). Clique sobre qualquer linha para carregar os parâmetros nos seletores e atualizar os gráficos instantaneamente.</p>
            """,

            "ga": css_style + r"""
            <h1>🧬 3. Algoritmo Genético (GA / DEAP)</h1>
            <p>O módulo de Algoritmo Genético emprega o framework bioinspirado <b>DEAP</b> para explorar automaticamente o vasto espaço dimensional de parâmetros da heteroestrutura e descobrir configurações que maximizem o desempenho óptico e elétrico.</p>

            <h2>Espaço de Parâmetros (Genoma)</h2>
            <ul>
                <li><b>3 Parâmetros:</b> Otimiza as espessuras <code>LQW</code>, <code>LQB</code> e <code>MQW</code>. Os blocos esquerdo e direito são simétricos, com número de poços fixo em 5 e 1.</li>
                <li><b>5 Parâmetros:</b> Otimiza <code>LLQW</code>, <code>RLQW</code>, além das espessuras <code>LQW</code>, <code>LQB</code> e <code>MQW</code> (espessuras simétricas, número de poços variável).</li>
                <li><b>7 Parâmetros:</b> Otimização assimétrica completa de todos os 7 parâmetros estruturais.</li>
            </ul>

            <div class="warn">
                <b>Restrição Estrutural Física Mandatória (Defeito Central):</b><br/>
                Para formar o estado de aprisionamento dominante, o poço quântico principal central deve ser estritamente maior que os poços laterais:<br/>
                <code>MQW &ge; max(LQW, RQW) + 0.5 nm</code><br/>
                O gerador de indivíduos e os operadores genéticos filtram e penalizam severamente qualquer estrutura que viole essa condição.
            </div>

            <h2>Objetivos de Otimização e Função de Aptidão (Fitness)</h2>
            <ul>
                <li><b>Fotocorrente (PC):</b> Maximiza o valor de pico da fotocorrente absoluta através da escala logarítmica $(\log_{10}(PC) \times 4) + 50$.</li>
                <li><b>Força de Oscilador (OS):</b> Maximiza a força de oscilador óptica através do termo $OS \times 20$.</li>
                <li><b>Ambos (OS + PC):</b> Combina absorção óptica e extração de fotocorrente simultaneamente.</li>
            </ul>

            <h2>Janela de Energia Alvo e Política de Penalização Contínua</h2>
            <ul>
                <li><b>Energia Alvo (Target Energy):</b> Padrão <b>300.0 meV</b> (correspondente a ~4.13 &mu;m na faixa de infravermelho médio).</li>
                <li><b>Margem de Tolerância:</b> Padrão <b>20.0 meV</b> (define a faixa desejada de 280.0 a 320.0 meV).</li>
                <li><b>Penalização Suave (Sem Descarte Abrupto):</b> Soluções que apresentem pico ligeiramente fora da borda (por exemplo, 320.05 meV) <b>não são descartadas sumariamente</b>! Em vez de eliminar uma estrutura de excelente intensidade, é aplicada uma penalização suave e contínua proporcional à distância da borda, garantindo gradiente genético estável para guiar as mutações de volta à faixa desejada.</li>
            </ul>

            <h2>Parâmetros Evolutivos</h2>
            <table>
                <tr><th>Parâmetro</th><th>Padrão</th><th>Descrição</th></tr>
                <tr><td>Tamanho da População</td><td>20</td><td>Quantidade de indivíduos avaliados por geração.</td></tr>
                <tr><td>Gerações</td><td>15</td><td>Número total de ciclos evolutivos de seleção e recombinação.</td></tr>
                <tr><td>Elitismo</td><td>2</td><td>Número de melhores indivíduos clonados intactos para a próxima geração.</td></tr>
                <tr><td>Crossover (Probabilidade)</td><td>80%</td><td>Frequência com que indivíduos sofrem recombinação (Two-Point, Uniform, Blend).</td></tr>
                <tr><td>Mutação (Probabilidade)</td><td>20%</td><td>Frequência com que genes sofrem perturbações aleatórias delimitadas.</td></tr>
            </table>

            <h2>Recursos Avançados</h2>
            <ul>
                <li><b>Multiprocessamento Paralelo:</b> Permite selecionar o número de núcleos da CPU para paralelizar o cálculo das simulações de cada geração.</li>
                <li><b>Sementes (Seeds):</b> Permite fornecer estruturas conhecidas na primeira geração ou capturar a estrutura da aba Fortran Sim clicando no botão <i>"Adicionar Estrutura Atual"</i>.</li>
                <li><b>Retomar Otimização (CSV):</b> Continua uma otimização interrompida a partir de um arquivo CSV anterior.</li>
                <li><b>Inspeção Interativa por Clique:</b> Clique em qualquer ponto do gráfico de evolução para visualizar no painel inferior o texto formatado, a estrutura e as curvas individuais de fotocorrente e força de oscilador do indivíduo selecionado.</li>
                <li><b>Carregar na Aba Fortran Sim:</b> Transfere os parâmetros da solução otimizada com 1 clique para inspeção profunda na aba de simulação individual.</li>
            </ul>
            """,

            "automate": css_style + r"""
            <h1>🔄 4. Automate (Varredura de Parâmetros)</h1>
            <p>A aba <b>Automate</b> permite executar varreduras em grade (<i>Grid Search</i>) para investigar sistematicamente a influência de um ou múltiplos parâmetros estruturais sobre o desempenho do dispositivo.</p>

            <h2>Configuração da Varredura</h2>
            <p>Para cada um dos 7 parâmetros da heteroestrutura, é possível selecionar se ele será <b>Fixo</b> ou <b>Variável</b>:</p>
            <ul>
                <li><b>Parâmetro Fixo:</b> Mantém um único valor constante ao longo de todo o lote.</li>
                <li><b>Parâmetro Variável:</b> Define o valor <b>Inicial</b>, <b>Final</b> e o <b>Passo (Step)</b> para percorrer uma faixa de valores.</li>
            </ul>
            <p>O software calcula e exibe em tempo real a <b>estimativa do número total de simulações</b> ($N_1 \times N_2 \times \dots$), permitindo planejar o tempo computacional antes do início.</p>

            <h2>Aceleração e Execução</h2>
            <ul>
                <li><b>Multiprocessamento Paralelo:</b> Habilite o checkbox <i>"Execução Paralela"</i> e selecione quantos núcleos de CPU deseja alocar.</li>
                <li><b>Monitoramento em Tempo Real:</b> Barra de progresso com porcentagem, tempo decorrido, taxa de processamento e logs em tempo real na barra de atividades inferior.</li>
                <li><b>Botão Cancelar:</b> Permite interromper a varredura a qualquer momento sem perder os resultados já obtidos.</li>
            </ul>

            <h2>Tabela de Resultados e Exportação</h2>
            <p>Ao término de cada ponto, os resultados (estruturas, <code>PC Max</code>, <code>PC Peak E</code>, <code>OS Max</code>, <code>OS Trans E</code> e tempo) são inseridos na tabela com filtros tipo Excel e podem ser exportados diretamente para arquivo CSV via botão <b>"📁 Exportar CSV"</b>.</p>
            """,

            "filters": css_style + """
            <h1>🔍 5. Filtros tipo Excel e Gráficos Interativos</h1>
            <p>Tanto a tabela de amostras simuladas do Fortran quanto a tabela da varredura automatizada contam com um sistema avançado de filtragem e ordenação inspirado no Microsoft Excel.</p>

            <h2>Como Utilizar os Filtros de Coluna</h2>
            <ol>
                <li>Clique no botão <b>[ ▼ ]</b> localizado no cabeçalho de qualquer coluna (ou clique com o botão direito / duplo-clique no cabeçalho).</li>
                <li>No menu pop-up que se abrir, você pode:
                    <ul>
                        <li><b>Ordenar:</b> Selecione <i>"Ordenar A → Z (Crescente)"</i> ou <i>"Ordenar Z → A (Decrescente)"</i>.</li>
                        <li><b>Filtro Numérico:</b> Escolha entre <code>Maior que (&gt;)</code>, <code>Maior ou Igual (&gt;=)</code>, <code>Menor que (&lt;)</code>, <code>Menor ou Igual (&lt;=)</code>, <code>Igual (==)</code> ou <code>Entre (Faixa Min-Max)</code>.</li>
                        <li><b>Filtro por Valores Únicos:</b> Marque ou desmarque valores na lista de caixas de seleção (checkboxes). Utilize o campo de busca rápida para filtrar rapidamente itens específicos.</li>
                    </ul>
                </li>
                <li>Clique em <b>"Aplicar"</b>. A tabela atualizará imediatamente as linhas visíveis e o cabeçalho exibirá um indicador visual de filtro ativo <b>[ 🔻 ]</b> em destaque verde.</li>
                <li>Para remover o filtro, abra o menu da coluna e clique em <i>"Limpar Filtro desta Coluna"</i> ou utilize o botão <b>"Limpar Filtros"</b>.</li>
            </ol>

            <h2>Gráfico de Dispersão Interativo (Scatter Plot)</h2>
            <ul>
                <li>Abaixo da tabela, você pode selecionar livremente quais colunas deseja plotar nos <b>Eixos X e Y</b> (por exemplo, <i>MQW vs PC Max</i> ou <i>OS Max vs PC Peak E</i>).</li>
                <li>O gráfico é atualizado dinamicamente respeitando apenas as linhas atualmente filtradas na tabela.</li>
                <li><b>Interatividade Bidirecional por Clique:</b> Clique diretamente com o mouse sobre qualquer ponto azul do gráfico para selecionar e destacar automaticamente a linha correspondente na tabela! O ponto selecionado receberá um anel vermelho de destaque no gráfico.</li>
                <li><b>Barra de Ferramentas Matplotlib:</b> Utilize os botões de Lupa (Zoom), Mão (Pan/Arrastar) e Disco (Salvar Figura em alta resolução).</li>
            </ul>
            """,

            "config": css_style + """
            <h1>⚙️ 6. Pastas e Portabilidade do Software</h1>
            <p>O E-mulate foi projetado para funcionar de forma totalmente portátil, facilitando a cópia entre diferentes computadores ou diretórios sem quebras de referências de arquivos.</p>

            <h2>Barra de Pasta de Simulações (Output Folder)</h2>
            <p>Na aba <b>Fortran Sim</b>, logo acima da tabela de amostras simuladas, está posicionada a barra <b>Pasta de Simulações</b>:</p>
            <ul>
                <li>Exibe o caminho atual onde os arquivos de simulação e o cache são lidos e gravados.</li>
                <li>Clique no botão <b>"📁 Procurar..."</b> para escolher uma nova pasta no computador. O caminho será gravado no arquivo <code>conf.py</code> e a tabela será atualizada imediatamente com os dados da nova pasta.</li>
                <li>Clique em <b>"🔄 Atualizar"</b> para recarregar o conteúdo da pasta atual a qualquer momento.</li>
            </ul>

            <h2>Menu Principal "Configurações"</h2>
            <p>No menu superior do aplicativo, entre <b>Arquivo</b> e <b>Ajuda</b>, o menu <b>Configurações</b> oferece acesso rápido a:</p>
            <ul>
                <li><b>⚙️ Configurar Pastas e Executável Fortran...:</b> Abre a janela de configuração com diagnóstico em tempo real do diretório e do arquivo <code>.exe</code>.</li>
                <li><b>📁 Selecionar Pasta de Simulações:</b> Atalho para selecionar a pasta de banco de dados.</li>
                <li><b>⚡ Selecionar Executável Fortran (.exe):</b> Permite selecionar o binário compilado.</li>
                <li><b>🔄 Restaurar Caminhos Padrão do Repositório:</b> Restaura tanto a pasta quanto o executável para os arquivos locais do repositório (<code>windows_executable/</code>), ideal caso o projeto tenha sido copiado para um novo computador.</li>
            </ul>

            <div class="tip">
                <b>Dica de Portabilidade:</b> Ao copiar a pasta do E-mulate para outro computador, o arquivo <code>conf.py</code> utiliza referências dinâmicas ao diretório onde o projeto está instalado, garantindo que o programa abra e simule de imediato sem necessidade de reconfiguração manual.
            </div>
            """
        }

    def OnSectionChanged(self, row: int):
        if row < 0: return
        item = self.nav_list.item(row)
        if item:
            tag = item.data(Qt.UserRole)
            html_text = self.sections.get(tag, "<p>Seção não encontrada.</p>")
            self.browser.setHtml(html_text)
            self.browser.verticalScrollBar().setValue(0)


class SobreWindow(QMainWindow):
    """
    About window
    """

    def __init__(self, parent=None):
        super(SobreWindow, self).__init__(parent)
        # uic.loadUi(os.path.join(os.getcwd(), "GUI", "Sobre_Gui.ui"), self)
        uic.loadUi(os.path.join(os.path.dirname(__file__), "GUI", "Sobre_Gui.ui"), self)
        self.webpage_btn.clicked.connect(self.webpage)
        self.Ok_btn.clicked.connect(self.close)

    def webpage(self):
        webbrowser.open("http://www.if.ufrj.br/~gpenello/")


class NewSimWindow(QWidget):
    """
    Window that allows the user to create a new simulation.
    """

    signal_new_title = pyqtSignal(str, str)
    signal_updated_current_number = pyqtSignal(int)

    def __init__(self, current_number, parent=None):
        super(NewSimWindow, self).__init__(parent)
        uic.loadUi(
            os.path.join(os.path.dirname(__file__), "GUI", "New_Simulation.ui"), self
        )

        self.ConnectSignals()
        # Load the information from the configuration file to the UI and updates other values
        self.InterfaceSetup(current_number)
        # Creates a timer to update the texts
        self.CreateInterfaceTimer()

    def InterfaceSetup(self, current_number):
        """
        Fills the interface with the updated values.
        """
        # Set the options in the comboboxes
        options = ["No", "Before Title", "After Title"]
        for option in options:
            self.title_number_cbox.addItem(option)
            self.title_date_cbox.addItem(option)
            self.title_time_cbox.addItem(option)
            self.title_materials_cbox.addItem(option)

        # Loads the available materias from the file
        self.mat = configparser.ConfigParser()
        self.mat.read(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "materials.data")
        )

        for pair in self.mat.sections():
            self.materials_cbox.addItem(pair)

        # Updates the current simulation number
        self.number_spb.setValue(current_number)

    def ConnectSignals(self):
        """
        Definition of the interaction between button clicks, signals and actions.
        """
        # Set the focus to the title line, so that the user can type right away
        self.title_line.setFocus()

        # The window might be closed in three ways:
        # pressing enter while the title_line is selected
        self.title_line.returnPressed.connect(self.SetTitle)
        # or pressing enter while it's selected
        self.create_simulation_btn.setDefault(True)
        # or by clicking the create_simulation button
        self.create_simulation_btn.clicked.connect(self.SetTitle)

        # When the user changes the simulation number, the main window gets a signal
        # If the value in the number spinbox was modified, update the self.current_number
        self.number_spb.valueChanged.connect(self.UpdatedCurrentNumber)

    def UpdatedCurrentNumber(self):
        self.current_number = self.number_spb.value()
        self.signal_updated_current_number.emit(self.number_spb.value())

    def CreateInterfaceTimer(self):
        """
        Creates a QTimer that regularly updates the interface, in order to keep track of the current
        time.
        """
        self.UpdateTexts()  # Runs the function that updates the title once just to keep the ui tidy
        self.txt_timer = QTimer()
        txt_update_interval = 200  # ms
        self.txt_timer.start(txt_update_interval)
        self.txt_timer.timeout.connect(self.UpdateTexts)

    def SetTitle(self):
        """
        When the user clicks the button or press Enter, this function will get the title, the
        current selection of materials and emit a signal with this information, so that the main
        window can create a new simulation.
        """
        title = self.ComposeTitle()
        materials = self.materials_cbox.currentText()
        self.signal_new_title.emit(title, materials)

    def UpdateTexts(self):
        """
        Keeps the info shown on the interface updated
        """
        self.number_spb.setValue(self.current_number)
        now = time.localtime()
        self.date_text.setText(time.strftime("%Y-%m-%d", now))
        self.time_text.setText(time.strftime("%Hh%Mm%Ss", now))
        self.materials_text.setText(self.materials_cbox.currentText())
        self.name_preview_text.setText(self.ComposeTitle())

    def ComposeTitle(self):
        """
        Function that reads the options defined on the interface and returns the title based on that
        The items are inserted in the order the comboboxes appear:
        Number Date Time Material Title Number Date Time Material
        """
        title = self.title_line.text()
        now = time.localtime()

        if self.title_materials_cbox.currentIndex() == 1:
            title = self.materials_cbox.currentText() + " " + title
        if self.title_time_cbox.currentIndex() == 1:
            title = time.strftime("%Hh%Mm%Ss", now) + " " + title
        if self.title_date_cbox.currentIndex() == 1:
            title = time.strftime("%Y-%m-%d", now) + " " + title
        if self.title_number_cbox.currentIndex() == 1:
            title = str(self.number_spb.value()) + " " + title

        # Just in case there are trailing spaces due to the title being empty
        title = title.rstrip()

        if self.title_materials_cbox.currentIndex() == 2:
            title += " " + self.materials_cbox.currentText()
        if self.title_time_cbox.currentIndex() == 2:
            title += " " + time.strftime("%Hh%Mm%Ss", now)
        if self.title_date_cbox.currentIndex() == 2:
            title += " " + time.strftime("%Y-%m-%d", now)
        if self.title_number_cbox.currentIndex() == 2:
            title += " " + str(self.number_spb.value())

        # Just in case there are leading spaces due to the title being empty
        title = title.lstrip()

        return title


class RenameSimWindow(QWidget):
    """
    Window that is shown when the user renames or copies a simulation
    """

    signal_renamed = pyqtSignal()

    def __init__(self, sim, parent=None):
        super(RenameSimWindow, self).__init__(parent)
        # uic.loadUi(os.path.join(os.getcwd(), "GUI", "Sobre_Gui.ui"), self)
        uic.loadUi(
            os.path.join(os.path.dirname(__file__), "GUI", "RenameSimulation.ui"), self
        )
        self.rename_btn.clicked.connect(lambda: self.Rename(sim))
        # Set the focus to the title line, so that the user can type right away
        self.title_line.setFocus()
        # The window might be closed in three ways:
        # pressing enter while the title_line is selected
        self.title_line.returnPressed.connect(lambda: self.Rename(sim))
        # or pressing enter while it's selected
        self.rename_btn.setDefault(True)
        # or by clicking the create_simulation button
        self.rename_btn.clicked.connect(lambda: self.Rename(sim))

        self.title_line.setText(sim.title)

    def Rename(self, sim):
        """
        Function called when the user clicks on the "rename" button. Renames the title of sim.
        """
        # Gets the string from the text box
        title = self.title_line.text()
        # Just in case there are leading spaces due to the title being empty
        sim.title = title.lstrip()
        self.signal_renamed.emit()


if __name__ == "__main__":
    # Create the GUI application
    app = QApplication(sys.argv)
    app.setStyleSheet(MODERN_APP_STYLESHEET)

    # Creating splash screen
    splash_pix = QPixmap(os.path.join("Imagens", "SplashScreen5.png"))
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()
    time.sleep(0.2)
    splash.close()
    # app.processEvents()

    # instantiate the main window
    mw = MainWindow()
    # show it
    mw.show()
    # Revert changes to matplotlib rc params
    plt.rcParams.update(plt.rcParamsDefault)
    # start the Qt main loop execution, exiting from this script
    # with the same return code of Qt application
    sys.exit(app.exec_())
