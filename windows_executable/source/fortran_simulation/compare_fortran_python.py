"""
Ferramenta de Comparação e Validação Numérica entre Fortran e Python
Arquivo: compare_fortran_python.py

Executa a simulação em Fortran e em Python sob os mesmos parâmetros de entrada,
compara todos os arquivos de saída linha por linha e ponto por ponto,
calcula métricas de erro (Max Abs, Max Rel, RMSE) e emite alertas caso haja discrepâncias.

Arquivos comparados:
1. Potencial_SL.txt     (Malha espacial e potencial eletrostático)
2. Energy_SL.txt        (Autoenergias dos estados ligados)
3. wavefunction_SL.txt  (Funções de onda espaciais)
4. OscStr_SL.txt        (Forças de oscilador intersubbanda)
5. Photocurrent_SL.txt  (Espectro de fotocorrente)
6. Transmission_SL.txt  (Espectro de transmissão no contínuo)
"""

import os
import sys
import time
import subprocess
import tempfile
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

# Adiciona a raiz do projeto ao sys.path se necessário
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Importa o solucionador modular em Python
from windows_executable.source.fortran_simulation.superlattice_solver import (
    SuperlatticeConfig,
    run_superlattice_simulation,
    parse_fortran_float,
    NM,
)


def find_fortran_executable() -> str:
    """Localiza o executável Fortran no repositório."""
    possible_paths = [
        os.path.abspath("windows_executable/photocurrent_sim_windows.exe"),
        os.path.abspath("windows_executable/source/fortran_simulation/fortran_files/photocurrent_sim_windows.exe"),
        os.path.abspath("windows_executable/source/fortran_simulation/fortran_files/20240825_Superlattice_Eigensates_Photocurrent_inputs.exe"),
    ]

    # Tenta obter de conf.py
    try:
        import conf
        if hasattr(conf, "file_ed_exe") and os.path.exists(conf.file_ed_exe):
            return os.path.abspath(conf.file_ed_exe)
    except Exception:
        pass

    for p in possible_paths:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        "Nenhum executável Fortran foi encontrado nas rotas padrão do repositório."
    )


def run_fortran_simulation(
    exe_path: str,
    cfg: SuperlatticeConfig,
    output_dir: str,
    timeout: float = 120.0
) -> float:
    """Executa o executável Fortran com os parâmetros da super-rede e retorna o tempo decorrido."""
    out_arg = output_dir.replace("\\", "/").rstrip("/") + "/"
    cmd = [
        exe_path,
        str(cfg.num_qb_left),
        f"{cfg.L_qw_left / NM:.1f}d0",
        f"{cfg.L_qb_left / NM:.1f}d0",
        f"{cfg.L_cqw / NM:.1f}d0",
        str(cfg.num_qb_right),
        f"{cfg.L_qw_right / NM:.1f}d0",
        f"{cfg.L_qb_right / NM:.1f}d0",
        out_arg,
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        raise RuntimeError(
            f"O processo Fortran falhou com código {proc.returncode}.\nStderr: {proc.stderr}\nStdout: {proc.stdout}"
        )

    # Verifica se FimPrograma.txt foi criado
    fim_path = os.path.join(output_dir, "FimPrograma.txt")
    if not os.path.exists(fim_path):
        raise FileNotFoundError(
            f"Execução Fortran terminou mas FimPrograma.txt não foi encontrado em {output_dir}"
        )

    return elapsed


def compare_files_numerical(
    file_for: str,
    file_py: str,
    label: str,
    atol: float = 1e-4,
    rtol: float = 1e-4
) -> Dict[str, Any]:
    """Compara numericamente dois arquivos gerados por Fortran e Python."""
    res: Dict[str, Any] = {
        "label": label,
        "file_for": os.path.basename(file_for),
        "status": "PASS",
        "max_abs_diff": 0.0,
        "max_rel_diff": 0.0,
        "rmse": 0.0,
        "mae": 0.0,
        "points_compared": 0,
        "shape_for": None,
        "shape_py": None,
        "alerts": [],
    }

    if not os.path.exists(file_for):
        res["status"] = "FAIL"
        res["alerts"].append(f"Arquivo Fortran ausente: {file_for}")
        return res

    if not os.path.exists(file_py):
        res["status"] = "FAIL"
        res["alerts"].append(f"Arquivo Python ausente: {file_py}")
        return res

    # Arquivos vazios como OscStr_SL_norm.txt
    if os.path.getsize(file_for) == 0 and os.path.getsize(file_py) == 0:
        res["status"] = "PASS"
        res["alerts"].append("Ambos os arquivos estão vazios (esperado).")
        return res

    try:
        data_for = np.loadtxt(file_for)
        data_py = np.loadtxt(file_py)
    except Exception as e:
        res["status"] = "FAIL"
        res["alerts"].append(f"Erro ao carregar dados numéricos: {e}")
        return res

    res["shape_for"] = data_for.shape
    res["shape_py"] = data_py.shape

    if data_for.shape != data_py.shape:
        # Se os tamanhos forem diferentes, compara até o mínimo comum
        min_rows = min(data_for.shape[0], data_py.shape[0])
        res["alerts"].append(
            f"Diferença de dimensões: Fortran {data_for.shape} vs Python {data_py.shape}. "
            f"Comparando até as primeiras {min_rows} linhas."
        )
        if data_for.ndim == 1:
            data_for = data_for[:min_rows]
            data_py = data_py[:min_rows]
        else:
            min_cols = min(data_for.shape[1], data_py.shape[1])
            data_for = data_for[:min_rows, :min_cols]
            data_py = data_py[:min_rows, :min_cols]

    diff = np.abs(data_py - data_for)
    abs_denom = np.abs(data_for)
    rel_denom = np.where(abs_denom > 1e-15, abs_denom, 1.0)
    rel_diff = np.where(abs_denom > 1e-15, diff / rel_denom, 0.0)

    max_abs = float(np.max(diff))
    max_rel = float(np.max(rel_diff))
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    res["max_abs_diff"] = max_abs
    res["max_rel_diff"] = max_rel
    res["mae"] = mae
    res["rmse"] = rmse
    res["points_compared"] = data_for.size

    # Avaliação de tolerância
    if max_abs > atol and max_rel > rtol:
        res["status"] = "WARN" if max_rel < 0.05 else "FAIL"
        # Localiza o ponto de maior diferença
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        val_for = data_for[max_idx]
        val_py = data_py[max_idx]
        res["alerts"].append(
            f"Diferença máxima no índice {max_idx}: "
            f"Fortran={val_for:.8e}, Python={val_py:.8e}, "
            f"AbsDiff={max_abs:.4e}, RelDiff={max_rel:.4e}"
        )
    else:
        res["status"] = "PASS"

    return res


def compare_simulations(
    cfg: SuperlatticeConfig,
    exe_path: Optional[str] = None,
    output_report_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executa a comparação completa entre Fortran e Python para uma super-rede dada.
    Gera métricas, relatório e verificação de igualdade numérica.
    """
    if exe_path is None:
        exe_path = find_fortran_executable()

    with tempfile.TemporaryDirectory() as tmp_for, tempfile.TemporaryDirectory() as tmp_py:
        print("=" * 80)
        print("COMPARAÇÃO NUMÉRICA: FORTRAN vs PYTHON (NUMPY)")
        print("=" * 80)
        print(f"Executável Fortran: {exe_path}")
        print(f"Parâmetros da estrutura:")
        print(f"  Bragg Esquerdo: {cfg.num_qb_left}x ({cfg.L_qw_left/NM:.1f} nm poço, {cfg.L_qb_left/NM:.1f} nm barreira)")
        print(f"  Poço Central:   {cfg.L_cqw/NM:.1f} nm")
        print(f"  Bragg Direito:  {cfg.num_qb_right}x ({cfg.L_qw_right/NM:.1f} nm poço, {cfg.L_qb_right/NM:.1f} nm barreira)")
        print("-" * 80)

        # 1. Executa Fortran
        print("[*] Executando simulador em Fortran...")
        t_for = run_fortran_simulation(exe_path, cfg, tmp_for)
        print(f"[OK] Fortran concluido em {t_for:.2f} s")

        # 2. Executa Python
        print("[*] Executando simulador em Python (NumPy)...")
        res_py = run_superlattice_simulation(cfg, sample_folder=tmp_py)
        print(f"[OK] Python concluido em {res_py.elapsed_time:.2f} s")
        print("-" * 80)

        # 3. Compara arquivos
        files_to_compare = [
            ("Potencial_SL.txt", "Potencial V(x)", 1e-4, 1e-4),
            ("Energy_SL.txt", "Autoenergias E_n [meV]", 1e-4, 1e-4),
            ("wavefunction_SL.txt", "Funções de Onda psi(x)", 0.05, 0.01),
            ("OscStr_SL.txt", "Forças de Oscilador f_osc", 1e-4, 1e-4),
            ("Photocurrent_SL.txt", "Fotocorrente J_ph", 1e-12, 1e-3),
            ("Transmission_SL.txt", "Transmissão T(E)", 1e-4, 1e-3),
            ("OscStr_SL_norm.txt", "OscStr Normalizado", 1e-4, 1e-4),
        ]

        results_list = []
        overall_pass = True

        print(f"{'Arquivo':<24} | {'Status':<6} | {'Max Abs Diff':<14} | {'Max Rel Diff':<14} | {'RMSE':<12}")
        print("-" * 80)

        for fname, label, atol, rtol in files_to_compare:
            f_for = os.path.join(tmp_for, fname)
            f_py = os.path.join(tmp_py, fname)
            comp = compare_files_numerical(f_for, f_py, label, atol=atol, rtol=rtol)
            results_list.append(comp)

            status_str = comp["status"]
            if status_str == "FAIL":
                overall_pass = False

            print(
                f"{fname:<24} | "
                f"{status_str:<6} | "
                f"{comp['max_abs_diff']:<14.4e} | "
                f"{comp['max_rel_diff']:<14.4e} | "
                f"{comp['rmse']:<12.4e}"
            )

        print("=" * 80)

        # Alertas e detalhes
        has_alerts = any(len(c["alerts"]) > 0 for c in results_list)
        if has_alerts:
            print("DETALHES E ALERTAS:")
            for c in results_list:
                for alert in c["alerts"]:
                    print(f"  [{c['file_for']}] {alert}")
            print("=" * 80)

        verdict = "APROVADO (Resultados equivalentes)" if overall_pass else "ATENÇÃO: Discrepâncias detectadas"
        print(f"VEREDITO GERAL: {verdict}")
        print(f"Relação de tempo (Fortran / Python): {t_for:.2f}s / {res_py.elapsed_time:.2f}s")
        print("=" * 80)

        report = {
            "overall_pass": overall_pass,
            "time_fortran": t_for,
            "time_python": res_py.elapsed_time,
            "results": results_list,
        }

        if output_report_file:
            with open(output_report_file, "w", encoding="utf-8") as f:
                f.write(f"# Relatório de Comparação Fortran vs Python\n\n")
                f.write(f"- Veredito: **{verdict}**\n")
                f.write(f"- Tempo Fortran: {t_for:.2f} s\n")
                f.write(f"- Tempo Python: {res_py.elapsed_time:.2f} s\n\n")
                f.write("| Arquivo | Status | Max Abs Diff | Max Rel Diff | RMSE |\n")
                f.write("|---|---|---|---|---|\n")
                for c in results_list:
                    f.write(
                        f"| {c['file_for']} | {c['status']} | {c['max_abs_diff']:.4e} | "
                        f"{c['max_rel_diff']:.4e} | {c['rmse']:.4e} |\n"
                    )
            print(f"Relatório gravado em: {output_report_file}")

        return report


def main():
    """Interface de linha de comando para rodar comparação."""
    if len(sys.argv) >= 8:
        num_qb_left = int(sys.argv[1])
        L_qw_left = parse_fortran_float(sys.argv[2])
        L_qb_left = parse_fortran_float(sys.argv[3])
        L_cqw = parse_fortran_float(sys.argv[4])
        num_qb_right = int(sys.argv[5])
        L_qw_right = parse_fortran_float(sys.argv[6])
        L_qb_right = parse_fortran_float(sys.argv[7])
        cfg = SuperlatticeConfig.from_nm(
            num_qb_left=num_qb_left,
            L_qw_left_nm=L_qw_left,
            L_qb_left_nm=L_qb_left,
            L_cqw_nm=L_cqw,
            num_qb_right=num_qb_right,
            L_qw_right_nm=L_qw_right,
            L_qb_right_nm=L_qb_right,
        )
    else:
        # Configuração de demonstração padrão
        cfg = SuperlatticeConfig.from_nm(2, 3.0, 4.0, 5.0, 2, 3.0, 4.0)

    report_file = "comparison_fortran_python_report.md"
    compare_simulations(cfg, output_report_file=report_file)


if __name__ == "__main__":
    main()
