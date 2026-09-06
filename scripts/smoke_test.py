#!/usr/bin/env python
"""Smoke test del entorno.

Ejecutalo como PRIMER paso, antes de cualquier otra cosa:

    uv run python scripts/smoke_test.py

Por que existe: la auditoria del repositorio encontro que la unica verificacion
de entorno era `python -V; uv --version`, que no prueba nada de lo que
realmente falla. Los diagramas eran punteros de Git LFS sin traer, el puerto de
MLflow estaba en 5000 en unos archivos y 5001 en otros, `plotly` se importaba
sin estar declarado, y en Windows la ExecutionPolicy bloqueaba el script de
setup. Cada uno de esos problemas costaba entre 10 y 40 minutos de clase.

Este script los diagnostica en unos segundos y devuelve exit code != 0 si algo
esta mal, de modo que tambien sirve como paso de CI.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

VERDE = "\033[92m"
ROJO = "\033[91m"
AMARILLO = "\033[93m"
GRIS = "\033[90m"
FIN = "\033[0m"

if os.name == "nt" and not os.getenv("WT_SESSION"):
    VERDE = ROJO = AMARILLO = GRIS = FIN = ""

# (modulo importable, nombre de distribucion, sesion en la que se usa)
PAQUETES: list[tuple[str, str, str]] = [
    ("pandas", "pandas", "datos"),
    ("numpy", "numpy", "datos"),
    ("sklearn", "scikit-learn", "modelo"),
    ("pyarrow", "pyarrow", "datos"),
    ("pandera", "pandera", "contratos"),
    ("mlflow", "mlflow", "tracking"),
    ("optuna", "optuna", "hpo"),
    ("xgboost", "xgboost", "modelo"),
    ("prefect", "prefect", "orquestacion"),
    ("fastapi", "fastapi", "serving"),
    ("pydantic", "pydantic", "serving"),
    ("evidently", "evidently", "monitoreo"),
    ("prometheus_client", "prometheus-client", "monitoreo"),
    ("plotly", "plotly", "reportes"),
    ("matplotlib", "matplotlib", "reportes"),
    ("scipy", "scipy", "monitoreo"),
    ("click", "click", "cli"),
]

PUERTOS = [
    (5001, "MLflow tracking server"),
    (4200, "Prefect server / UI"),
    (8000, "API de inferencia (FastAPI)"),
]

resultados: list[tuple[str, str, str]] = []  # (estado, titulo, detalle)


def ok(titulo: str, detalle: str = "") -> None:
    resultados.append(("OK", titulo, detalle))


def falla(titulo: str, detalle: str) -> None:
    resultados.append(("FAIL", titulo, detalle))


def aviso(titulo: str, detalle: str) -> None:
    resultados.append(("WARN", titulo, detalle))


# =============================================================================
# Verificaciones
# Comprobacion de que el entorno esta listo.
# =============================================================================
def verificar_python() -> None:
    v = sys.version_info
    detalle = f"{v.major}.{v.minor}.{v.micro} ({platform.system()} {platform.machine()})"
    if (v.major, v.minor) >= (3, 11):
        ok("Python >= 3.11", detalle)
    else:
        falla("Python >= 3.11", f"tienes {detalle}. Corre: uv python install 3.11")


def verificar_venv() -> None:
    en_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if en_venv:
        ok("Entorno virtual activo", sys.prefix)
    else:
        aviso(
            "Entorno virtual activo",
            "estas usando el Python del sistema. Usa `uv run ...` o activa .venv",
        )


def verificar_paquetes() -> None:
    faltantes: list[str] = []
    for modulo, dist, sesion in PAQUETES:
        try:
            importlib.import_module(modulo)
            try:
                version = md.version(dist)
            except md.PackageNotFoundError:
                version = "?"
            ok(f"import {modulo}", f"{version}  [{sesion}]")
        except ImportError:
            faltantes.append(dist)
            falla(f"import {modulo}", f"no instalado — requerido en {sesion}")
    if faltantes:
        falla(
            "Dependencias completas",
            "corre `uv sync --group dev`. Faltan: " + ", ".join(faltantes),
        )


def verificar_git_lfs() -> None:
    if shutil.which("git") is None:
        falla("git disponible", "instala Git antes de continuar")
        return
    ok("git disponible", "")

    if shutil.which("git-lfs") is None:
        falla(
            "git-lfs instalado",
            "los diagramas .png se versionan con Git LFS. "
            "Instala git-lfs y corre: git lfs install && git lfs pull",
        )
        return

    try:
        salida = subprocess.run(
            ["git", "lfs", "ls-files"],
            cwd=RAIZ,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        aviso("git lfs ls-files", str(exc))
        return

    lineas = [ln for ln in salida.stdout.splitlines() if ln.strip()]
    if not lineas:
        aviso("Archivos LFS traidos", "no hay archivos LFS registrados")
        return

    # Un puntero LFS sin traer pesa ~130 bytes. Si el archivo real es pequeno,
    # el diagrama no se va a ver en el README.
    punteros = []
    for linea in lineas:
        partes = linea.split(" ", 2)
        if len(partes) == 3:
            ruta = RAIZ / partes[2].strip()
            if ruta.exists() and ruta.stat().st_size < 200:
                punteros.append(partes[2].strip())
    if punteros:
        falla(
            "Archivos LFS traidos",
            f"{len(punteros)} de {len(lineas)} archivos son punteros sin descargar. "
            "Corre: git lfs install && git lfs pull",
        )
    else:
        ok("Archivos LFS traidos", f"{len(lineas)} archivos")


def verificar_herramientas() -> None:
    """Comprueba las herramientas que el repositorio NO puede instalar por si mismo.

    Este bloque existe por un problema de huevo y gallina: `make setup` instala
    todo el entorno, pero necesita que `uv` y `make` ya esten ahi. En una maquina
    recien formateada no hay ninguno de los dos, y el estudiante recibe un
    "command not found" que no explica nada.

    Este script, en cambio, corre con cualquier Python 3.11+ y sin una sola
    dependencia instalada. Por eso es el PASO 0: diagnostica la maquina antes de
    que haya entorno que diagnosticar.
    """
    # --- uv: sin esto no hay nada ---
    if shutil.which("uv") is None:
        falla(
            "uv instalado",
            "es el gestor de entorno y `make setup` no arranca sin el.\n"
            "         macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh\n"
            '         Windows:     powershell -c "irm https://astral.sh/uv/install.ps1 | iex"\n'
            "         Cierra y reabre la terminal despues de instalarlo.",
        )
    else:
        proc = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, timeout=30, check=False
        )
        ok("uv instalado", proc.stdout.strip() or "responde")

    # --- make: comodidad, no requisito ---
    if shutil.which("make") is None:
        if os.name == "nt":
            aviso(
                "make disponible",
                "no viene en Windows. NO es obligatorio: cada target del Makefile es "
                "un `uv run ...`.\n         Abre el Makefile y copia el comando, o "
                "instalalo con: winget install GnuWin32.Make",
            )
        else:
            aviso(
                "make disponible",
                "en macOS se instala con: xcode-select --install",
            )
    else:
        ok("make disponible", "los atajos del Makefile funcionan")


def verificar_hooks() -> None:
    """Verifica que los hooks de pre-commit quedaron REALMENTE instalados.

    Existe por un fallo silencioso concreto. `make setup` configuraba
    `core.hooksPath` apuntando a `.githooks/` y despues corria
    `pre-commit install`, que se niega cuando esa variable esta puesta:

        [ERROR] Cowardly refusing to install hooks with `core.hooksPath` set.

    El target terminaba sin error y el repositorio se quedaba sin ruff, sin
    gitleaks, sin nbstripout y sin los hooks propios. Nadie se
    enteraba hasta que un secreto o un notebook con outputs llegaba al
    historial.
    """
    hooks_path = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"],
        cwd=RAIZ,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    ).stdout.strip()

    if hooks_path:
        falla(
            "core.hooksPath sin configurar",
            f"apunta a '{hooks_path}'. pre-commit se niega a instalarse con eso "
            "puesto. Corre: git config --unset-all core.hooksPath && make setup",
        )
        return
    ok("core.hooksPath sin configurar", "pre-commit es el unico sistema de hooks")

    if os.getenv("CI"):
        aviso(
            "Hooks de pre-commit instalados",
            "omitido (CI: los hooks viven en la maquina de cada estudiante)",
        )
        return

    esperados = {
        "pre-commit": "ruff, gitleaks, nbstripout y los hooks propios",
        "commit-msg": "formato de conventional commits",
        "pre-push": "convencion de nombre de rama",
    }
    faltantes = [
        f"{nombre} ({para})"
        for nombre, para in esperados.items()
        if not (RAIZ / ".git" / "hooks" / nombre).exists()
    ]
    if faltantes:
        falla(
            "Hooks de pre-commit instalados",
            "faltan: " + "; ".join(faltantes) + ". Corre: make setup",
        )
    else:
        ok("Hooks de pre-commit instalados", ", ".join(esperados))


def verificar_puertos() -> None:
    for puerto, servicio in PUERTOS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.4)
            ocupado = sock.connect_ex(("127.0.0.1", puerto)) == 0
        if ocupado:
            extra = ""
            if puerto == 5000:
                extra = " (en macOS suele ser AirPlay Receiver)"
            aviso(
                f"Puerto {puerto} libre",
                f"ocupado — {servicio} no podra arrancar ahi{extra}",
            )
        else:
            ok(f"Puerto {puerto} libre", servicio)


def verificar_paquete() -> None:
    sys.path.insert(0, str(RAIZ / "src"))
    try:
        from BeijingAir import config
        from BeijingAir.data import contract as dc
        from BeijingAir.features import contract as fc
    except ImportError as exc:
        falla("import BeijingAir", f"{exc} — corre `uv sync`")
        return

    ok(
        "import beijing Air",
        f"{len(fc.FEATURES)} features, modelo '{config.MODELO_REGISTRADO}'",
    )

    # Prueba el contrato con datos sinteticos: no requiere red.
    import pandas as pd

    horas = pd.date_range("2023-01-02 08:00", periods=1200, freq="1min")
    # Por qué * 300 en cada columna: horas tiene 1200 filas;
    #: * 300 repite la lista de 4 valores hasta 1200 filas.
    #: Con 1200 filas cumples el check volumen_minimo (≥ 100) y
    #: variacion_de_target (PM2.5 no constante)
    crudo = pd.DataFrame(
        {
            fc.COL_TIEMPO: horas,
            "station": ["Aotizhongxin", "Changping", "Dingling", "Dongsi"] * 300,
            "wd": ["N", "NE", "ESE", "NNW"] * 300,
            "PM2.5": [55.0, 120.0, 30.0, 90.0] * 300,
            "PM10": [70.0, 150.0, 45.0, 110.0] * 300,
            "SO2": [8.0, 15.0, 5.0, 12.0] * 300,
            "NO2": [40.0, 60.0, 25.0, 55.0] * 300,
            "CO": [800.0, 1200.0, 400.0, 900.0] * 300,
            "O3": [60.0, 40.0, 90.0, 70.0] * 300,
            "TEMP": [5.0, 10.0, 8.0, 12.0] * 300,
            "PRES": [1015.0, 1020.0, 1018.0, 1012.0] * 300,
            "DEWP": [-5.0, 0.0, 2.0, 5.0] * 300,
            "RAIN": [0.0, 0.0, 0.0, 0.0] * 300,
            "WSPM": [3.0, 5.0, 2.0, 4.0] * 300,
        }
    )
    try:
        dc.validar_crudos(crudo)
        ok("Contrato de datos crudos", "valida un dataframe sintetico correcto")
    except Exception as exc:
        falla("Contrato de datos crudos", f"{type(exc).__name__}: {exc}")

    try:
        derivado = fc.construir_features(crudo)
        for col in (fc.COL_TEMPORADA, fc.COL_HORA, fc.COL_DIA_SEMANA, fc.COL_MES):
            assert col in derivado.columns
        ok(
            "Derivacion de features",
            f"crea {fc.COL_TEMPORADA}, {fc.COL_HORA}, {fc.COL_DIA_SEMANA} y {fc.COL_MES}",
        )
    except Exception as exc:
        falla("Derivacion de features", f"{type(exc).__name__}: {exc}")

    # El contrato debe RECHAZAR un dato roto. Un contrato que nunca falla no
    # esta protegiendo nada.
    roto = crudo.copy()
    roto["PM2.5"] = roto["PM2.5"] * 100  # ug/m3 absurdos, fuera del rango del sensor
    try:
        dc.validar_crudos(roto)
        falla(
            "Contrato rechaza datos invalidos",
            "acepto PM2.5 fuera de rango — el contrato no protege nada",
        )
    except Exception:
        ok("Contrato rechaza datos invalidos", "detecta PM2.5 fuera de rango")


def verificar_mlflow_arranca(rapido: bool) -> None:
    if rapido:
        aviso("MLflow arranca", "omitido (--rapido)")
        return
    try:
        import mlflow  # noqa: F401
    except ImportError:
        falla("MLflow arranca", "mlflow no esta instalado")
        return
    proc = subprocess.run(
        [sys.executable, "-m", "mlflow", "--version"],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    if proc.returncode == 0:
        ok("CLI de MLflow", proc.stdout.strip() or "responde")
    else:
        falla("CLI de MLflow", proc.stderr.strip()[:200])


def verificar_prefect(rapido: bool) -> None:
    if rapido:
        aviso("Prefect responde", "omitido (--rapido)")
        return
    try:
        import prefect  # noqa: F401
    except ImportError:
        falla("Prefect responde", "prefect no esta instalado")
        return
    ok("Prefect importable", md.version("prefect"))


def verificar_docker() -> None:
    if shutil.which("docker") is None:
        aviso(
            "Docker disponible",
            "necesario desde la S05. Instala Docker Desktop antes de esa sesion",
        )
        return
    proc = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if proc.returncode == 0:
        ok("Docker disponible", f"server {proc.stdout.strip()}")
    else:
        aviso("Docker disponible", "instalado pero el daemon no responde")


def verificar_estructura() -> None:
    esperados = [
        "src/BeijingAir/config.py",
        "Makefile",
        "pyproject.toml",
        ".pre-commit-config.yaml",
        "data",
        "notebooks",
        "tests",
        "scripts",
    ]
    faltan = [p for p in esperados if not (RAIZ / p).exists()]
    if faltan:
        falla("Estructura del repo", "faltan: " + ", ".join(faltan))
    else:
        ok("Estructura del repo", f"{len(esperados)} rutas clave presentes")


# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test del entorno del MLOps")
    parser.add_argument(
        "--rapido",
        action="store_true",
        help="omite las verificaciones que lanzan subprocesos lentos",
    )
    args = parser.parse_args()

    print(f"\n{GRIS}{'=' * 72}{FIN}")
    print("  SMOKE TEST — Proyecto Beijing Air")
    print(f"{GRIS}{'=' * 72}{FIN}\n")

    verificar_python()
    verificar_herramientas()
    verificar_venv()
    verificar_paquetes()
    verificar_paquete()
    verificar_estructura()
    verificar_git_lfs()
    verificar_hooks()
    verificar_puertos()
    verificar_mlflow_arranca(args.rapido)
    verificar_prefect(args.rapido)
    verificar_docker()

    ancho = max(len(t) for _, t, _ in resultados) + 2
    for estado, titulo, detalle in resultados:
        if estado == "OK":
            marca = f"{VERDE}  OK  {FIN}"
        elif estado == "WARN":
            marca = f"{AMARILLO} WARN {FIN}"
        else:
            marca = f"{ROJO} FAIL {FIN}"
        print(f"[{marca}] {titulo:<{ancho}} {GRIS}{detalle}{FIN}")

    fallos = sum(1 for e, _, _ in resultados if e == "FAIL")
    avisos = sum(1 for e, _, _ in resultados if e == "WARN")

    print(f"\n{GRIS}{'-' * 72}{FIN}")
    if fallos:
        print(f"{ROJO}{fallos} verificacion(es) FALLARON{FIN}, {avisos} aviso(s).")
        print("Arregla los FAIL antes de seguir. Cada linea dice que hacer.")
        print(f"{GRIS}{'-' * 72}{FIN}\n")
        return 1
    print(f"{VERDE}Entorno listo.{FIN} {avisos} aviso(s) — revisalos, no bloquean.")
    print(f"{GRIS}{'-' * 72}{FIN}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
