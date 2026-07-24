from pathlib import Path
import subprocess
import sys


def test_demo_scripts_exist():
    assert Path("scripts/00_make_demo_data.py").exists()
    assert Path("scripts/02_run_landscape_flux.py").exists()
    assert Path("src/scthermoflux/thermo.py").exists()
