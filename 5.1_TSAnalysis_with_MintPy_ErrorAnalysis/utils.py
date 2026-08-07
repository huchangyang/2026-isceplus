"""Utilities."""

import os
import sys


# Ensure ! commands find the conda environment's executables
os.environ['PATH'] = os.pathsep.join([os.path.dirname(sys.executable), os.environ.get('PATH', '')])

for _var, _sub in [('PROJ_DATA', 'share/proj'), ('GDAL_DATA', 'share/gdal')]:
    if _var not in os.environ:
        _path = os.path.join(sys.prefix, _sub)
        if os.path.isdir(_path):
            os.environ[_var] = _path

def get_local_path():
    """Directory containing this utils.py (used by smallbaselineApp_aria.ipynb)."""
    return os.path.dirname(os.path.realpath(__file__))


def write_config_file(out_file, CONFIG_TXT, mode='a'): 
    """Write configuration files for MintPy to process NISAR sample products"""
    if not os.path.isfile(out_file) or mode == 'w':
        with open(out_file, "w") as fid:
            fid.write(CONFIG_TXT)
        print('write configuration to file: {}'.format(out_file))
    else:
        with open(out_file, "a") as fid:
            fid.write("\n" + CONFIG_TXT)
        print('add the following to file: \n{}'.format(CONFIG_TXT))
    return
