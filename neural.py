"""
Script principal para iniciar el REPL de Redes Neuronales.
Ejecuta este script directamente para iniciar el REPL con todo el contexto cargado.

Uso:
    py neural_repl.py
    o
    python neural_repl.py
"""

import sys
import os

# Agregar directorio raíz al path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Importar y ejecutar el script del REPL
from repl.repl import main

if __name__ == '__main__':
    main()
