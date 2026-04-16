"""
Script de comando para iniciar el REPL de redes neuronales.
Este script puede ejecutarse directamente o usarse como alias.
"""

import sys
import os

# Agregar directorio raíz al path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    """Iniciar el REPL de redes neuronales."""
    # Intentar usar ipython si está disponible
    try:
        from IPython import start_ipython
        from IPython.terminal.embed import InteractiveShellEmbed
        
        # Cargar el contexto del REPL
        exec(open(os.path.join(project_root, 'repl', 'neural_repl.py')).read(), globals())
        
        # Iniciar IPython con el contexto cargado
        ipshell = InteractiveShellEmbed(
            banner1="REPL de Redes Neuronales (IPython Enhanced)\n",
            exit_msg="Saliendo del REPL de Redes Neuronales..."
        )
        ipshell()
    except ImportError:
        print("IPython no disponible. Usando REPL estándar de Python.")
        print("Para mejor experiencia, instala ipython: pip install ipython")
        print()
        
        # Cargar el contexto del REPL
        exec(open(os.path.join(project_root, 'repl', 'neural_repl.py')).read(), globals())
        
        # Iniciar REPL estándar con el contexto cargado
        import code
        code.interact(local=globals())

if __name__ == '__main__':
    main()
