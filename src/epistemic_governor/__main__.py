"""
Allow running the package as a module:
    python -m epistemic_governor status
    python -m epistemic_governor repl
"""

from epistemic_governor.cli import main

if __name__ == '__main__':
    main()
