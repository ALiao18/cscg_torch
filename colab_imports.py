# Simple redirect to the fixed version
try:
    from .colab_imports_fixed import *
except ImportError:
    # Fallback for direct execution
    from colab_imports_fixed import *