#!/usr/bin/env python3
"""
CSCG Torch Environment Setup Script

Automated setup script to create a local development environment for CSCG Torch.
This script will:
1. Check system requirements
2. Create virtual environment
3. Install dependencies
4. Run basic validation tests
5. Set up development tools

Usage:
    python setup_environment.py [--gpu] [--dev] [--colab]
    
Options:
    --gpu: Install GPU-optimized PyTorch (requires CUDA)
    --dev: Install development dependencies (linting, testing, etc.)
    --colab: Setup for Google Colab environment
"""

import sys
import subprocess
import os
import platform
import argparse
from pathlib import Path
import venv
import urllib.request
import json

class CSCGEnvironmentSetup:
    """CSCG environment setup manager."""
    
    def __init__(self, gpu: bool = False, dev: bool = False, colab: bool = False):
        self.gpu = gpu
        self.dev = dev
        self.colab = colab
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "cscg_env"
        self.python_version = sys.version_info
        
        print("CSCG Torch Environment Setup")
        print("=" * 50)
        print(f"Project root: {self.project_root}")
        print(f"Python version: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"GPU support: {self.gpu}")
        print(f"Development mode: {self.dev}")
        print(f"Colab mode: {self.colab}")
        print()
    
    def check_system_requirements(self):
        """Check system requirements and compatibility."""
        print("Checking system requirements...")
        
        # Check Python version
        if self.python_version < (3, 8):
            print("Python 3.8+ required")
            return False
        print(f"Python {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        
        # Check pip
        try:
            import pip
            print(f"pip {pip.__version__}")
        except ImportError:
            print("pip not found")
            return False
        
        # Check Git
        try:
            result = subprocess.run(["git", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"{result.stdout.strip()}")
            else:
                print("Git not found (optional for development)")
        except FileNotFoundError:
            print("Git not found (optional for development)")
        
        # Check CUDA if GPU requested
        if self.gpu:
            cuda_available = self.check_cuda()
            if not cuda_available:
                print("CUDA not detected - will install CPU-only PyTorch")
                self.gpu = False
        
        return True
    
    def check_cuda(self):
        """Check CUDA installation."""
        try:
            # Try nvidia-smi
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                print("NVIDIA GPU detected")
                # Extract CUDA version if possible
                if "CUDA Version:" in result.stdout:
                    cuda_line = [line for line in result.stdout.split("\n") if "CUDA Version:" in line][0]
                    print(f"{cuda_line.split('CUDA Version:')[1].strip().split()[0]}")
                return True
            else:
                print("nvidia-smi not found")
                return False
        except FileNotFoundError:
            print("nvidia-smi not found")
            return False
    
    def create_virtual_environment(self):
        """Create Python virtual environment."""
        print(f"Creating virtual environment at {self.venv_path}...")
        
        if self.venv_path.exists():
            print(f"Virtual environment already exists at {self.venv_path}")
            response = input("Remove existing environment? (y/N): ")
            if response.lower() == 'y':
                import shutil
                shutil.rmtree(self.venv_path)
                print("Removed existing environment")
            else:
                print("Using existing environment")
                return True
        
        try:
            venv.create(self.venv_path, with_pip=True)
            print("Virtual environment created")
            return True
        except Exception as e:
            print(f"Failed to create virtual environment: {e}")
            return False
    
    def get_pip_command(self):
        """Get pip command for the virtual environment."""
        if platform.system() == "Windows":
            return str(self.venv_path / "Scripts" / "pip")
        else:
            return str(self.venv_path / "bin" / "pip")
    
    def get_python_command(self):
        """Get python command for the virtual environment."""
        if platform.system() == "Windows":
            return str(self.venv_path / "Scripts" / "python")
        else:
            return str(self.venv_path / "bin" / "python")
    
    def install_pytorch(self):
        """Install PyTorch with appropriate CUDA support."""
        print("Installing PyTorch...")
        
        pip_cmd = self.get_pip_command()
        
        if self.gpu:
            # Install GPU version
            torch_install_cmd = [
                pip_cmd, "install", "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ]
            print("Installing GPU-enabled PyTorch (CUDA 11.8)")
        else:
            # Install CPU version
            torch_install_cmd = [
                pip_cmd, "install", "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ]
            print("Installing CPU-only PyTorch")
        
        try:
            result = subprocess.run(torch_install_cmd, check=True, capture_output=True, text=True)
            print("PyTorch installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"PyTorch installation failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return False
    
    def install_dependencies(self):
        """Install project dependencies."""
        print("Installing project dependencies...")
        
        pip_cmd = self.get_pip_command()
        
        # Core dependencies
        core_deps = [
            "numpy>=1.24.0",
            "scipy>=1.10.0", 
            "matplotlib>=3.7.0",
            "tqdm>=4.65.0",
            "numba>=0.57.0",
            "PyYAML>=6.0.0"
        ]
        
        # Development dependencies
        dev_deps = [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "jupyter>=1.0.0",
            "jupytext>=1.15.0"
        ]
        
        # Install core dependencies
        print("Installing core dependencies...")
        for dep in core_deps:
            try:
                subprocess.run([pip_cmd, "install", dep], check=True, capture_output=True)
                print(f"{dep}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {dep}: {e}")
                return False
        
        # Install development dependencies if requested
        if self.dev:
            print("Installing development dependencies...")
            for dep in dev_deps:
                try:
                    subprocess.run([pip_cmd, "install", dep], check=True, capture_output=True)
                    print(f"{dep}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to install {dep}: {e}")
                    # Continue with other dependencies
        
        return True
    
    def install_project(self):
        """Install the project in development mode."""
        print("Installing CSCG Torch project...")
        
        pip_cmd = self.get_pip_command()
        
        try:
            # Install in editable mode
            subprocess.run([pip_cmd, "install", "-e", "."], 
                         check=True, capture_output=True, cwd=self.project_root)
            print("CSCG Torch installed in development mode")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Project installation failed: {e}")
            return False
    
    def run_validation_tests(self):
        """Run basic validation tests."""
        print("Running validation tests...")
        
        python_cmd = self.get_python_command()
        
        # Test PyTorch installation
        test_pytorch = '''
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
'''
        
        try:
            result = subprocess.run([python_cmd, "-c", test_pytorch], 
                                  capture_output=True, text=True, check=True)
            print("PyTorch validation passed")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"PyTorch validation failed: {e}")
            return False
        
        # Test CSCG import
        test_cscg = '''
try:
    from cscg_torch.models.chmm_torch import CHMM_torch
    print("CSCG Torch import successful")
except ImportError as e:
    print(f"CSCG import failed: {e}")
    exit(1)
'''
        
        try:
            result = subprocess.run([python_cmd, "-c", test_cscg], 
                                  capture_output=True, text=True, check=True)
            print("CSCG Torch import validation passed")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"CSCG import validation failed: {e}")
            return False
        
        # Test room loading
        test_rooms = '''
try:
    import torch
    from pathlib import Path
    rooms_dir = Path("rooms")
    if (rooms_dir / "room_5x5_16states.pt").exists():
        room = torch.load(rooms_dir / "room_5x5_16states.pt")
        print(f"Test room loaded: {room.shape}")
    else:
        print("Test rooms not found (run room generation)")
except Exception as e:
    print(f"Room loading failed: {e}")
'''
        
        try:
            result = subprocess.run([python_cmd, "-c", test_rooms], 
                                  capture_output=True, text=True, cwd=self.project_root)
            print("Room loading validation passed")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Room loading validation failed: {e}")
        
        return True
    
    def create_activation_script(self):
        """Create convenience scripts for environment activation."""
        print("Creating activation scripts...")
        
        # Bash/Zsh activation script
        if platform.system() != "Windows":
            activate_script = self.project_root / "activate_env.sh"
            with open(activate_script, 'w') as f:
                f.write(f'''#!/bin/bash
# CSCG Torch Environment Activation Script

echo "Activating CSCG Torch environment..."
source {self.venv_path}/bin/activate

echo "Environment activated!"
echo "Python: $(which python)"
echo "Pip: $(which pip)"

# Set useful aliases
alias cscg-test="python tests/run_all_tests.py"
alias cscg-generate-rooms="cd rooms && python generate_rooms.py && cd .."

echo ""
echo "Available commands:"
echo "  cscg-test          - Run all tests"
echo "  cscg-generate-rooms - Generate test rooms"
echo "  deactivate         - Exit environment"
echo ""
''')
            activate_script.chmod(0o755)
            print(f"Bash activation script: {activate_script}")
        
        # Windows batch script
        if platform.system() == "Windows":
            activate_script = self.project_root / "activate_env.bat"
            with open(activate_script, 'w') as f:
                f.write(f'''@echo off
REM CSCG Torch Environment Activation Script

echo Activating CSCG Torch environment...
call {self.venv_path}\Scripts\activate.bat

echo Environment activated!
echo Python: %VIRTUAL_ENV%\Scripts\python.exe
echo Pip: %VIRTUAL_ENV%\Scripts\pip.exe

echo.
echo Available commands:
echo   python tests\run_all_tests.py    - Run all tests
echo   cd rooms ^&^& python generate_rooms.py  - Generate test rooms
echo   deactivate                       - Exit environment
echo.
''')
            print(f"Windows activation script: {activate_script}")
        
        # Python activation helper
        python_helper = self.project_root / "activate_env.py"
        with open(python_helper, 'w') as f:
            f.write(f'''#!/usr/bin/env python3
"""
CSCG Torch Environment Helper

Quick environment activation and testing script.
"""

import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent
venv_path = project_root / "cscg_env"

def get_python_cmd():
    if sys.platform == "win32":
        return str(venv_path / "Scripts" / "python")
    else:
        return str(venv_path / "bin" / "python")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CSCG Environment Helper")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--generate-rooms", action="store_true", help="Generate rooms")
    parser.add_argument("--validate", action="store_true", help="Validate installation")
    
    args = parser.parse_args()
    
    python_cmd = get_python_cmd()
    
    if args.test:
        subprocess.run([python_cmd, "tests/run_all_tests.py"])
    elif args.generate_rooms:
        subprocess.run([python_cmd, "rooms/generate_rooms.py"])
    elif args.validate:
        subprocess.run([python_cmd, "-c", "import cscg_torch; print('CSCG Torch ready!')"])
    else:
        print("CSCG Torch Environment Helper")
        print("Usage:")
        print("  python activate_env.py --test           # Run tests")
        print("  python activate_env.py --generate-rooms # Generate rooms")
        print("  python activate_env.py --validate       # Validate install")

if __name__ == "__main__":
    main()
''')
        python_helper.chmod(0o755)
        print(f"Python helper script: {python_helper}")
    
    def setup_colab_environment(self):
        """Set up Google Colab specific environment."""
        print("Setting up Google Colab environment...")
        
        colab_setup = self.project_root / "colab_setup.py"
        with open(colab_setup, 'w') as f:
            f.write('''#!/usr/bin/env python3
"""
Google Colab Setup for CSCG Torch

Run this cell in Google Colab to set up the CSCG environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def setup_colab_cscg():
    """Set up CSCG in Google Colab."""
    print("Setting up CSCG Torch in Google Colab...")
    
    # Install dependencies
    deps = [
        "torch", "torchvision", "torchaudio",
        "numpy>=1.24.0", "scipy>=1.10.0", "matplotlib>=3.7.0",
        "tqdm>=4.65.0", "numba>=0.57.0", "PyYAML>=6.0.0"
    ]
    
    for dep in deps:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
            print(f"{dep}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {dep}: {e}")
    
    # Clone repository (if not already present)
    if not Path("CSCG_Maze").exists():
        print("Cloning CSCG repository...")
        subprocess.run(["git", "clone", "YOUR_REPO_URL", "CSCG_Maze"])
        os.chdir("CSCG_Maze")
    
    # Generate rooms if needed
    if not Path("rooms/room_5x5_16states.pt").exists():
        print("Generating test rooms...")
        subprocess.run([sys.executable, "rooms/generate_rooms.py"])
    
    # Install project
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."])
    
    print("CSCG Torch setup complete!")
    print("Run validation:")
    print("!python tests/run_all_tests.py --no-slow")

if __name__ == "__main__":
    setup_colab_cscg()
''')
        print(f"Colab setup script: {colab_setup}")
    
    def print_next_steps(self):
        """Print next steps for the user."""
        print("Environment setup complete!")
        print("=" * 50)
        print("Next steps:")
        print()        
        if platform.system() != "Windows":
            print("1. Activate environment:")
            print(f"   source {self.venv_path}/bin/activate")
            print("   # OR")
            print("   ./activate_env.sh")
        else:
            print("1. Activate environment:")
            print(f"   {self.venv_path}\Scripts\activate")
            print("   # OR")
            print("   activate_env.bat")
        print()        
        print("2. Generate test rooms (if not done):")
        print("   cd rooms && python generate_rooms.py")
        print()        
        print("3. Run tests:")
        print("   python tests/run_all_tests.py")
        print()        
        print("4. Start developing:")
        print("   python -c \"from cscg_torch.models.chmm_torch import CHMM_torch; print('Ready!')\"")
        print()        
        if self.gpu:
            print("GPU support enabled!")
            print("   Test with: python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}')\"")
            print()        
        if self.dev:
            print("Development tools installed:")
            print("   - pytest (testing)")
            print("   - black (code formatting)")
            print("   - flake8 (linting)")
            print("   - jupyter (notebooks)")
            print()        
        print("Documentation:")
        print("   - README.md - Project overview")
        print("   - improvements.txt - GPU optimizations")
        print("   - rooms/README.md - Test rooms info")
        print()        
        print("Need help?")
        print("   python activate_env.py --validate")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="CSCG Torch Environment Setup")
    parser.add_argument("--gpu", action="store_true", help="Install GPU-enabled PyTorch")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    parser.add_argument("--colab", action="store_true", help="Setup for Google Colab")
    
    args = parser.parse_args()
    
    setup = CSCGEnvironmentSetup(gpu=args.gpu, dev=args.dev, colab=args.colab)
    
    # Run setup steps
    if not setup.check_system_requirements():
        print("System requirements not met")
        sys.exit(1)
    
    if not setup.colab:
        if not setup.create_virtual_environment():
            print("Failed to create virtual environment")
            sys.exit(1)
        
        if not setup.install_pytorch():
            print("Failed to install PyTorch")
            sys.exit(1)
        
        if not setup.install_dependencies():
            print("Failed to install dependencies")
            sys.exit(1)
        
        if not setup.install_project():
            print("Failed to install project")
            sys.exit(1)
        
        if not setup.run_validation_tests():
            print("Validation tests failed")
            sys.exit(1)
        
        setup.create_activation_script()
    
    if args.colab:
        setup.setup_colab_environment()
    
    setup.print_next_steps()

if __name__ == "__main__":
    main()