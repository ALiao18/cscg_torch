from setuptools import setup, find_packages
from pathlib import Path

# Read README
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "GPU-optimized PyTorch implementation of CSCG Hidden Markov Models"

# Read requirements
try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    # Fallback requirements
    requirements = [
        "numpy>=1.24.0",
        "torch>=2.1.0", 
        "tqdm>=4.65.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "numba>=0.57.0",
        "PyYAML>=6.0.0",
        "plotly>=5.0.0",
        "seaborn>=0.12.0"
    ]

setup(
    name="cscg-torch",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="GPU-optimized PyTorch implementation of CSCG Hidden Markov Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/cscg_torch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "jupytext>=1.15",
            "plotly>=5.0",
        ],
        "graph": [
            "python-igraph>=0.10.0",
        ],
    },
    # entry_points={
    #     "console_scripts": [
    #         "cscg-train=cscg_torch.cli:main",
    #     ],
    # },
    include_package_data=True,
    zip_safe=False,
)