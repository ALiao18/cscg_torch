#!/usr/bin/env python3
"""
Setup script for CSCG-Torch package
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements from requirements.txt
def read_requirements():
    requirements = []
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    # Extract package name and version
                    if '>=' in line:
                        requirements.append(line)
                    elif line.startswith('numpy'):
                        requirements.append('numpy>=1.24.0')
                    elif line.startswith('torch'):
                        requirements.append('torch>=2.1.0')
                    elif line.startswith('tqdm'):
                        requirements.append('tqdm>=4.65.0')
                    elif line.startswith('scipy'):
                        requirements.append('scipy>=1.10.0')
                    elif line.startswith('matplotlib'):
                        requirements.append('matplotlib>=3.7.0')
                    elif line.startswith('numba'):
                        requirements.append('numba>=0.57.0')
                    elif line.startswith('jupyter'):
                        requirements.append('jupyter>=1.0.0')
                    elif line.startswith('jupytext'):
                        requirements.append('jupytext>=1.15.0')
                    elif line.startswith('PyYAML'):
                        requirements.append('PyYAML>=6.0.0')
                    elif line.startswith('plotly'):
                        requirements.append('plotly>=5.0.0')
                    elif line.startswith('seaborn'):
                        requirements.append('seaborn>=0.12.0')
    return requirements

# Core requirements (required for basic functionality)
install_requires = [
    'numpy>=1.24.0',
    'torch>=2.1.0',
    'tqdm>=4.65.0',
    'scipy>=1.10.0',
    'matplotlib>=3.7.0',
    'numba>=0.57.0',
    'PyYAML>=6.0.0',
]

# Optional requirements
extras_require = {
    'visualization': ['plotly>=5.0.0', 'seaborn>=0.12.0'],
    'notebooks': ['jupyter>=1.0.0', 'jupytext>=1.15.0'],
    'graphs': ['python-igraph>=0.10.0'],
    'dev': [
        'pytest>=7.0.0',
        'pytest-cov>=4.0.0',
        'black>=23.0.0',
        'flake8>=6.0.0',
        'mypy>=1.0.0',
    ],
    'all': [
        'plotly>=5.0.0', 'seaborn>=0.12.0',
        'jupyter>=1.0.0', 'jupytext>=1.15.0',
        'python-igraph>=0.10.0',
    ]
}

setup(
    name='cscg-torch',
    version='1.0.0',
    author='Andrew Liao',
    author_email='yl8520@nyu.edu',
    description='A PyTorch implementation of Compositional State-Action Graph (CSCG) models for reinforcement learning',
    long_description=read_file('README.md') if os.path.exists('README.md') else 'A PyTorch implementation of Compositional State-Action Graph (CSCG) models for reinforcement learning and navigation tasks, optimized for modern GPUs.',
    long_description_content_type='text/markdown',
    url='https://github.com/ALiao18/cscg_torch',
    project_urls={
        'Bug Reports': 'https://github.com/ALiao18/cscg_torch/issues',
        'Source': 'https://github.com/ALiao18/cscg_torch',
        'Documentation': 'https://github.com/ALiao18/cscg_torch#readme',
    },
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    zip_safe=False,
    keywords='pytorch, reinforcement learning, navigation, cognitive graphs, CSCG, GPU acceleration',
) 