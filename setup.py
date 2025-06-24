from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

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
    },
    entry_points={
        "console_scripts": [
            "cscg-train=cscg_torch.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)