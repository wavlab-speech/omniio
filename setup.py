#!/usr/bin/env python
"""Setup script for omniio package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="omniio",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Efficient multimedia I/O for binary archive blobs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wavlab-speech/omniio",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Video",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "av>=10.0.0",
        "soundfile>=0.12.0",
        "requests>=2.28.0",
        "zstandard>=0.19.0",
        "pyarrow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
        "huggingface": [
            "datasets>=2.0.0",
            "huggingface-hub>=0.19.0",
        ],
    },
    keywords="multimedia audio video text archive blob io",
    project_urls={
        "Bug Reports": "https://github.com/wavlab-speech/omniio/issues",
        "Source": "https://github.com/wavlab-speech/omniio",
    },
)
