"""
Setup script for Howie CLI
"""

from setuptools import setup, find_packages

# Try to read README_ENHANCED.md, fallback to a simple description
try:
    with open("README_ENHANCED.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Enhanced Howie CLI with Multi-Model Fantasy Football AI Assistant"

with open("requirements_multimodel.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="howie-cli",
    version="2.0.0",
    author="Trevor Gurgick",
    description="Claude-like Fantasy Football AI Assistant CLI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tgurgick/projectHowie",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Games/Entertainment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "howie=howie_enhanced:cli",
            "howie-classic=howie:cli",
            "howie-demo=demo_agents:main_sync",
        ],
    },
    include_package_data=True,
    package_data={
        "howie_cli": ["*.json", "*.yaml"],
    },
)