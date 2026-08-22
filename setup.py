"""
Setup script for Howie CLI
"""

from setuptools import setup, find_packages

# Try to read README.md, fallback to a simple description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Enhanced Howie CLI with Multi-Model Fantasy Football AI Assistant"

# Core: what howie3 (the supported implementation) needs to run.
CORE_REQUIREMENTS = [
    "click>=8.0.0",
    "rich>=13.0.0",
    "textual>=0.58.0",
    "pandas>=1.3.0",
    "numpy>=1.21.0",
    "requests>=2.25.0",
    "python-dotenv>=1.0.0",
    "nfl_data_py>=0.3.0",
    "beautifulsoup4>=4.9.0",
    "pyarrow>=10.0.0",  # nflverse parquet fallback for recent seasons
]

EXTRAS = {
    "ai": ["anthropic>=0.21.0"],
    "viz": ["matplotlib>=3.5.0", "seaborn>=0.12.0"],
    "dev": ["pytest>=7.0.0", "pytest-asyncio>=0.21.0", "black>=22.0.0", "mypy>=1.0.0", "types-requests"],
    # Everything the deprecated v2 stack additionally needs
    "legacy": [
        "openai>=1.0.0", "anthropic>=0.21.0", "sqlalchemy>=2.0.0",
        "aiohttp>=3.8.0", "pydantic>=2.0.0", "scikit-learn>=1.0.0",
        "joblib>=1.0.0", "openpyxl>=3.0.0", "python-dateutil>=2.8.0",
    ],
}
EXTRAS["all"] = sorted({dep for deps in EXTRAS.values() for dep in deps})

setup(
    name="howie-cli",
    version="2.5.0",
    author="Trevor Gurgick",
    description="Advanced Fantasy Football AI Assistant with Comprehensive Search Workflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tgurgick/projectHowie",
    packages=find_packages(),
    py_modules=["howie_enhanced"],
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
    python_requires=">=3.9",
    install_requires=CORE_REQUIREMENTS,
    extras_require=EXTRAS,
    entry_points={
        "console_scripts": [
            "howie=howie3.cli:main",
            "howie3=howie3.cli:main",
            "howie-legacy=howie_enhanced:tui_cli",
            "howie-cli=howie_enhanced:cli",
        ],
    },
    include_package_data=True,
    # Explicit allowlist — no databases, pickles, raw data, or .env (enforced
    # by tests/test_boundary.py)
    package_data={
        "howie3": ["schema.sql", "ui/*.html", "ui/*.js", "ui/*.css"],
        "howie_cli": ["*.json", "*.yaml"],
    },
)
