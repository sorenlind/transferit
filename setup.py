"""Setup script for package."""
import re
from setuptools import setup, find_packages

match = re.search(r'^VERSION\s*=\s*"(.*)"', open("transferit/version.py").read(), re.M)
VERSION = match.group(1) if match else "???"
with open("README.md", "rb") as f:
    LONG_DESCRIPTION = f.read().decode("utf-8")

setup(
    name="transferit",
    version=VERSION,
    description="Train a model using transfer learning and serve it using TF Serving.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Soren Kristiansen",
    author_email="sorenlind@mac.com",
    url="",
    keywords="",
    packages=find_packages(),
    install_requires=["tf-nightly", "matplotlib", "requests", "scikit-learn", "tqdm"],
    extras_require={
        "dev": [
            "black",
            "coverage",
            "flake8",
            "jupyter",
            "lxml-stubs",
            "matplotlib",
            "mypy",
            "pycodestyle",
            "pydocstyle",
            "pylint",
            "rope",
            "pytest",
            "pytest-cov",
            "tox",
        ],
        "test": ["coverage", "pytest", "pytest-cov", "tox"],
    },
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
    ],
    entry_points={
        "console_scripts": [
            "transferit = transferit.__main__:main",
        ]
    },
)
