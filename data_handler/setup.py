from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="data_handler",
    version="0.1.0",
    author="Stephan Goerttler",
    author_email="goerttlers@uni.coventry.ac.uk",
    description="A data handler package for the sleep stage classification project",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
)