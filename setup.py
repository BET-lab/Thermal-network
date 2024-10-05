from setuptools import setup, find_packages

setup(
    name="thermal_network",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
    ],
    author="Habin Jo",
    author_email="habinjo0608@gmail.com",
    description="A library for thermal simulation calculations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/BET-lab/Thermal-network",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
