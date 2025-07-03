from setuptools import setup, find_packages

setup(
    name="mgnn",
    version="1.0",
    packages=find_packages(),  # Sucht in "." automatisch nach Ordnern mit __init__.py
    install_requires=[
        "torch",
        "torch-geometric",
        "numpy",
        "scipy",
        "torch_scatter",
        "tqdm",
        "pyscf"
    ],
)

