from setuptools import setup

setup(
        name='ppar',
        version='0.0.1',
        description='Analysis of parallel momentum distributions at NSCL and RIBF',
        author='Tobias Beck',
        author_email='beck@frib.msu.edu',
        python_requires='>=3.9',
        packages=['ppar'],
        install_requires=['numpy','scipy','matplotlib','uproot'],#'PyAtima'],
)