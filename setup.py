from setuptools import setup

setup(
        name='ppar',
        version='1.0.0',
        description='Analysis of parallel momentum distributions at NSCL/FRIB and RIBF',
        author='Tobias Beck',
        author_email='tobias.beck@kuleuven.be',
        license_files=('LICENSE'),
        python_requires='>=3.9',
        packages=['ppar'],
        install_requires=['numpy','scipy','matplotlib','uproot'],#'PyAtima'],
)