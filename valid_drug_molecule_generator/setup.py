import io
import os
from pathlib import Path

from setuptools import find_packages,setup

NAME = 'valid_drug_molecule_generator'
DESCRIPTION = 'Train and deploy valid_drug_molecule_generator'
URL = 'https://github.com/aimlrl/drug-discovery_smiles'
EMAIL = 'mlresearch.axis@gmail.com'
AUTHOR = 'AiML'
REQUIRES_PYTHON = '3.10'

absolute_path = os.path.abspath(os.path.dirname(__file__))


def list_requirements(file_name='requirements.txt'):

    with io.open(os.path.join(absolute_path,file_name),encoding='utf-8') as file_handle:
        return file_handle.read().splitlines()
    
project_description = DESCRIPTION



ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / NAME
about_project = {}

with open(PACKAGE_DIR / 'VERSION') as file_handle:
    _version = file_handle.read().strip()
    about_project['__version__'] = _version


list_reqs = list_requirements()
list_reqs.append('python==3.10')


setup(
    name=NAME,
    version=about_project['__version__'],
    description=DESCRIPTION,
    long_description=project_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    requires=list_reqs,
    url=URL,
    packages=find_packages(exclude=('tests')),
    package_data={'valid_drug_molecule_generator':['VERSION']},
    ext_modules={},
    ext_modules={},
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ]
)