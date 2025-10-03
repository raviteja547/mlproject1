from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirments(file_path:str)->List[str]:
    '''
    This function will return the list of reqirments
    '''
    requirments = []
    with open(file_path) as file_obj:
        requirments = file_obj.readlines()
        requirments = [req.replace("\n", "") for req in requirments]

        if HYPEN_E_DOT in requirments:
            requirments.remove(HYPEN_E_DOT)
    
    return requirments

setup(
    name = "firstmlproject",
    version = "0.0.1",
    author = "Raviteja",
    author_email = "raviteja.chinthavatla@gmail.com",
    packages = find_packages(),
    install_requires = get_requirments('requirements.txt')
)