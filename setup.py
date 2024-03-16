# to install local pkg in virtual environment

from setuptools import setup, find_packages

setup(
    name="mcqgenerator",
    version="0.0.1",
    author="zohaib",
    author_email="m.zohaibnasir6@gmail.com",
    install_requires=["openai", "langchain", "streamlit", "python-dotenv", "PyPDF2"],
    packages=find_packages(),  # find local pkg from local directory # find directory with `__init__.py` file
)

"""
to install python pkgs you can you
`pip install -r requirements.txt`
but to installl local pkgs into env
1. you run install this setup.py as `python setup.py`
or 
2. -e .




"""
