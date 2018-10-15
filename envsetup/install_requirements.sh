#!/bin/bash
python --version
pip install azureml-sdk[notebooks]==0.1.58
pip install aiohttp==3.3.2
pip install toolz==0.9.0
pip install tqdm==4.23.4
pip install azure-cli==2.0.41
pip install lightgbm==2.1.2
pip install papermill==0.14.1
pip install git+https://github.com/theskumar/python-dotenv
conda install -y -f nb_conda==2.2.0
pip install -r requirements.txt