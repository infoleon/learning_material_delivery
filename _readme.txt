


Procedure to install conda and required libraries:


To install Mini-Conda, please install the file miniconda.exe, or download it through the Powershell command:

wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -outfile ".\miniconda.exe"


After a sucessful instalation of miniconda, the option to open the miniconda prompt will be available.



After finished, please, open the miniconda prompt and 

conda clean -y --index-cache

conda env create --name workshop --file workshop.yml








