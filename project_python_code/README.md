## This is code repository for CEVE 543 : Project 1

## Deep Priors for Precipitation Downscaling - Kushal Vyas (kv30)

### Instruction to run code:

In order to run the code, please use a CUDA enabled GPU such as Nvidia RTX 2080 / A100 / etc.

Please create a new virtual environment with `python==3.10.12`. Following which, you can install all the required packages by running `pip install -r requirements.txt` and then activate the environment.


Please open the jupyter-lab notebook and run all cells. It will execute the experiments. You can change the SR_FACTOR global flag to change the precipitation downscaling factor.

Script:

`fit_dip_rainfall_superres.ipynb` : Runs the precipitation downscaling for a single precip grid

`fit_dip_temporal.ipynb` : Runs the spatio-temporally consistent precipitation downscaling


