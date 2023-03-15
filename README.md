# LQR-based Path Tracking for MEAM 517 Final Project

Development is based on the [repository of the F1TENTH Gym environment](https://github.com/f1tenth/f1tenth_gym). 
You can find the [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) of the environment here.

## Installation

We modified the original installation setup as the new conda environment in Windows system.
After download the repository, install [anaconda](https://www.anaconda.com/products/distribution) and [WSL](https://learn.microsoft.com/en-us/windows/terminal/install).
[This page](https://www.geeksforgeeks.org/how-to-setup-anaconda-path-to-environment-variable/) might help you in configuring the environmental variable of anaconda.

If you use powershell, you can follow [this page](https://www.programmersought.com/article/83207512680/#:~:text=CommandNotFoundError%3A%20Your%20shell%20has%20not%20been%20properly%20configured,Player%20is%20loading.%20This%20is%20a%20modal%20window.) as administrator to enable changes for configuring anaconda.

For Windows 11 + latest Anaconda, first, make sure the conda can get into the "base" env with the help of [this page](https://forum.qt.io/topic/118934/importerror-dll-load-failed-while-importing-qtcore-the-specified-module-could-not-be-found?_=1678923299303): 
```bash
cmd /k "C:\ProgramData\Anaconda3\Scripts\activate.bat C:\ProgramData\Anaconda3
```

Then, open "command prompt" as administrator, and type in following commands:
```bash
conda activate  # enter base env
conda update --all  # update all the stuff
```
Thanks to the help of [this page](https://github.com/conda/conda/issues/11795).

Configure the environment as follows:
```bash
cd <repo_name>  # navigate to the root directory of this project
conda create -n f110_lqr python=3.8  # create a new conda environment with Python 3.8
conda activate f110_lqr  # activate the environment
pip install -e .  # install the dependencies for F1TENTH gym.
pip install -r requirements.txt  # install other dependencies
```

Then you can run a quick lqr steering example by:
```bash
cd lqr_steering
python main.py
```
Or just config PyCharm and press Ctrl+Shift+F10.

## Postscript

For MPC development, please check my MPC repo for developed version.
