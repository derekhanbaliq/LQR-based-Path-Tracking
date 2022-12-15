# LQR-based-Path-Tracking for MEAM 517 Final Project

Development is based on the [repository of the F1TENTH Gym environment](https://github.com/f1tenth/f1tenth_gym). 
You can find the [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) of the environment here.

## Installation

We modified the original installation setup as the new conda environment in Windows system.
After download the repository, install [anaconda](https://www.anaconda.com/products/distribution) and [WSL](https://learn.microsoft.com/en-us/windows/terminal/install).
[This page](https://www.geeksforgeeks.org/how-to-setup-anaconda-path-to-environment-variable/) might help you in configuring the environmental variable of anaconda.
If you use powershell, you can follow [this page](https://www.programmersought.com/article/83207512680/#:~:text=CommandNotFoundError%3A%20Your%20shell%20has%20not%20been%20properly%20configured,Player%20is%20loading.%20This%20is%20a%20modal%20window.) as administrator to enable changes for configuring anaconda.

Configure the environment as follows:
```bash
cd <repo_name>  # navigate to the root directory of this project
conda create -n env517 python=3.8  # create a new conda environment with Python 3.8
conda activate env517  # activate the environment
pip install -e .  # install the dependencies for F1TENTH gym.
pip install -r requirements.txt  # install other dependencies
```

Then you can run a quick waypoint follow example by:
```bash
cd examples
python waypoint_follow.py
```

## Postscript

For MPC development, please check my MPC repo for developed version.
