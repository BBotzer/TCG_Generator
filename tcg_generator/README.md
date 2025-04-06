# tcg_generator

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A generative AI tool that generates MTG TCG cards.


This section is to help a user get the MTG Card Generator application up and running on their local machine.  For best performance, it is recommended that users have a NVIDIA GPU, though the model will run on a CPU.  For users that wish to generate artwork, a Stabel Diffusion 3.5 access token is required.


After cloning the GitHub repository (Botzer et al., 2025), users will need to build a virtual environment.  This can be done using Conda and the provided enduser_env.yml file.  The models run inference using PyTorch which can be finiky to install as versions are often dependent on specific hardware and CUDA versions.  Users should inspect the enduser_env.yml to ensure the correct version of PyTorch is being installed for their system.  For more information regarding PyTorch installation, follow the instructions provided by PyTorch to install locally: https://pytorch.org/get-started/locally/.


After setting up the virtual environment, the MTG Card Generator application can be run by launching the flask_app.py file.  The easiest way to do this is by opening a Anaconda/miniconda terminal or Powershell Prompt.  Navigate to the .\TCG_Generator\tcg_generator\tcg_generator directory.  With the tcg_generator_enduser environment active, run the application with the python flask_app.py command.  This will launch the application locally on your local port.  By default, the application will run on http://127.0.0.1:5000 which can be accessed via a web browser.



## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.  aka. Scryfall data.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         tcg_generator and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── tcg_generator   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes tcg_generator a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

