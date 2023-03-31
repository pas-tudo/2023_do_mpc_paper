# 2022_do_mpc_paper


## Using Poetry

This project uses [Poetry](https://python-poetry.org/) for dependency management. To install Poetry, follow the instructions on the [Poetry website](https://python-poetry.org/docs/#installation).
For a video introduction and motivation for Poetry watch [this video](https://www.youtube.com/watch?v=0f3moPe_bhk).

**Installation and settings:**

1. Install poetry:
```
pip install poetry
```
2. Recommended settings for Poetry:
    1. Set the virtual environment to be located in the project directory:
    ```
    poetry config virtualenvs.in-project true
    ```
    2. Deactivate modern installer:
    ```
    poetry config experimental.new-installer false
    ```
    This fixes the the issue mentioned [here](https://github.com/python-poetry/poetry/issues/7686).

3. To install the dependencies, run:
    ```
    poetry install
    ```
4. To activate the virtual environment, run:
    ```
    poetry shell
    ```