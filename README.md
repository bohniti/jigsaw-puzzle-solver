# Jigsaw Puzzle Solver


## Intro

Langdong is a jigsaw puzzle solver written in python and uses pytorch-lightning.
This master project was developed by unter the supervising of Vincent Chrislein.

The main contribution is the proposal of a deep siamese network architecture,
called Langdong , designed for historical fragment matching.

It is inspiered by the [work](https://hal.archives-ouvertes.fr/hal-02367779/document) of [Pirrone](mailto:antoine.pirrone@labri.fr) et al. 


## Get the data

```bash
wget https://zenodo.org/record/3893807/files/hisfrag20_train.zip?download=1 &&
wget https://zenodo.org/record/3893807/files/hisfrag20_test.zip?download=1`
```

## Get the code

```bash
git clone https://github.com/bohniti/jigsaw-puzzle-solver
```

## Get the results
[Project-Report](https:linktoreport)<br>
[Results-Notebook](https:linktoreport)



## Get requirenments

You can use the environment.yml file and activate it.

```bash
conda env create -f environment.yml -p /Users/beantown/PycharmProjects/jigsaw-puzzle-solver/conda-env &&
conda activate /Users/beantown/PycharmProjects/jigsaw-puzzle-solver/conda-env
```
>NOTE: If you want to use another package manger, you have to mangage it py your own. Sorry.

## Run it on you own
#### EDA and Preproceccing
```bash
(/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/conda-env): $jupyter notebook ./notebooks/eda_preproceccing.ipynb
```
#### Main
```bash
(/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/conda-env): $python3 main.py
```
#### Training configuration
```bash
(/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/conda-env): $vim ./config/config_local.toml
...
...
(/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/conda-env): $vim./config/config_local.toml
```

#### Use custom steps

```python
from puzzle_solver.core import some_steps

def custom_init_step():
    ...
    return config, transform, model


def main():
    config, transform, model = custom_init_step()
    train_dataloader, val_dataloader = load_step(config, transform)
    tb_logger = log_step(config)
    train_step(config, model, train_dataloader, val_dataloader, tb_logger)


if __name__ == "__main__":
    main()
```
>NOTE: Step-functions must return the same as the original step-function. Not tested yet, sorry.

## License?

Pretty much the BSD license, just don't repackage it and call it your own please!

Also if you do make some changes, feel free to make a pull request and help make things more awesome!

## Contact Info?

If you have any support requests please feel free to [email](mailto:timo.bohnstedt@icloud.com) me.

Otherwise, feel free to follow me on [Twitter](https://twitter.com/bohniti)!

## Special Thanks

[Dr.-Ing. Vincent Christlein](https://lme.tf.fau.de/person/seuret/) <br>
[Mathias Seuret, M. Sc.](https://lme.tf.fau.de/person/christlein)