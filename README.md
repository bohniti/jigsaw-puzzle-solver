<div style="border-bottom:none;">
  <div align="center"> 
    <img style="border-bottom:none;" src="./readme-head.png">
    <h1>Langdon</h1>
    <h2>Towards solving Jigsaw Puzzles out of Historical Fragments</h2>
  </div>
</div>


## 

### Intro

[Langdon](https://en.wikipedia.org/wiki/Robert_Langdon), the most excellent puzzle solver of all time, is a jigsaw
puzzle solver written in python. Langdon uses [pytorch](https://pytorch.org)-[lightning](https://www.pytorchlightning.ai) and [partial convolution](https://arxiv.org/pdf/1811.11718.pdf). I developed langdon
while studying computer science at [University of Erlangen Nuermberg](https://www.fau.eu).

>The **main contribution** is the proposal of a [deep siamese](https://arxiv.org/pdf/1707.02131.pdf) [residual network architecture](https://arxiv.org/pdf/1512.03385.pdf) , called Langdong , designed for [historical fragment](https://lme.tf.fau.de/competitions/hisfragir20-icfhr-2020-competition-on-image-retrieval-for-historical-handwritten-fragments/) matching. It is inspiered by
the [work](https://hal.archives-ouvertes.fr/hal-02367779/document) of [Pirrone](mailto:antoine.pirrone@labri.fr) et al.

### Get the data
 Raw
```bash
$wget https://zenodo.org/record/3893807/files/hisfrag20_train.zip?download=1 &&
$wget https://zenodo.org/record/3893807/files/hisfrag20_test.zip?download=1`
```
#### Prepared
You will find them in the data directory as [csv-files](https://github.com/bohniti/jigsaw-puzzle-solver/tree/master/data/hisfrag20/prepared/paris_as_csv) which points to the original files.<br>

*Note: Preproceccing is will be performed **online**. The files just split the data and provides pairs for the siamiese approach.*

### Get the code

```bash
git clone https://github.com/bohniti/jigsaw-puzzle-solver
```

### Get the results

[Project-Report](https:linktoreport)<br>
[Results-Notebook](https:linktoreport)

### Get requirenments

```bash
conda env create -f environment.yml -p /Users/beantown/PycharmProjects/jigsaw-puzzle-solver/conda-env &&
conda activate /Users/beantown/PycharmProjects/jigsaw-puzzle-solver/conda-env
```

*Note: If you want to use another package manger, you have to mangage it py your own. Sorry.*

### Run it on you own

EDA and Preproceccing

```bash
(/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/conda-env): $jupyter notebook ./notebooks/eda_preproceccing.ipynb
```

Main

```bash
(/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/conda-env): $python3 main.py
```

Training configuration

```bash
(/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/conda-env): $vim ./config/config_local.toml
...
...
(/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/conda-env): $vim./config/config_local.toml
```

Results

```bash
(/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/conda-env): $tensorboard --logdir ./results/default/version_X
```

*Note: You can change directory in config files. So, you must change it in the tensorboard command as well.*

Custom steps

```python
from langdon.core import some_steps


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

*Note: step-functions must return the same as the original step-function. Not tested yet, sorry.*

### License

Pretty much the [BSD 3-Clause License](https://github.com/bohniti/jigsaw-puzzle-solver/blob/master/LICENSE), just don't repackage it and call it your own please!<br>
Also if you do make some changes, feel free to make a pull request and help make things more awesome!

### Get in touch

If you have any support requests please feel free to [email](mailto:timo.bohnstedt@icloud.com) me.<br>
Otherwise, feel free to follow me on [Twitter](https://twitter.com/bohniti)!

### Special Thanks

Many thanks to all supervisors for their excellent supervising, patience, and collecting the data:

[Dr.-Ing. Vincent Christlein](https://lme.tf.fau.de/person/seuret/) <br>
[Mathias Seuret, M. Sc.](https://lme.tf.fau.de/person/christlein)