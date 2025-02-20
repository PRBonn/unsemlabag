<div align="center">
    <h1>Unsupervised Semantic Label Generation in Agricultural Fields</h1>
    <br />
    <img src='pics/overview.png'>
    <a href=https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/roggiolani2025frai.pdf>Paper</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href=https://github.com/PRBonn/unsemlabag/issues>Contact Us</a>
  <br />
  <br />
</div>

## Setup

Build docker image:

```commandline
make build
```

You may need to [install the NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to bring the GPUs to work with Docker.

## Data Samples and Weights

We provide one small rgb map in `samples/maps` to test our label generation.

Inside of `samples/network` we provide three folders `train`, `val` and `test`, they all contain the same single image from [PhenoBench](https://www.phenobench.org/) with its semantic label. This allows to check the training procedure and to test the provided weights on one sample. To perform a full training-validation, one can download the PhenoBench dataset from `https://www.phenobench.org/`. 

### Option 1: Manual download
Download the data samples here: [samples.zip](https://www.ipb.uni-bonn.de/html/projects/roggiolani2025frai/samples.zip),

and the weights here: [weights.ckpt](https://www.ipb.uni-bonn.de/html/projects/roggiolani2025frai/weights.ckpt).

Unzip the samples and copy the folder samples and the weights into the main folder.

### Option 2: Automated download

Execute
```commandline
make download
```

## Test on samples

We provide some samples to try the generation of the labels and the evaluation of our trained model. 
You can test the label generation running

```commandline
make generate
```

You can see the sample map in `/samples/maps` and the generated label image will be saved in results.
To visualize the results of the trained model, you can run

```commandline
make test
```

The images used for the inference are in `samples/network/test/images` and one image displaying input = network rpediction - ucnertainty - corrected prediction will be saved in results.

## How to generate a dataset from the map

You can generate images from the complete map using 

```commandline
make map_to_images
```

only after you generated the labels, otherwise it will fail. The resulting images will be saved in `results/generated`. 

## How to train the network 

You can train the network running 

```commandline
make train
```

By default this will use the samples in `samples/network`, where there are three defined folders train, val, test.
If you want to use the generated samples you need to modify the `config/config.yaml` file, specifically 

```commandline
##Data
data:
    name: 'map' # either "pb" or "map" 
    ...
    root_dir: "./results/generated"

```

which specify the dataloader to use, i.e., MapData, and the root directory for the data, i.e., samples/generated.

## How make the code your own

### Data

If you want to train on data that is not inside of the folder, you need to
 
1. In the makefile change the DATA_PATH to point at your data
2. Write or import the dataloader:
	a. Implement the dataloader in the datasets folder
	b. change the __init__.py file in the datasets folder to import the dataloader
	c. Change your config file data name and root_dir to access your new data 


### Style Guidelines

In general, we follow the Python [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines. Please install [black](https://pypi.org/project/black/) to format your python code properly.
To run the black code formatter, use the following command:

```commandline
black -l 120 path/to/python/module/or/package/
```

To optimize and clean up your imports, feel free to have a look at this solution for [PyCharm](https://www.jetbrains.com/pycharm/guide/tips/optimize-imports/).

