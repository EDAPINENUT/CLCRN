# Conditional Local Convolution for Spatio-Temporal Meteorological Forecasting


This is a PyTorch implementation of CLCRN in the following paper: 


## Requirements
* torch>=1.7.0 
* torch-geometric-temporal (installation see [Github: torch_geometric_temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal))
* xarray

Dependency can be installed using the following command:
```bash
conda env create --file env_clcrn.yaml
conda activate CLCRN_env
```

## Data Preparation
#### 1） Using the preprocessed dataset:
 The four datasets after preprocessed are available at [Google Drive](https://drive.google.com/drive/folders/1sPCg8nMuDa0bAWsHPwskKkPOzaVcBneD?usp=sharing).

Download the dataset and copy it into `data/` dir. And Unzip them, and obtain `data/{cloud_cover,component_of_wind,humidity,temperature}/`

#### 2） Generating dataset from scratch: 
The raw datasets WeatherBench([Arxiv](https://arxiv.org/abs/2002.00469)) can be downloaded from [Github: WeatherBench](https://github.com/pangeo-data/WeatherBench). And the provided `scripts/generate_training_data.py` is used for data preprocessing.

Dump them into `dataset_release/` files, and run the following commands to generate train/test/val dataset.
```bash
# Dataset preprocess
python scripts/generate_training_data  --input_seq_len=12 --output_horizon_len=12
                                       --k_neighbors=25   --local_map=fast 
```


## Training the Model
The configuration is set in `/experiments/config_clcrn.yaml` file for training process. There are three config files for clcrn/clcstn/baselines training. Run the following commands to train the target model.

```bash
# CLCRN
python train_clcrn.py --config_filename=./experiments/config_clcrn.yaml

# CLCSTN
python train_clcstn.py --config_filename=./experiments/config_clcstn.yaml

# Baselines
python train_baseline.py --config_filename=./experiments/config_baseline.yaml
```


## Citation

If you find this repository, e.g., the code and the datasets, useful in your research, please cite the following paper:
```
TBD
```

## Acknowledgement
The repository is mainly based on DCRNN's Readme, seeing:
https://github.com/liyaguang/DCRNN

And

https://arxiv.org/abs/1707.01926

The baselines are implementated based on torch-geometric-temporal, seeing:

https://arxiv.org/abs/2104.07788