# Dense residual neural network for isolated microgrid (ResnetD - IMG)

This repository contains the implementation for our paper: ["Real-time power scheduling for an isolated microgrid with renewable energy and energy storage system via a supervised-learning-based strategy"](https://doi.org/10.1016/j.est.2024.111506), published in the Journal of Energy Storage.


## Setup 

```bash
conda env create -n torchtf --file env.yml
conda activate torchtf
```


## Structure

```bash
.
├── data/                                  # Directory for data
│   ├── raw/                               # Raw input data
│   ├── processed/                         # Processed datasets
│   └── generated/                         # Generated datasets
└── src/
    ├── networks/                          # Contains network-related logic
    │   ├── __init__.py
    │   └── resnetd.py/                    # Leaner model
    ├── solver/
    │   ├── methods/
    │   │   ├── data_loader.py
    │   │   ├── dataset_aggregation.py
    │   │   ├── feature_engineering.py
    │   │   ├── optimization.py
    │   │   └── util.py
    │   ├── platform/
    │   │   ├── components/                # Components of the system
    │   │   │   ├── __init__.py
    │   │   │   ├── diesel_generator.py
    │   │   │   ├── distflow.py
    │   │   │   ├── energy_storage.py
    │   │   │   ├── flexible_load.py
    │   │   │   └── renewables.py
    │   │   ├── microgrid.py               # Microgrid optimization
    │   │   ├── test_env.py                # Microgrid environment setup and management (for testing)
    │   │   └── util.py
    │   ├── utils/                         # General utility functions
    │   │   ├── __init__.py
    │   │   ├── file_util.py
    │   │   └── numeric_util.py
    │   ├── __init__.py
    │   └── config.py                      # Configuration file for parameters
    ├── utils/                             # High-level utility scripts
    │   ├── __init__.py
    │   ├── common_util.py
    │   ├── preprocessing_util.py
    │   ├── test_util.py
    │   └── train_util.py
    ├── data_generation.py                 # Data generation scripts
    ├── preprocessing.py                   # Data preprocessing scripts
    ├── test_model.py                      # Model testing scripts
    └── train_model.py                     # Model training scripts
```


## How to run

### 1. Data Generation
To generate the necessary data for training and testing:

```
python3 data_generation.py
```

### 2. Data Preprocessing
Prepare the data for training by running the preprocessing script:

```
python3 preprocessing.py
```

### 3. Training the ResnesD Model
Train the ResnesD model using the generated data:

```
python3 train_model.py --data_dir data/generated/ \\
                      --model resnetd --batch_size 200 \\
                      --epochs 200 --learning_rate 0.005 --gpu_device 0 \\
                      --lr_decay_epochs 50 --use_early_stop --patience 30
```

### 4. Testing the Model
Test the trained model on the microgrid environment:

```
python3 test_model.py --env microgrid \\
                      --data_path data/processed/ObjVal.csv \\
                      --num_test_scenarios 90 \\
                      --pretrained_model resnetd \\
                      --learning_rate 0.005 --batch_size 48 --epochs 200 \\
```

## Citation
If you find the code useful in your research, please consider citing our paper:
```
@article{Huy2024,
    author = {Truong Hoang Bao Huy and Tien-Dat Le and Pham Van Phu and Seongkeun Park and Daehee Kim},
    doi = {10.1016/j.est.2024.111506},
    issn = {2352152X},
    journal = {Journal of Energy Storage},
    month = {5},
    pages = {111506},
    title = {Real-time power scheduling for an isolated microgrid with renewable energy and energy storage system via a supervised-learning-based strategy},
    volume = {88},
    year = {2024},
}
```
<!-- ## License
[MIT LICENSE](LICENSE) -->
