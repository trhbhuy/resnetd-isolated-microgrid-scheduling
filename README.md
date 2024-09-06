# Real-time power scheduling for isolated microgrid using Dense Residual Neural Network (ResnesD - IMG)

This repository contains the implementation for our paper: ["Real-time power scheduling for an isolated microgrid with renewable energy and energy storage system via a supervised-learning-based strategy"](https://doi.org/10.1016/j.est.2024.111506), published in the Journal of Energy Storage.

<!-- ## Environment 

- tensorflow: 2.0
- torch: 1.9 -->

<!-- ## Dataset
We opensource in this repository the model used for the ISO-NE test case. Code for ResNetPlus model can be found in /ISO-NE/ResNetPlus_ISONE.py

The dataset contains load and temperature data from 2003 to 2014. -->

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
    │   │   ├── data_loader.py             # Data loading methods
    │   │   ├── dataset_aggregation.py     # Dataset aggregation logic
    │   │   ├── feature_engineering.py     # Feature engineering scripts
    │   │   ├── run_scenario.py            # Running scenarios or simulations
    │   │   └── util.py                    # Utility functions specific to methods
    │   ├── platform/
    │   │   ├── components/                # Components of the microgrid platform
    │   │   │   ├── __init__.py
    │   │   │   ├── deg.py                 # Diesel engine generator
    │   │   │   ├── distflow.py            # Distribution network constraints
    │   │   │   ├── ess.py                 # Energy storage system logic
    │   │   │   ├── load.py                # Load (both flexible and inflexible) logic
    │   │   │   └── renewables.py          # Renewable energy sources (PV, Wind, etc.)
    │   │   ├── microgrid_env.py           # Microgrid environment setup and management (for testing)
    │   │   ├── microgrid.py               # Microgrid optimization logic
    │   │   └── util.py                    # Utility functions for the platform
    │   ├── utils/                         # General utility functions
    │   │   ├── __init__.py
    │   │   ├── file_util.py               # File handling utilities
    │   │   └── numeric_util.py            # Numerical operations utilities
    │   ├── __init__.py
    │   └── config.py                      # Configuration file for parameters
    ├── utils/                             # High-level utility scripts
    │   ├── __init__.py
    │   ├── common_util.py                 # Common utility functions
    │   ├── preprocessing_util.py          # Preprocessing utility functions
    │   ├── test_util.py                   # Utility functions for testing
    │   └── train_util.py                  # Utility functions for training
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
## License
[MIT LICENSE](LICENSE)
