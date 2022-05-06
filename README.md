# *-former Models for Long-Term Series Forecasting

This is the High-Flyer implementation of *-former Models, which aim at conducting Long-Term Series Forecasting (LTSF).

#### Models
+ [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (AAAI 2021)](https://arxiv.org/abs/2012.07436). The raw code and data are from [Github:Informer](https://github.com/zhouhaoyi/Informer2020).
+ [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting (NeurIPS 2021)](https://arxiv.org/abs/2106.13008). The raw code and data are from [Github:Autoformer](https://github.com/thuml/Autoformer).

## Requirements

- Python 3.6
- matplotlib == 3.1.1
- numpy == 1.19.4
- pandas == 0.25.1
- torch == 1.8.0


## Data
The long-term series data can be load from the High-Flyer data warehouse, including:

#### 1. ETT
This dataset contains the data collected from electricity transformers, including load and oil temperature that are recorded every 15 minutes between July 2016 and July 2018.
```
[data_dir]
    ETTh1.csv
    ETTh2.csv
    ETTm1.csv
    ETTm2.csv
```

#### 2. Electricity
This dataset contains the hourly electricity consumption of 321 customers from 2012 to 2014.
```
[data_dir]
    electricity.csv
```

#### 3. Exchange 
This dataset records the daily exchange rates of eight different countries ranging from 1990 to 2016.
```
[data_dir]
    exchange_rate.csv
```

#### 4. Traffic
This dataset is a collection of hourly data from California Department of Transportation, which describes the road occupancy rates measured by different sensors on San Francisco Bay area freeways.
```
[data_dir]
    traffic.csv
```

#### 5. ILI
This dataset includes the weekly recorded influenza-like illness (ILI) patients data from Centers for Disease Control and Prevention of the United States between 2002 and 2021, which describes the ratio of patients seen with ILI and the total number of the patients.
```
[data_dir]
    national_illness.csv
```


## Load Dataset
You can use `hfai.datasets` to load the dataset, such as:
```python
from hfai.datasets import LTSF

dataset = LTSF(data_name="ETTh1", split="train", seq_len=96, label_len=48, pred_len=24, features="S", target="OT", timeenc=0, freq="h")
loader = dataset.loader(batch_size=64, num_workers=4)

for x, y, x_mark, y_mark in loader:
    # training model
```

## Usage

Commands for training and testing LTSF models in High-Flyer AIHPC:

```bash
hfai python train.py -- -n 1 -g jd_a100 -p 30
```


## Results

| model                            | MSE    | Cost Time per Epoch | 
|----------------------------------|--------|---------------------|
| informer_ett_univariate          | 0.0616 | 2.6068s             | 
| autoformer_ett_univariate        | 0.0872 | 9.8183s             |
| informer_ett_multivariate        | 0.1607 | 4.4638s             |
| autoformer_ett_multivariate      | 0.1671 | 12.5459s            |
| informer_exchange_multivariate   | 0.0000 | 4.1323s             |
| autoformer_exchange_multivariate | 0.0000 | 7.7215s             |
| informer_traffic_multivariate    | 0.0000 | 25.8238s            |
| autoformer_traffic_multivariate  | 0.0000 | 34.5530s            |
| informer_ili_multivariate        | -      | 1.6633s             |
| autoformer_ili_multivariate      | -      | 2.4563s             |

