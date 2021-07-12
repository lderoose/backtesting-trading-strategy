# Backtesting-Trading-Strategy
------------------------------


This project is an environment for backtesting a trading strategy based on crypto-currency market. The aim is to have a customizable environment where testing can be done quickly and easily for identifying a profitable strategy.

It does not provide any ways for collecting data, does not use Twitter data or any textual data and it is not a trading bot either.


## Strategy
-----------

### *Description*

The strategy use 2 assets : 
- a stable coin called SC (for example : USDT, BUSD, USDC, ...)
- an unstable crypto-currency called U (for example : BTC, ETH...).

And at each period, predict an action depending on the last position :

- if long U at last_position:
    - if beneficial:
        * do nothing
    - else: 
        * short U against SC
- else:
    - if beneficial: 
        * short SC against U
    - else: 
        * do nothing

NB : all the amount of SC or U is trade if the position change.


### *What beneficial means ?*
Due to transaction fees, the notion of beneficial cannot mean only next expected return of U > 0 or < 0, so it depends to rate of transaction fees. 

First, I compute for each period :
- optimal actions if we long on U (label U).
- optimal actions if we long SC (label SC).

Then, I fit a binary classifier algorithm (like logistic regression for example) on label U and an another on label SC. These fitted classifier will predict probability for each period.

Finally, probabilities are used with different aggregation rules for defining actions.


## Installation
---------------
The easiest way is to use Anaconda3 or Miniconda3 on Linux distribution. 

You can download Anaconda3 at this link : https://www.anaconda.com/products/individual#Downloads or Miniconda3 at this link : https://docs.conda.io/en/latest/miniconda.html. (recommend python3.8).

Next,


- Download or git clone project.

- On script located in bin/install_with_conda.sh : modify the PYTHONPATH with your root of project according to your file system and save.

- launch with terminal : 

    ```bash
     bin/install_with_conda.sh

     conda activate backtesting
    ```

## Usage
--------

### 1 - Collect data:
Use this project : https://github.com/binance/binance-public-data
for catching klines Binance historical data and save them into data folder in the root of this project.

NB : if you collect data by any other means you must insert your data in the database by your own. Keep in mind that the SQL table have this structure:

market|frequency|open_time|open|high|low|close|volume|close_time|quote_asset_volume|number_of_trades|taker_buy_base_asset_volume|taker_buy_quote_asset_volume|ignore|
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |


### 2 - Insert collected data into database:
In conf/database.conf, modify parameters 'frequency', 'market' and 'path_historical_klines' according to your collected data.

launch :
```python
python src/database/add_binance_vision_zip_to_db.py
```

This script will open zip files and insert the data in 'path_historical_klines' to SQL database. 


### 3 - Preprocessing
In conf/preprocessing.conf, modify parameters like you want.
    
launch:
```python  
python src/preprocessing/preprocessing.py
```

This script will do :
    
- request SQL database according to parameters specified in preprocessing.conf.

- load data.

- compute features.

- compute labels (see src/preprocessing/cl_labelling.py for more details).

- save all these data in 'save_path' into csv format. 

NB : if frequency is different than 1h or 1min, create your own feature_engineering_`frequency` function in src/preprocessing.py.
     but column "rdt_crypto_`frequency`" is required.

### 4 - Machine learning
In conf/predict.conf, modify parameters like you want.

launch:
```python 
python src/ml/generate_predictions.py
```

This script will fit or refit many times 2 binary classifier on last data and save predicted probabilities into csv format.

NB : model and theirs parameters must be change in src/ml/generate_predictions.py directly.


### 5 - Find strategy
In conf/strategy.conf, modify parameters like you want.

launch:
```python 
python src/strategy/find_and_evaluate_strategy.py
```

This script will do :

- open .csv of predicted probabilities for both models.
- decide actions according to simulation parameters.
- find the best simulation.
- evaluate the performance of the best simulation.
- generate a customizable evaluation report (.txt) and selected figures.

NB : 
- the parameters of simulation must be change in src/strategy/find_and_evaluate_strategy.py directly.
- details of each probabilities transformation rule are provided in src/strategy/cl_strategy_simulator.py.
- it can take long time for doing simulations so I speed up with numba.
- depending on your hardwares and DIC_PARAMS you can met MemoryError. 

## Troubleshooting
------------------
please open an issue

## Disclaimer
-------------

All investment strategies and investments involve risk of loss. Nothing contained in this program, scripts, code or repository should be construed as investment advice. Any reference to an investment's past or potential performance is not, and should not be construed as, a recommendation or as a guarantee of any specific outcome or profit. By using this program you accept all liabilities, and that no claims can be made against the developers or others connected with the program.

## License
----------
MIT

## Donation
-----------
If you find this project helpful, feel free to make a donation at this address :
0x21eAa2A429735410733320065D800ad14Aa6F5DB