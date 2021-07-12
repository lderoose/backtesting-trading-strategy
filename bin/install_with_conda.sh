#!/usr/bin/bash


PYTHONPATH="/../../backtesting-trading-strategy"  # only replace this path
cd $PYTHONPATH
awk -i inplace -v var="$PYTHONPATH" '{sub(/to_replace/, var, $0); print}' environment.yml 
conda env create --file environment.yml 