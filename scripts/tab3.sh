#!/bin/bash

for lambda in 1e-4 5e-4 7e-4 1e-3 5e-3 7e-3 1e-2 2e-2 5e-2
do 
    python certifair.py german property1 "german_p1_${lambda}" --lr 0.007 --fr "${lambda}" --layers "30,30,1" --property_loss
done
