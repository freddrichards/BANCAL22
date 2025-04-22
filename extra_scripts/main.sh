#!/bin/bash

# config.ini
output_datetime=$(grep "date" config.ini | awk '{print $3}')
echo $output_datetime

python3 make_sampled_posterior.py -d $output_datetime

