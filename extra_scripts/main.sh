#!/bin/bash

output_datetime=$(grep "date" config.ini | awk '{print $3}')

python3 make_sampled_posterior.py -d $output_datetime

