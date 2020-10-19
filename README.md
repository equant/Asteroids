
![2-panel image/heatmap of asteroid belt](https://github.com/equant/Asteroids/blob/master/docs/assets/Figure_01_H_Heatmap.png)

# Asteroids Repo

A repo of codes developed for papers written about the asteroid belt.

## Completion Limit
<img align='left' width='33%' src='https://raw.githubusercontent.com/equant/Asteroids/master/docs/assets/completion_limit-fitting-MPCORB-0.02binwidth.png'><br>

Determine the observational completion limit (Hlim) of your data as a function of semi-major axis (a).  This code, written in Python 3, requires numpy, emcee, pandas, matplotlib.

The file src/estimate_completion_limit.py implements the methods from Hendler & Malhotra (2020).  This example script uses test data (sample_data.csv) which is a subset of the minor planet center database.  You will need to provide the data you want analysed as a pandas dataframe with columns 'a' (semi-major axis) and 'H' (absolute magnitude).

<br clear='all'>
