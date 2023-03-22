# SPACOexperments

##  simulations
respath be the directory path for saving running and save the results, and you should run all code in this directory. Make sure there are data and results folders in respath. Let simcodepath be the path linked to simulation scripts.
```ruby
python simcodepath/dataGen.py
```
Simulated data are saved under curpath+/data/.
### 1. Benchmarking at the true rank
I: num of subjects;
T: num of time points;
J: dimnesion of features;
q: dimension of auxiliary covariates;
rate: non-missing rate along the time dimension;
SNR1: signal-to-noise ratio in features;
SNR2: signal-to-noise ratio in auxiliary covariates;

Please modify the default curpath in reconstruction_compare.py to your own curpath. The last argument is the number of repetitions for each setup.
#### Run comparison for one setting
```ruby
python simcodepath/reconstruction_compare.py I T J q rate SNR1 SNR2 20
```
#### Run all settings in bulk
```ruby
```
#### Summarize and plot the results
Summarize the results evaluating reconstruction qualities: (1) Figure 2 (tensor reconstruction); (2) Figure 3 (reconstruction of U, SPACO vs SPACO-); (2) Figure 3 (reconstruction of Phi, SPACO vs SupCP); (3) Figure S? (random initialization vs proposed initialization); (4) Figure S? (tensor reconstruction, missing entries only)
```ruby
python simcodepath/summary.py
```
### 2. Evaluation on rank estimation
```ruby
```

## Real data experiments
### Data preparation
We reorganize the data to (1) separate the immune profiles, risks and clinical responses, (2) remove immune profiles with high missinness (>20%) and perform mofa imputation for the remaining, and (3) generate randomized risk variables.
```ruby
python IMPACT_preparation.py
```
###SPACO and SPACO- on IMPACT data
```ruby
python IMPACT_SPACOrun.py
```






