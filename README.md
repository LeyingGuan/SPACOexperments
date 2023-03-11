# SPACOexperments

##  simulations
curpath be the directory path for saving running and save the results, and should be the path to SPACOexperments/simulations/. Make sure there are data and results folders in curpath.
```ruby
python dataGen.py
```
Simulated data are saved under curpath+/data/.
### Benchmarking at the true rank
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
python reconstruction_compare.py I T J q rate SNR1 SNR2 20
```
#### Run all settings in bulk
```ruby
```
#### Summarize and plot the results




