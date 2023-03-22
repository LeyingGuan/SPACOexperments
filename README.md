# SPACOexperments

##  simulations
Let simcodepath be the path linked to simulation scripts. All code has been configured to work with the authors local enviromenet. Please make sure the enviroment path is correctly specified to your computing enviroment. For example, curpath = '/home/lg689/project/SPACOresults' is used by the authors, and you will need to change it to where you want to save the results. Depending on which file you are running, the results are saved to different folders under curpath (please create the folders beforehand).
```ruby
python simcodepath/dataGen.py
```
Simulated data are saved under curpath+/data/.
### 1. Benchmarking at the true rank
```ruby
I: num of subjects;
T: num of time points;
J: dimnesion of features;
q: dimension of auxiliary covariates;
rate: non-missing rate along the time dimension;
SNR1: signal-to-noise ratio in features;
SNR2: signal-to-noise ratio in auxiliary covariates;
 ```
The last argument is the number of repetitions for each setup and results are saved under Simulated data are saved under curpath+/results/construction/.
#### Run comparison for one setting
```ruby
python simcodepath/reconstruction_compare.py I T J q rate SNR1 SNR2 20
```
#### Summarize and plot the results (all settings)
Summarize the results evaluating reconstruction qualities: (1) Figure 2 (tensor reconstruction); (2) Figure 3 (reconstruction of U, SPACO vs SPACO-); (2) Figure 3 (reconstruction of Phi, SPACO vs SupCP); (3) Figure S? (random initialization vs proposed initialization); (4) Figure S? (tensor reconstruction, missing entries only)
```ruby
python simcodepath/summary.py
```
### 2. Evaluation on rank estimation
results are saved under Simulated data are saved under curpath+/results/rank_selection/
#### Run rank selection for one setting
```ruby
python simcodepath/rank_selection.py  I  T J q rate SNR1 SNR2
```
#### Summarize and plot the results (all settings)
```ruby
python simcodepath/summary_rank.py
```
### 3. Evaluation on hypothesis testing
results are saved under Simulated data are saved under curpath+/results/test/
```ruby
python simcodepath/testing.py  I  T J q rate SNR1 SNR2
```
#### Summarize and plot the results (all settings)
```{ruby}
pyyhon simcodepath/summary_rank
```
## Real data experiments
Please run real data experiments under SPACOexperiments.
### Data preparation
We reorganize the data to (1) separate the immune profiles, risks and clinical responses, (2) remove immune profiles with high missinness (>20%) and perform mofa imputation for the remaining, and (3) generate randomized risk variables.
```ruby
python RealData/IMPACT_preparation.py
```
###SPACO and SPACO- on IMPACT data
```ruby
python RealData/IMPACT_SPACOrun.py
```






