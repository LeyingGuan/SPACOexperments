# SPACOexperments

##  simulations
curpath be the directory path for saving running and save the results, and should be the path to SPACOexperments/simulations/. Make sure there are data and results folders in curpath.
```ruby
python dataGen.py
```
Simulated data are saved under curpath+/data/.
### Benchmarking at the true rank
Modify the default curpath to your own curpath.
```ruby
python reconstruction_compare.py I T J q rate SNR1 SNR2 iteration
```

