import spaco
import numpy as np
import sys
import random


rank = 3
comparison_res = spaco.comparison_pipe(Phi0=data[5],
                                 V0 = data[6],
                                 U0 = data[7],
                                 X = data[2],
                                 Z =  data[9],
                                 O = data[3],
                                 time_stamps = data[4])

comparison_res.compare_run(rank = 3, max_iter = 30)

for method in comparison_res.eval_dict.keys():
    if method != "empirical":
        print(method)
        print(comparison_res.eval_dict[method].alignment)

comparison_res.eval_dict['empirical']






