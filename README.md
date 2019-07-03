# Input-Uncertainty

Hi Michael!!

Hi Juan!!!

## IU_Optimizer
contains all files and code to do with KG and Delta loss  and gaussain processes and monte carlo etc. All modelling code goes here.

## TestFuns
contains all code to do with simulators, their parameter ranges and uncertainty parameters (not quite ready yet).  All simultors and information sources go here. (No optimizer or modelling code)

## IU_Optimizer_(your favourite problem)\_runner.py
each file loads the code from IU_optimizer, and a single test function from TestFuns, it then runs the optmizer and saves outpus
