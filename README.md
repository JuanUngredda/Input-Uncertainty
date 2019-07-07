# Input-Uncertainty

Hi Michael!!

Hi Juan!!!

## TestFuns
contains all code to do with simulators, their parameter ranges and uncertainty parameters (not quite ready yet).  All simultors and information sources go here. (No optimizer or modelling code)

## input_uncertainty_general.py
this is a customised version of the original code.

## Modular Version
This is a version of the original code with functions split up over files.
## IU_Optimizer
contains all files and code to do with KG and Delta loss  and gaussain processes and monte carlo etc. All ***modelling and optimizer*** code goes here.

## runner_(your favourite problem).py
each file loads the code from IU_optimizer, and a single test function from TestFuns, it then runs the optmizer and saves outputs (doesnt work yet)
