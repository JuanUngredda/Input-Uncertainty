     
  
 # HOW TO USE

 Calling "python fork0_to_csc.py" WITH NO ARGUMENTS just loads a tmux session with loads of htops, 
 use this first to check which machines are free.

 Call fork0_to_csc.py from terminal on local machine with arguments:

   exp_script: abs. path to the script on CSC storage. This script must be callable
               from terminal as python (exp_script) (dirname) (experiment run k)
               Therefore treat it as a python 2.7 wrapper with all experiment settings
               in a lookup table then source .bashrc, load conda env, call bash to execute
               the real source code with row k of settings and save output in (dirname).
 
   exp_num: int, the total number of experiments that will be run
            with exp_script with k = 0,1,2,3,...,exp_num

   --first_fork: ssh name of first CSC desktop to connect to

   --basedir: base dir within which a new unique dirname will be created. 
              dirname is then passed to exp_script for saving outputs.

   This fork0_to_csc will logon to args.first_fork, and call fork1_on_csc. The fork1_on_csc will, 
   inside basedir, make a new unique dirname and then the call the next fork2_on_csc. This
   will then execute the exp_script on all machines using the default system python. So exp_script
   should setup the conda environment and call the actuall source code and run the k^th experiment.

 e.g. call from local machine
    python fork0_to_csc.py /home/maths/phrnaj/MCBO/run_MCBO_exp.py 1000 --first_fork adobo --basedir /home/maths/phrnaj/MCBO_results/

 This will result in runs " python home/maths/phrnaj/MCBO/run_MCBO_exp.py basedir  k" repeatedly exp_num times spread over CSC desktops.


## CSC Desktops 
kumeta

keiko

rilyeno

torta

adobo

bulalo

kinilaw

okoy

embutido

jamon

halabos 

caldereta

dinuguan

sinuglaw

lechon 

niliga

inihaw

menudo
