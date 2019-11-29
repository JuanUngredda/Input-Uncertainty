
#########################################################################################    
#
#          HOW TO USE THIS SCRIPT!!!
#
#########################################################################################
#
# REQUIREMENTS
#   - ssh keys for passwordless login between local->CSC and CSC<->CSC
#   - conda to be installed on CSC account eg ":~$ source $HOME/.bashrc; conda activate TFenv; python myTFscript.py"
#   - tmux installed (sudo apt install tmux) for htop viewer
#
#########################################################################################
#
# TLDR:
# STEP 1. check machines are awake + free and delete bad ones from list below (literally edit the list in this file).
# From local/CSC terminal type "python fork0_to_csc.py; tmux attach -t Nhtop"
# with each call to this fork0, this file is copied to CSC so the list of machines in this file
# is the list of machines that will be used for jobs.
#
# STEP 2: to run the example script 10 times, type the following command in local terminal:
# "python fork0_to_csc.py \$HOME/cond_bayes_opt/scripts/demo_infra_usage.py 10 -v"
# which will call
#  "python (CSC_HOME_DIR)/cond_bayes_opt/scripts/demo_infra_usage.py (new bespoke dirname) (k)"
# for k in 0,...,9 spread over all the CSC machines in the list below. See demo_infra_usage.py
# for further usage. It is also possible to set a custom conda env, and custom output dir, branch.
# the -v flag is for verbose printing stdout+stderr to terminal otherwise it saves to files.
# there is a --nopull flag mean CSC repo will not pull from github.
# 
#########################################################################################
#
# CALLING THIS SCRIPT WITHOUT ARGUMENTS 
#
# just loads a tmux session with loads of htops, 
# use this first to check which CSC machines are free. In bash, type 
#           "python fork0_to_csc.py; tmux attach -t Nhtop".
# and if some machines do not respond or are already in use, delete them from the CSC_NAMES list below.
# This script also copies this file (with the latest CSC_NAMES) over to CSC storage so the next
# fork1_on_csc.py and fork2_on_csc.py executed from CSC storage will have the same copy of CSC_NAMES.
#
#########################################################################################
#
# CALLING THIS SCRIPT WITH ARGUMENTS
#
#   exp_script: absolute path to the script on CSC storage. This script must be callable
#               with the given conda env as "python (exp_script) (dirname) (k)"
#               Therefore treat exp_script as a single master runner with all experiment settings
#               in a lookup table. Then calling this script with arguments (dirname) (k) 
#               executes the experiment with row k of settings and saves output in (dirname).
# 
#   exp_num: int, the total number of experiments that will be run
#            with exp_script with k = 0,1,2,3,...,exp_num
#
#   --basedir: output base dir on csc storage within which a new unique output dirname will be created.
#              repo contents are coptied into dirname and dirname is then passed to 
#              exp_script ( and can be used for saving outputs).
#              Default basedir is $HOME/RESULTS/(repo name)/
#
#   --first_fork: str, (optional) default=jamon, ssh name of first CSC desktop to connect to
#
#   --conda: str, (optional) default="base". the name of the conda environment to be used on the CSC
#
#   --branch: str, (optional) name of git repo branch to use in the repo on CSC 
#             note that (branch setting is persistent so no need to set every time)
#
#   -v: verbose mode, stdout+stderr go to terminal. Otherwise stdout+stderr will be saved to unique .txt files.
#
#   -nopull: this flag STOPS the default behaviour, which is to logon to CSC and "cd (path_exp_script); git pull".
#
#   This fork0_to_csc will copy this fork0_to_csc.py file over to csc so that CSC_MACHINES list 
#   on the CSC storage is
#   up-to-date. fork0_to_csc will then logon to args.first_fork, and call fork1_on_csc. The fork1_on_csc will
#   navigate to basedir, make a new unique dirname, then partition the 1,...,exp_num jobs into sets for each CSC machine
#   in the list above (used for tmux Nhtop). Then fork1_on_csc.py will logon to each csc machine
#   and call the next fork2_on_csc. This next script will then use the provided conda environment and the partition of jobs
#   to execute "python exp_script (dirname) (k)"" for all k in this csc machine partition of 1,..,exp_num jobs.
#   The new dirname is appended to a list of dirnames in godzilla:~/forkinghellpython/python_savefiles.
#
#   The new dirname is saved into godzilla:~/forkinghellpython/python_savefiles
#
# e.g. call from local machine
#    python fork0_to_csc.py /home/maths/phrnaj/MCBO/run_MCBO_exp.py 1000 --first_fork adobo --basedir /home/maths/phrnaj/MCBO_results/ --conda TFgpu
#
# This will result in runs "python home/maths/phrnaj/MCBO/run_MCBO_exp.py basedir  k" repeatedly for k in 1,...,exp_num spread over CSC desktops.
#
##########################################################################################
