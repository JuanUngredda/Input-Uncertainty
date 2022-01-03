import argparse
import subprocess as sp
from time import sleep

N_PROCESSES = 8
     
  
# HOW TO USE THIS SCRIPT!!!
#
# Requirements
#   - ssh keys for passwordless login between local->CSC and CSC<->CSC
#   - tmux (sudo apt install tmux) for htop viewer
#
# Calling this script WITHOUT ARGUMENTS just loads a tmux session with loads of htops, 
# use this first to check which machines are free. In bash, type 
#           "python fork0_to_csc.py; tmux attach -t Nhtop". 
# This script also copies this file (with the latest CSC_NAMES) over to CSC storage so the next
# forks have the same copy of CSC_NAMES.
#
# Call this script from terminal on local machine WITH ARGUMENTS:
#   exp_script: absolute path to the script on CSC storage. This script must be callable
#               with csc default python as "python (exp_script) (dirname) (experiment run k)"
#               Therefore treat exp_script as a single  experiment runner with all experiment settings
#               in a lookup table and it then loads .bashrc, conda env, and calls bash to execute
#               the real source code with row k of settings and save output in (dirname).
# 
#   exp_num: int, the total number of experiments that will be run
#            with exp_script with k = 0,1,2,3,...,exp_num
#
#   --first_fork: (optional) ssh name of first CSC desktop to connect to
#
#   --basedir: (optional but highly recomended) base dir within which a new unique dirname will be created. 
#              dirname is then passed to exp_script for saving outputs.
#
#   This fork0_to_csc will copy this fork0_to_csc.py file over to csc so that CSC_MACHINES list is
#   up-to-date. fork0_to_csc will then logon to args.first_fork, and call fork1_on_csc. The fork1_on_csc will, 
#   inside basedir, make a new unique dirname, then partition the 1,...,exp_num array for each CSC machine
#   in the list above (used for tmux Nhtop) and then the logon to each csc machine
#   and call the next fork2_on_csc. This next script will then use sytem default
#   python to execute the exp_script (dirname) (k) for all k in this csc machine partition of 1,..,exp_num.
#
# e.g. call from local machine
#    python fork0_to_csc.py /home/maths/phrnaj/MCBO/run_MCBO_exp.py 1000 --first_fork adobo --basedir /home/maths/phrnaj/MCBO_results/
#
# This will result in runs " python home/maths/phrnaj/MCBO/run_MCBO_exp.py basedir  k" repeatedly exp_num times spread over CSC desktops.



############################### DEFINE COMPUTER ARRAY #######################################

# set the computers you want to use here, tmux will load to show if they are active.
CSC_NAMES = ["rilyeno", "torta", "adobo", "bulalo", "kinilaw", "okoy",
             "embutido", "jamon", "caldereta", "dinuguan", "lechon",
             "niliga", "inihaw", "halabos", "sinuglaw", "keiko", "kumeta"]

# default list uses all computers, but some may need to be removed.
# working with names is a bitch, instead use the numbers of tmux windows.
U = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
#U = [9,10,12,13,14,15,16]

CSC_NAMES = [CSC_NAMES[i] for i in U]


################################ UTILITY FUNCTIONS ##########################################

# Make a tmux session full of htop windows
def Nhtop(names=CSC_NAMES):
    sp.Popen(['/bin/bash', '-c', 'tmux new -d -s Nhtop'])
    sp.Popen(['/bin/bash', '-c', 'tmux new-window -t Nhtop'])
    sp.Popen(['/bin/bash', '-c', 'tmux swap-window -t Nhtop:0 -s Nhtop:1'])
    sp.Popen(['/bin/bash', '-c', 'tmux kill-window -t Nhtop:1'])

    sleep(0.1)

    # create panes and load htop
    for i in names:
        if i != names[0]:
            sp.call(["/bin/bash", "-c", "tmux split-window -v -t Nhtop"])
            sp.call(["/bin/bash", "-c", "tmux select-layout -t Nhtop tiled"])

        sp.call(["/bin/bash", "-c", "tmux send-keys -t Nhtop 'ssh " + i + "' 'C-m'"])
        # sp.call(["/bin/bash", "-c", "tmux send-keys -t Nhtop 'ssh " + i + "' 'C-m'"])
        #sleep(1) 
        #sp.call(["/bin/bash", "-c", "tmux send-keys -t Nhtop 'yes' 'C-m'"])
        sp.call(["/bin/bash", "-c", "tmux send-keys -t Nhtop 'htop' 'C-m'"])

        sleep(0.15) 


def callbash(cmd):
    _ = sp.check_output(["/bin/bash", "-c", cmd])



################################ THE MAIN EVENT ##########################################

if __name__ == '__main__':
    home = sp.check_output(['echo $HOME'], shell=True).decode()[:-1]
    Nhtop()
    print("\n\nView htop array bash command: tmux attach -t Nhtop\n\n")

    # copy over this file to CSC storage so that CSC_NAMES list is up-to-date.
    print("Copying: $HOME/MCBO/infra/fork0_to_csc.py godzilla:~/MCBO/infra/\n")
    callbash("scp $HOME/MCBO/infra/fork0_to_csc.py godzilla:~/MCBO/infra/")

    parser = argparse.ArgumentParser(description='Run experiments on CSC desktops')
    parser.add_argument('exp_script', type=str, help='Experiment script')
    parser.add_argument('exp_num', type=int, help='Number of experiments to run')
    parser.add_argument('--first_fork', type=str, default='adobo', help='which CSC desktop to start from')
    parser.add_argument('--basedir', type=str, default=home+'/debug/', help='Folder to store results in')

    # This will crash if there are no arguments
    args = parser.parse_args()
    args.basedir = args.basedir+"/"

    # This will not run if there are no arguments
    print("Calling all CSC machines! Your country needs you!")
    callbash(
        f"ssh {args.first_fork} 'source ~/.bashrc; conda activate TFgpu; python ~/MCBO/infra/fork1_on_csc.py " + \
        args.exp_script + " " + str(args.exp_num) + " " + args.basedir + "'&")
