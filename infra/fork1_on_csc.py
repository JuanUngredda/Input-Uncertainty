import subprocess as sp
import numpy as np
import argparse
from datetime import datetime
import os
from fork0_to_csc import CSC_NAMES, callbash
import shutil



# Call this script from terminal on CSC machine with arguments
#   exp_script: abs path to the script on CSC storage
#   exp_num: int, number of experiments that will be run
#            where exp_script will be called with one integer argument
#   basedir: root folder within which to make new dirname and pass to exp_script
#
#   This script will create a new unique folder, then divide the work up amongst
#   all of the desktops.



def create_exp_dir(base_dir="/home/maths/phrnaj/debug/", git_root="/home/maths/phrnaj/MCBO/"):

    # read the ID.txt file to get the integer uid by counting experiments
    with open('/home/maths/phrnaj/python_savefiles') as f:
        for i, l in enumerate(f):
            pass
    uid = str(i+1)

    # get timestamp
    timestamp = str(datetime.now()).replace(' ', '.')[:-7]

    # git commit hash  TODO: strip out \n safe ?
    os.chdir(git_root)
    git = sp.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode()[:-1]

    # read current user
    user = sp.check_output(['whoami']).decode()[:-1]

    # construct the new full directory name!
    dirname = base_dir + uid +"."+ user + "."+ timestamp + "."+ git

    # append name to ID.txt and create directory
    with open('/home/maths/phrnaj/python_savefiles', 'a') as f:
        f.write(dirname + '\n')
    
    # make the dir if it does not exist
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        os.makedirs(dirname+"/src")
        os.makedirs(dirname+"/StdOut")

    # copy source code
    src_files = os.listdir(git_root)
    print("\n\nCopying source Code to "+dirname+"/src/")
    for f in src_files:
        if ".py" in f:
            src_file = git_root + "/" +f
            shutil.copy2(src_file, dirname+"/src/")
            print(src_file)
    print("\n")
    
    
    return dirname


######################### THE MAIN EVENT ################################

def run(args):

    N_MACHINES = len(CSC_NAMES)

    # make a new unique dir within the base dir
    dirname = create_exp_dir(args.basedir, 
                             os.path.dirname(args.exp_script))

    # list of jobs and shuffle them
    exp_ids = np.arange(args.exp_num)
    np.random.shuffle(exp_ids)

    # Divide the jobs over CSCmachines
    split_points = np.round(np.arange(1, N_MACHINES) * args.exp_num / (N_MACHINES)).astype('int')
    splits = np.split(exp_ids, split_points)

    # For loop over CSC machines
    for name, split in zip(CSC_NAMES, splits):
        # the list of experiment numbers for this CSC machine
        fork_args = [f'--k {i} ' for i in split]

        # call each machine and allocate it some jobs
        # command: python fork_again_on_csc.py (abs path to script on CSC storage) (new dirname) (list of jobs)
    
        callbash("ssh " + name + " 'python ~/MCBO/infra/fork2_on_csc.py '" + \
                 args.exp_script + ' ' + dirname + ' ' + ' '.join(fork_args) + '&')
    
    # DONE!


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments on CSC desktops')
    parser.add_argument('exp_script', type=str, help='Experiment script')
    parser.add_argument('exp_num', type=int, help='Number of experiments to run')
    parser.add_argument('basedir', type=str,default='/home/maths/phrnaj/debug/', help='Folder to store result in')

    args = parser.parse_args()
    run(args)
