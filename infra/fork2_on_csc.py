import argparse
import multiprocessing as mp
import os
import socket

from fork0_to_csc import CSC_NAMES, N_PROCESSES, callbash



def run(args):

    print(f'{socket.gethostname()} received args: {args.k}')

    # parallel version
    pool = mp.Pool(N_PROCESSES)
    command_prefix = 'nice -n 10 python ' + args.exp_script + ' ' + args.dirname + ' '
    [pool.apply(callbash, args=(command_prefix + str(k),)) for k in args.k]
    # for k in args.k:
        # print(command_prefix + str(k))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments on CSC desktops')
    parser.add_argument('exp_script', type=str, help='Experiment script')
    parser.add_argument('dirname', type=str, help='Experiment directory')
    parser.add_argument('--k', type=int, action='append', help='Experiment id')

    args = parser.parse_args()

    run(args)
