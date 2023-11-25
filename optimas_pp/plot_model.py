#!/usr/bin/env python
import argparse, os
from natsort import natsorted
from optimas_pp.post_processing import PostProcOptimization
import matplotlib.pyplot as plt


def parse_args():
    # Command argument line parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('paths', nargs='+', default=[],
                        help=('list of paths to analyze '
                              '(either a Optimas history file '
                              'or a directory containing it)'))
    parser.add_argument('-pars', nargs='+', default=[],
                        help='list of parameters of the model')
    parser.add_argument('-xname', type=str, default=None,
                        help='name of the x-axis parameter')
    parser.add_argument('-yname', type=str, default=None,
                        help='name of the y-axis parameter')
    parser.add_argument('-obj', type=str, default='f',
                        help='name of the objective parameter')
    parser.add_argument('-opath', type=str, dest='opath', default=None,
                        help='output folder')
    parser.add_argument('--max', action='store_true', default=False,
                        help='toggles maximization')
    parser.add_argument('--stddev', action='store_true', default=False,
                        help='shows in addition the std_dev of the fitted model')

    args = parser.parse_args()
    return args


def main():
    # parse command line
    args = parse_args()

    path_list = natsorted(args.paths)

    for path in path_list:
        ppo = PostProcOptimization(path)

        hist_file = ppo.hist_file
        print('History file: %s' % hist_file)
        base_dir = os.path.dirname(os.path.abspath(hist_file))
        dirlist = os.listdir(base_dir)
        if 'evaluations' in dirlist:
            base_dir, _ = os.path.split(base_dir)

        # Set output path
        if args.opath is None:
            opath = os.path.join(base_dir, 'plots')
        os.makedirs(opath, exist_ok=True)

        parnames = args.pars
        objname = args.obj
        minimize = not args.max
        # build model and return the AxModelManager object
        amm = ppo.build_model(parnames=parnames, objname=objname,
                              minimize=minimize)

        fname = 'model_%s' % objname
        xname = args.xname
        yname = args.yname
        if None not in [xname, yname]:
            fname += '_vs_%s_%s.png' % (xname, yname)
        else:
            fname += '.png'

        figsize = (6, 5)
        if args.stddev:
            figsize = (10,5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        figname = os.path.join(opath, '%s' % fname)
        amm.plot_model(xname=args.xname, yname=args.yname,
                       filename=figname, cmap='Spectral',
                       stddev=args.stddev)


if __name__ == '__main__':
    main()
