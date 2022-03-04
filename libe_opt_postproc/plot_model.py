#!/usr/bin/env python
import argparse, os
from libe_opt_postproc.post_processing import PostProcOptimization


def parse_args():
    # Command argument line parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('paths', nargs='+', default=[],
                        help=('list of paths to analyze '
                              '(either a libE_opt histoty file '
                              'or a directory containing it)'))
    parser.add_argument('-xname', type=str, default=None,
                        help='name of the x-axis parameter')
    parser.add_argument('-yname', type=str, default=None,
                        help='name of the y-axis parameter')
    parser.add_argument('-obj', type=str, default='f',
                        help='name of the objective parameter')
    parser.add_argument('-pars', nargs='+', default=[],
                        help='list with the names of the parameters of the model')
    parser.add_argument('--max', action='store_true', default=False,
                        help='toogles maximization')
    parser.add_argument('--stddev', action='store_true', default=False,
                        help='shows in addition the std_dev of the fitted model')
    parser.add_argument('-opath', type=str, dest='opath', default=None,
                        help='output folder')

    args = parser.parse_args()
    return args


def main():
    # parse command line
    args = parse_args()

    for path in args.paths:
        ppo = PostProcOptimization(path)

        hist_file = ppo.hist_file
        print('History file: %s' % hist_file)

        parnames = args.pars
        objname = args.obj
        minimize = not args.max
        ppo.build_model_ax(parnames=parnames, objname=objname, minimize=minimize)

        # Set output path
        if args.opath is None:
            base_dir = os.path.dirname(os.path.abspath(hist_file))
            opath = base_dir + '/plots'
        os.makedirs(opath, exist_ok=True)

        fname = 'model_%s' % objname
        xname = args.xname
        yname = args.yname
        if None not in [xname, yname]:
            fname += '_vs_%s_%s.png' % (xname, yname)
        else:
            fname += '.png'
        
        ppo.plot_model(xname=args.xname, yname=args.yname, filename=opath + '/%s' % fname, stddev=args.stddev)


if __name__ == '__main__':
    main()
