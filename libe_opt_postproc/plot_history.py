#!/usr/bin/env python
import argparse, os
from libe_opt_postproc.post_processing import PostProcOptimization
# --
import matplotlib
matplotlib.use('Agg')


def parse_args():
    # Command argument line parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('paths', nargs='+', default=[],
                        help=('list of paths to analyze '
                              '(either a libE_opt histoty file '
                              'or a directory containing it)'))
    parser.add_argument('-pars', nargs='+', default=[],
                        help='list of parameters to show')
    parser.add_argument('-opath', type=str, dest='opath', default=None,
                        help='output folder')
    parser.add_argument('--sort', action='store_true', default=False,
                        help='sort simulations by objecive function')
    parser.add_argument('-cut', type=float, dest='cut', default=None,
                        help='select entries with f below this cut')
    parser.add_argument('-top', type=int, dest='top', default=1,
                        help='show top scoring simulation')

    args = parser.parse_args()
    return args


def main():
    # parse command line
    args = parse_args()

    for path in args.paths:
        ppo = PostProcOptimization(path)

        hist_file = ppo.hist_file
        print('History file: %s' % hist_file)

        # Set output path
        if args.opath is None:
            base_dir = os.path.dirname(os.path.abspath(hist_file))
            opath = base_dir + '/plots'
        os.makedirs(opath, exist_ok=True)

        df = ppo.get_df()
        index_list = list(df.sort_values(by=['f'],
                                ascending=True).index)
        print('Show top %i simulations' % args.top)
        for i in range(args.top):
            idx = index_list[i]
            ppo.print_history_entry(idx)

        select = None
        if args.cut is not None:
            select = {'f': [None, args.cut]}

        if args.sort is not None:
            sort = {'f': False}
        else:
            sort = None
            
        if args.pars:
            parnames = args.pars
            ppo.plot_history(parnames=parnames, sort=sort, select=select,
                             filename=opath + '/history_pars.png')
        else:
            if ppo.varpars:
                parnames = ['f']
                parnames.extend(ppo.varpars)
                ppo.plot_history(parnames=parnames, sort=sort, select=select,
                                 filename=opath + '/history_varpars.png')
            if ppo.anapars:
                parnames = ['f']
                parnames.extend(ppo.anapars)
                ppo.plot_history(parnames=parnames, sort=sort, select=select,
                                 filename=opath + '/history_anapars.png')


if __name__ == '__main__':
    main()
