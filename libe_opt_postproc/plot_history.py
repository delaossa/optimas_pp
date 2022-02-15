#!/usr/bin/env python
import argparse
from libe_opt_postproc.post_processing import PostProcOptimization
# --
import matplotlib
matplotlib.use('Agg')


def parse_args():
    # Command argument line parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('paths', nargs='+', default=[],
                        help='list of paths to analyze')
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
        # hist_file_name = os.path.basename(os.path.abspath(hist_file))
        # print('History file name: %s' % hist_file_name)
        # base_dir = os.path.dirname(hist_file)
        # print('Ensemble directory: %s' % base_dir)
        # base_dir_name = os.path.basename(os.path.abspath(base_dir))
        # print('Ensemble directory name: %s' % base_dir_name)
        # print()

        df = ppo.get_df()
        idxmin = df['f'].idxmin()
        ppo.print_history_entry(idxmin)

        if args.top > 1:
            index_list = list(df.sort_values(by=['f'],
                                ascending=True).index)
            for i in range(1, args.top):
                idx = index_list[i]
                ppo.print_history_entry(idx)

        select = None
        if args.cut is not None:
            select = {'f': [None, args.cut]}

        parnames = None
        ppo.plot_history(parnames=parnames, sort=args.sort, select=select,
                         filename='kk.png')
                
                
if __name__ == '__main__':
    main()
