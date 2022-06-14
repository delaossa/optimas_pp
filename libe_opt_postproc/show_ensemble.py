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
    parser.add_argument('-top', type=int, dest='top', default=1,
                        help='show top scoring simulations')
    parser.add_argument('-keep', type=int, dest='keep', default=None,
                        help='keep only the top `keep` simulations (delete the rest)')

    args = parser.parse_args()
    return args


def main():
    # parse command line
    args = parse_args()

    for path in args.paths:
        ppo = PostProcOptimization(path)
        hist_file = ppo.hist_file
        base_dir = os.path.dirname(os.path.abspath(hist_file))
        
        print('Ensemble directory: %s' % base_dir)
        print('History file: %s' % hist_file)

        df = ppo.get_df()
        # get list of indexes ordered by objective function value
        index_list = list(df.sort_values(by=['f'],
                                ascending=True).index)

        top = args.top
        top_list = index_list[:top]
        print('Show top %i simulations: ' % top, top_list)
        for i, idx in reversed(list(enumerate(top_list))):
            print('top %i:' % (i + 1))
            ppo.print_history_entry(idx)

        if args.keep is not None:
            keep = args.keep
            sid_list = df.loc[index_list]['sim_id'].tolist()
            sid_tokeep = sid_list[:keep]
            sid_todelete = sid_list[keep:]
            sid_todelete.sort()
            print('keep top %i: ' % keep, sid_tokeep)
            print('delete the rest: ', sid_todelete)
            # ppo.delete_simulation_data(sid_todelete, edir='ensemble', ddir='diags')
            ppo.delete_simulation_data(sid_todelete)


if __name__ == '__main__':
    main()
