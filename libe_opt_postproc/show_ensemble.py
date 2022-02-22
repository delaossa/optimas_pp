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
        index_list = list(df.sort_values(by=['f'], ascending=True).index)
        print('Show top %i simulations' % args.top)
        for i in range(args.top):
            idx = index_list[i]
            ppo.print_history_entry(idx)

        if args.keep is not None:
            itop = args.keep
            sid_array = df.iloc[index_list]['sim_id'].tolist()
            print('keep top %i: ' % itop, sid_array[:itop])
            print('delete the rest: ', sid_array[itop:])
            for sid in sid_array[itop:]:
                simdir = ppo.get_sim_dir_name(sid, edir='ensemble')
                if simdir is not None:
                    print('deleting %s/diags .. ' % (simdir))
                    os.system('rm -rf %s/diags' % (simdir))


if __name__ == '__main__':
    main()
