#!/usr/bin/env python
import argparse, os
import pandas as pd
from copy import deepcopy
from natsort import natsorted
from libe_opt_postproc.post_processing import PostProcOptimization
# --
import matplotlib
import matplotlib.pyplot as plt
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
    parser.add_argument('-top', type=int, dest='top', default=1,
                        help='show top scoring simulations')
    parser.add_argument('-keep', type=int, dest='keep', default=None,
                        help='keep only the top `keep` simulations (delete the rest)')

    args = parser.parse_args()
    return args


def main():
    # parse command line
    args = parse_args()

    path_list = natsorted(args.paths)

    # List containing the best simulation case data frame
    # for each ensemble in the given list
    df_mins = []
    for path in path_list:
        ppo = PostProcOptimization(path)
        hist_file = ppo.hist_file
        dir_path = os.path.dirname(os.path.abspath(hist_file))
        dir_name = os.path.basename(os.path.abspath(dir_path))
        
        print('Ensemble directory: %s  \npath: %s' % (dir_name, dir_path))
        print('History file: %s' % hist_file)

        df = ppo.get_df()
        print(list(df.columns))
        # get list of indexes ordered by objective function value
        index_list = list(df.sort_values(by=['f'],
                                ascending=True).index)

        df_mins.append(deepcopy(df.loc[[index_list[0]]]))
        df_mins[-1]['dir_name'] = dir_name

        top = args.top
        top_list = index_list[:top]
        print('Show top %i simulations: ' % top, top_list)
        for i, idx in reversed(list(enumerate(top_list))):
            sim_name = os.path.basename(ppo.get_sim_dir_name(idx))
            print('top %i -> %s' % (i + 1, sim_name))
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

    # plot minimums together
    if (len(df_mins) > 1) and (len(args.pars) > 0):
        df_all = pd.concat(df_mins, ignore_index=True)
        df_all.to_csv('scan_data.cvs', encoding='utf-8', index=False)

        pars = args.pars
        fig, axs = plt.subplots(len(pars), sharex=True, figsize=(6, 1.5 * len(pars)), dpi=300)
        for i, par in enumerate(pars):
            axs[i].grid(color='lightgray', linestyle='dotted')
            axs[i].plot(df_all[par], '--o')
            axs[i].set_ylabel('$\mathrm{%s}$' % par.replace('_', '~'))

        axs[-1].set_xlabel('$\mathrm{ensemble~run}$')
        plt.tight_layout()
        fig.savefig('scan_parameters.pdf')


if __name__ == '__main__':
    main()
