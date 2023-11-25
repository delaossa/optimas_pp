#!/usr/bin/env python
import argparse, os
from natsort import natsorted
from optimas_pp.post_processing import PostProcOptimization
# --
import matplotlib
matplotlib.use('Agg')


def parse_args():
    # Command argument line parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('paths', nargs='+', default=[],
                        help=('list of paths to analyze '
                              '(either a Optimas history file '
                              'or a directory containing it)'))
    parser.add_argument('-pars', nargs='+', default=[],
                        help='list of parameters to show')
    parser.add_argument('-xname', type=str, default=None,
                        help='name of the x-axis parameter')
    parser.add_argument('-opath', type=str, dest='opath', default=None,
                        help='output folder')
    parser.add_argument('--sort', action='store_true', default=False,
                        help='sort simulations by objecive function')
    parser.add_argument('-cut', type=float, dest='cut', default=None,
                        help='select entries with f below this cut')
    parser.add_argument('-top', type=int, dest='top', default=1,
                        help='show top scoring simulations')

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

        df = ppo.get_df()
        index_list = list(df.sort_values(by=['f'],
                                ascending=True).index)

        top = args.top
        top_list = index_list[:top]
        print('Show top %i simulations: ' % top, top_list)
        for i, idx in reversed(list(enumerate(top_list))):
            sim_path = ppo.get_sim_path(idx)
            if sim_path.startswith(base_dir):
                sim_path = sim_path[len(base_dir) + 1:]
            print('top %i -> %s' % (i + 1, sim_path))
            # sim_name = os.path.basename(sim_path)
            # print('top %i -> %s' % (i + 1, sim_name))
            ppo.print_history_entry(idx)

        select = None
        if args.cut is not None:
            select = {'f': [None, args.cut]}

        if args.sort:
            sort = {'f': False}
        else:
            sort = None

        xname = args.xname

        if args.pars:
            parnames = args.pars
            ppo.plot_history(parnames=parnames, xname=xname,
                             sort=sort, select=select, top=top,
                             filename=os.path.join(opath, 'history_pars.png'))
        else:
            if ppo.varpars:
                parnames = ['f']
                parnames.extend(ppo.varpars)
                ppo.plot_history(parnames=parnames, xname=xname,
                                 sort=sort, select=select, top=top,
                                 filename=os.path.join(opath, 'history_varpars.png'))
            if ppo.anapars:
                parnames = ['f']
                parnames.extend(ppo.anapars)
                ppo.plot_history(parnames=parnames, xname=xname,
                                 sort=sort, select=select, top=top,
                                 filename=os.path.join(opath, 'history_anapars.png'))


if __name__ == '__main__':
    main()
