"""
This file contains a class that helps post-process libE optimization
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import warnings
from natsort import natsorted
import glob


class PostProcOptimization(object):

    def __init__(self, path, varpars=None, anapars=None):
        """
        Initialize a postprocessing object

        Parameter:
        ----------
        path: string
            Path to the folder that contains the libE optimization,
            or path to the individual `.npy` history file.
        """

        # Find the `npy` file that contains the results
        if os.path.isdir(path):
            # Get history file (if any)
            hist_list = natsorted(glob.glob('%s/libE_history_*.npy' % path))
            if len(hist_list) == 0:
                raise RuntimeError('The specified path does not contain any `.npy` file.')
            else:
                self.hist_file = hist_list[-1]
                if len(hist_list) > 1:
                    txt = ('The specified path contains multiple `.npy` files.\n'
                           'Using the last one (in alphabetical order).')
                    warnings.warn(txt)

        elif path.endswith('.npy'):
            self.hist_file = path

        else:
            raise RuntimeError('The path should either point to a folder or a `.npy` file.')

        # Load the file as a pandas DataFrame
        x  = np.load(self.hist_file)
        d = {label: x[label].flatten() for label in x.dtype.names
             if label not in ['x', 'x_on_cube']}
        self.df = pd.DataFrame(d)

        # Only keep the simulations that finished properly
        self.df = self.df[self.df.returned]

        # Make the time relative to the start of the simulation
        self.df['given_time'] -= self.df['gen_time'].min()
        self.df['returned_time'] -= self.df['gen_time'].min()
        self.df['gen_time'] -= self.df['gen_time'].min()

        self.varpars = varpars
        self.anapars = anapars

        if None in [self.varpars, self.anapars]:
            self._auto_detect_parameters()

    def get_df(self, select=None):
        """
        Return a pandas DataFrame containing the data from the simulation

        Parameters
        ----------
        select: dict, optional
            it lists a set of rules to filter the dataframe, e.g.
            'f' : [None, -10.] (get data with f < -10)
        """

        if select is not None:
            condition = ''
            for key in select:
                if select[key][0] is not None:
                    if condition != '':
                        condition += ' and '
                    condition += '%s > %f' % (key, select[key][0])
                if select[key][1] is not None:
                    if condition != '':
                        condition += ' and '
                    condition += '%s < %f' % (key, select[key][1])
            print('selecting according to the condition: ', condition)
            return self.df.query(condition)
        
        return self.df

    def _auto_detect_parameters(self):
        """
        Search optimization folder to find out the list of specific parameters

        Note: it is assumed that the current history file is located
        in the optimization folder
        """
 
        from libensemble.tools import fields_keys as fkeys
        # all parameters present in history file
        allpars = list(self.df.columns)
        # print('all variables: ', allpars)
        # list of libE reserved names always listed in H
        libE_field_names = [f[0] for f in fkeys.libE_fields]
        libE_field_names.extend(['resource_sets', 'f'])
        print('libE parameters:    ', libE_field_names)
        # list of specific parameter names (user defined)
        spepars = [x for x in allpars if x not in libE_field_names]
        # print('specific parameters available: ', spepars)
        # find out the varying parameters
        if self.varpars is None:
            varparfiles = glob.glob('**/varying_parameters.py', recursive=True)
            if len(varparfiles) == 0:
                self.varpars = []
                txt = ('Varying_parameters.py not found.')
                warnings.warn(txt)
            else:
                basedir = os.path.dirname(varparfiles[0])
                sys.path.insert(1, basedir)
                from varying_parameters import varying_parameters
                self.varpars = list(varying_parameters.keys())
        print('varying parameters: ', self.varpars)

        if self.anapars is None:
            self.anapars = [x for x in spepars if (x not in self.varpars) and (x != 'f')]
        print('analyzed quantities:', self.anapars)

    def sortby(self, parname='f', ascending=True):
        """
        Sort dataframe by the specified parameter.

        Parameters
        ----------
        parname: string, optional
            Name of the parameter to sort by

        ascending: bool, optional
            when `True` it sorts in ascending order,
            otherwise, in descending order
        """
        self.df = self.df.sort_values(by=[parname], ascending=ascending).reset_index(drop=True)

    def print_history_entry(self, idx):
        """
        Print parameters for row entry with index `idx`
        """

        h = self.df.iloc[idx]
        print('simulation %i:' % (h['sim_id']))
        print('objective function:')
        print('%20s = %10.5f' % ('f', h['f']))
        if self.varpars is not None:
            print('varying parameters:')
            for name in self.varpars:
                print('%20s = %10.5f' % (name, h[name]))
        if self.anapars is not None:
            print('analyzed parameters:')
            for name in self.anapars:
                print('%20s = %10.5f' % (name, h[name]))

    def plot_history(self, parnames=None, sort=False, select=None, filename=None):
        """
        Print selected parameters versus simulation index.

        Parameters
        ----------
        sort: bool, optional
            when `True`, it orders simulations acoording by the values of `f` (descendingly)

        select: dict, optional
            it lists a set of rules to filter the dataframe, e.g.
            'f' : [None, -10.] (get data with f < -10)

        filename: string, optional
            When defined, it saves the figure to the specified file.
        """

        index = list(self.df.index)
            
        # order list of simulations and re-index
        if sort:
            self.sortby('f', ascending=False)

        df = self.get_df()
            
        if select is not None:
            df_select = self.get_df(select)
        else:
            df_select = None

        if parnames is None:
            parnames = ['f']
            parnames.extend(self.varpars)
            print(parnames)
        
        nplots = len(parnames)
        
        # definitions for the axes
        left, width, width0 = 0.08, 0.72, 0.15
        bottom, top = 0.1, 0.04
        xspacing = 0.015
        yspacing = 0.04
        height = (1. - bottom - top - (nplots - 1) * yspacing) / nplots
        # fig_height = bottom + top + nplots * height + (nplots - 1) * yspacing
        nbins = 25
    
        plt.figure()

        ax_histy_list = []
        histy_list = []
        for i in range(nplots):

            bottom1 = bottom + (nplots - 1 - i) * (yspacing + height)
            rect_scatter = [left, bottom1, width, height]
            rect_histy = [left + width + xspacing, bottom1, width0, height]
            
            h = df[parnames[i]]
            ax_scatter = plt.axes(rect_scatter)
            plt.grid(color='lightgray', linestyle='dotted')
            plt.plot(index, h, 'o')
            if df_select is not None:
                index_cut = list(df_select.index)
                h_cut = df_select[parnames[i]]
                plt.plot(index_cut, h_cut, 'o')

            if (parnames[i] == 'f') and (not sort):
                cummin = df.f.cummin().values
                plt.plot(index, cummin, '-', c='black')
            
            plt.title(parnames[i].replace('_', ' '), fontdict={'fontsize': 8}, loc='right', pad=2)
        
            ax_histy = plt.axes(rect_histy)
            plt.grid(color='lightgray', linestyle='dotted')
            ymin, ymax = ax_scatter.get_ylim()
            binwidth = (ymax - ymin) / nbins
            bins = np.arange(ymin, ymax + binwidth, binwidth)
            histy, _, _ = ax_histy.hist(h, bins=bins,
                    weights=100. * np.ones(len(h)) / len(h), orientation='horizontal')
            if df_select is not None:
                h_cut = df_select[parnames[i]]
                ax_histy.hist(h_cut, bins=bins,
                              weights=100. * np.ones(len(h_cut)) / len(h), orientation='horizontal')
            ax_histy.set_ylim(ax_scatter.get_ylim())

            ax_histy_list.append(ax_histy)
            histy_list.append(histy)

            # ax_scatter.set_ylabel(parnames[i])
            # ax_scatter.set_ylabel('par$_{%i}$' % i)
            if i != nplots - 1:
                ax_scatter.tick_params(labelbottom=False)
                ax_histy.tick_params(direction='out', labelbottom=False, labelleft=False)
                if i == 0:
                    ax_scatter.set_ylabel('')
            else:
                ax_histy.tick_params(direction='out', labelleft=False)
                ax_scatter.set_xlabel('simulation number')
                ax_histy.set_xlabel('events [%]')
            
            histmax = 1.1 * max([h.max() for h in histy_list])
            for i, ax_h in enumerate(ax_histy_list):
                ax_h.set_xlim(-1, histmax)

        if filename is not None:
            plt.savefig(filename, dpi=300)

    def plot_optimization(self, fidelity_parameter=None, **kwargs):
        """
        Plot the values that where reached during the optimization

        Parameters:
        -----------
        fidelity_parameter: string or None
            Name of the fidelity parameter
            If given, the different fidelity will
            be plotted in different colors

        kwargs: optional arguments to pass to `plt.scatter`
        """
        if fidelity_parameter is not None:
            fidelity = self.df[fidelity_parameter]
        else:
            fidelity = None
        plt.scatter( self.df.returned_time, self.df.f, c=fidelity)

    def get_trace(self, fidelity_parameter=None,
                  min_fidelity=None, t_array=None,
                  plot=False, **kw):
        """
        Plot the minimum so far, as a function of time during the optimization

        Parameters:
        -----------
        fidelity_parameter: string
            Name of the fidelity parameter. If `fidelity_parameter`
            and `min_fidelity` are set, only the runs with fidelity
            above `min_fidelity` are considered.

        fidelity_min: float
            Minimum fidelity above which points are considered

        t_array: 1D numpy array
            If provided, th

        plot: bool
            Whether to plot the trace

        kw: extra arguments to the plt.plot function

        Returns:
        --------
        time, max
        """
        if fidelity_parameter is not None:
            assert min_fidelity is not None
            df = self.df[self.df[fidelity_parameter] >= min_fidelity]
        else:
            df = self.df.copy()

        df = df.sort_values('returned_time')
        t = np.concatenate( (np.zeros(1), df.returned_time.values))
        cummin = np.concatenate( (np.zeros(1), df.f.cummin().values))

        if t_array is not None:
            # Interpolate the trace curve on t_array
            N_interp = len(t_array)
            N_ref = len(t)
            cummin_array = np.zeros_like(t_array)
            i_ref = 0
            for i_interp in range(N_interp):
                while i_ref < N_ref - 1 and t[i_ref + 1] < t_array[i_interp]:
                    i_ref += 1
                cummin_array[i_interp] = cummin[i_ref]
        else:
            t_array = t
            cummin_array = cummin

        if plot:
            plt.plot( t_array, cummin_array, **kw)

        return t_array, cummin_array

    def plot_worker_timeline(self, fidelity_parameter=None):
        """
        Plot the timeline of worker utilization

        Parameter:
        ----------
            fidelity_parameter: string or None
                Name of the fidelity parameter
                If given, the different fidelity will
                be plotted in different colors
        """
        df = self.get_df()
        if fidelity_parameter is not None:
            min_fidelity = df[fidelity_parameter].min()
            max_fidelity = df[fidelity_parameter].max()

        for i in range(len(df)):
            start = df['given_time'].iloc[i]
            duration = df['returned_time'].iloc[i] - start
            if fidelity_parameter is not None:
                fidelity = df[fidelity_parameter].iloc[i]
                color = plt.cm.viridis( (fidelity - min_fidelity) / (max_fidelity - min_fidelity))
            else:
                color = 'b'
            plt.barh([str(df['sim_worker'].iloc[i])],
                     [duration], left=[ start],
                     color=color, edgecolor='k', linewidth=1)

        plt.ylabel('Worker')
        plt.xlabel('Time ')
