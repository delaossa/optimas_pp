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
from copy import deepcopy
# Ax utilities for model building
from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import (
    GenerationStep, GenerationStrategy)
from ax.modelbridge.registry import Models
from ax.modelbridge.factory import get_GPEI
from ax.core.observation import ObservationFeatures


class PostProcOptimization(object):

    def __init__(self, path, varpars=None, anapars=None):
        """
        Initialize a postprocessing object

        Parameter:
        ----------
        path: string
            Path to the folder that contains the libE optimization,
            or path to the individual `.npy` history file.
        
        varpars: list of string
            List with the names of the varying parameters (autodetected when omited)

        varpars: list of string
            List with the names of the analyzed quantities (autodetected when omited)
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
        # patch for older versions of the libE_opt history file
        if 'returned_time' in list(self.df.columns.values):
            self.df['returned_time'] -= self.df['gen_time'].min()
        self.df['gen_time'] -= self.df['gen_time'].min()

        self.varpars = varpars
        self.anapars = anapars

        # if None in [self.varpars, self.anapars]:
        if self.varpars is None:
            self._auto_detect_parameters()

        # Ax model building
        self.ax_client = None
        self.model = None

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
            return self.get_df_with_select(select)
        else:
            return self.df

    def get_df_with_select(self, select, df=None):

        if df is None:
            df = self.df

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
        print('Selecting according to the condition: ', condition)
        return df.query(condition)

    def plot_optimization(self, fidelity_parameter=None, **kw):
        """
        Plot the values that where reached during the optimization

        Parameters:
        -----------
        fidelity_parameter: string or None
            Name of the fidelity parameter
            If given, the different fidelity will
            be plotted in different colors

        kw: optional arguments to pass to `plt.scatter`
        """
        if fidelity_parameter is not None:
            fidelity = self.df[fidelity_parameter]
        else:
            fidelity = None
        plt.scatter( self.df.returned_time, self.df.f, c=fidelity, **kw)
        plt.ylabel('f')
        plt.xlabel('Time ')

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

    def _auto_detect_parameters(self):
        """
        Search optimization folder to find out the list of specific parameters

        Note: it is assumed that the current history file is located
        in its corresponding optimization folder
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

        # setup searching directories
        base_dir = os.path.dirname(os.path.abspath(self.hist_file))
        search_dirs = [base_dir, base_dir + '/sim_specific']

        # find out the varying parameters
        if self.varpars is None:
            varparfiles = []
            for dire in search_dirs:
                filepath = dire + '/varying_parameters.py'
                if os.path.isfile(filepath):
                    varparfiles.append(filepath)
                
            if len(varparfiles) == 0:
                self.varpars = []
                txt = ('varying_parameters.py not found.')
                warnings.warn(txt)
            else:
                basedir = os.path.dirname(varparfiles[0])
                sys.path.insert(1, basedir)
                from varying_parameters import varying_parameters
                self.varpars = list(varying_parameters.keys())
        print('Varying parameters: ', self.varpars)

        # find out the analized parameters
        if self.anapars is None:
            # self.anapars = [x for x in spepars if (x not in self.varpars) and (x != 'f')]
            anaparfiles = []
            for dire in search_dirs:
                filepath = dire + '/analysis_script.py'
                if os.path.isfile(filepath):
                    anaparfiles.append(filepath)

            if len(anaparfiles) == 0:
                self.anapars = []
                txt = ('analysis_script.py not found.')
                warnings.warn(txt)
            else:
                basedir = os.path.dirname(anaparfiles[0])
                sys.path.insert(1, basedir)
                from analysis_script import analyzed_quantities
                self.anapars = [x[0] for x in analyzed_quantities]
        print('Analyzed quantities:', self.anapars)

    def print_history_entry(self, idx):
        """
        Print parameters for row entry with index `idx`
        """

        h = self.df.iloc[idx]
        print('Simulation %i:' % (h['sim_id']))
        print('Objective function:')
        print('%20s = %10.5f' % ('f', h['f']))
        if self.varpars is not None:
            print('varying parameters:')
            for name in self.varpars:
                print('%20s = %10.5f' % (name, h[name]))
        if self.anapars is not None:
            print('analyzed parameters:')
            for name in self.anapars:
                print('%20s = %10.5f' % (name, h[name]))

    def get_sim_dir_name(self, sim_id, edir='ensemble'):
        # get simulation directory
        ensemble_path = os.path.dirname(os.path.abspath(self.hist_file)) + '/' + edir
        simdirs = os.listdir(ensemble_path)
        sim_name_id = 'sim%i_' % (sim_id)
        for simdir in simdirs:
            if sim_name_id in simdir:
                directory = '%s/%s' % (ensemble_path, simdir)
                return directory
        return None

    def plot_history(self, parnames=None, select=None, sort=None, filename=None):
        """
        Print selected parameters versus simulation index.

        Parameters
        ----------
        parnames: list of strings, optional
            List with the names of the parameters to show.

        select: dict, optional
            it lists a set of rules to filter the dataframe, e.g.
            'f' : [None, -10.] (get data with f < -10)

        sort: dict, optional
            A dict containing as keys the names of the parameres to sort by and,
            as values, a Bool indicating if ordering ascendingly (True) or descendingly otherwise.
            e.g. {'f': False} sort simulations according to `f` descendingly.

        filename: string, optional
            When defined, it saves the figure to the specified file.
        """

        df = self.df.copy()
        index = list(df.index)
            
        # order list of simulations and re-index
        if sort is not None:
            df = df.sort_values(by=list(sort.keys()), ascending=tuple(sort.values())).reset_index(drop=True)
            
        if select is not None:
            df_select = self.get_df_with_select(select, df)
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
            print('Saving figure to', filename)

    def build_model_ax(self, parnames=[], objname='f', minimize=True):
        """
        Initialize a the AxClient using the history data
        and fits a Gaussian Process model to it.

        Parameter:
        ----------
        parnames: list of string
            List with the names of the parameters of the model

        objname: string
            Name of the objective parameter

        minimize: bool
            Whether to minimize or maximize the objective
        """
        if not parnames:
            parnames = self.varpars

        parameters = [{'name': p_name,
                       'type': 'range',
                       'bounds': [self.df[p_name].min(), self.df[p_name].max()],
                       'value_type': 'float'
                       } for p_name in parnames]

        # create Ax client
        gs = GenerationStrategy([GenerationStep(model=Models.GPEI, num_trials=-1)])
        self.ax_client = AxClient(generation_strategy=gs, verbose_logging=False)
        self.ax_client.create_experiment(
            name='libe_opt_data',
            parameters=parameters,
            objective_name=objname,
            minimize=minimize,
        )

        # adds data
        metric_name = list(self.ax_client.experiment.metrics.keys())[0]
        for index, row in self.df.iterrows():
            params = {p_name: row[p_name] for p_name in parnames}
            _, trial_id = self.ax_client.attach_trial(params)
            self.ax_client.complete_trial(trial_id, {metric_name: (row[metric_name], np.nan)})

        # fit GP model
        experiment = self.ax_client.experiment
        self.model = get_GPEI(experiment, experiment.fetch_data())

    def plot_model(self, xname=None, yname=None, filename=None, npoints=200):
        """
        Plot model in the two selected variables, while others are fixed to the optimum.

        Parameter:
        ----------
        xname: string
            Name of the variable to plot in x axis.
        
        yname: string
            Name of the variable to plot in y axis.

        filename: string, optional
            When defined, it saves the figure to the specified file.

        npoints: int, optional
            Number of points in each axis
        """

        if self.ax_client is None:
            raise RuntimeError('AxClient not present. Run `build_model_ax` first.')

        if self.model is None:
            raise RuntimeError('Model not present. Run `build_model_ax` first.')

        # get experiment info
        experiment = self.ax_client.experiment
        parnames = list(experiment.parameters.keys())
        minimize = experiment.optimization_config.objective.minimize

        if len(parnames) < 2:
            raise RuntimeError('Insufficient number of parameters in data for this plot. Minimum 2.')

        # Make a parameter scan in two of the input dimensions
        if xname is None:
            xname = parnames[0]
        
        if yname is None:
            yname = parnames[1]

        print('Plotting the model in the %s vs %s plane' % (xname, yname))

        xaxis = np.linspace(experiment.parameters[xname].lower,
                            experiment.parameters[xname].upper, npoints)
        yaxis = np.linspace(experiment.parameters[yname].lower,
                            experiment.parameters[yname].upper, npoints)
        X, Y = np.meshgrid(xaxis, yaxis)
        xarray = X.flatten()
        yarray = Y.flatten()

        # Get optimum
        best_arm, best_point_predictions = self.model.model_best_point()
        best_pars = best_arm.parameters
        print('Best point parameters: ', best_pars)
        print('Best point prediction: ', best_point_predictions)
        
        obsf_list = []
        obsf_0 = ObservationFeatures(parameters=best_pars)
        for i in range(len(xarray)):
            predf = deepcopy(obsf_0)
            predf.parameters[xname] = xarray[i]
            predf.parameters[yname] = yarray[i]
            obsf_list.append(predf)

        mu, cov = self.model.predict(obsf_list)
        metric_name = list(self.ax_client.experiment.metrics.keys())[0]
        f_plt = np.asarray(mu[metric_name])
        sd_plt = np.sqrt(cov[metric_name][metric_name])

        # get numpy arrays with experiment parameters
        xtrials = np.zeros(experiment.num_trials)
        ytrials = np.zeros(experiment.num_trials)
        for i in range(experiment.num_trials):
            xtrials[i] = experiment.trials[i].arm.parameters[xname]
            ytrials[i] = experiment.trials[i].arm.parameters[yname]

        # get numpy array with the experiment metric
        # df = experiment.fetch_data().df
        # df = df[df.metric_name == metric_name]
        # ftrials = np.asarray(df['mean'])
        # get metric at optimal point
        # fopt = float(df[df.arm_name == best_arm.name]['mean'])

        f_plots = [f_plt, sd_plt]
        labels = ['value', 'std. deviation']
        fig, axs = plt.subplots(len(f_plots), figsize=(6.4, 9.6), dpi=100)
        fig.suptitle('Model for metric %s' % metric_name)
        for i, f in enumerate(f_plots):
            cmap = 'Spectral'
            if (i == 0) and (not minimize):
                cmap = 'Spectral_r'
            im = axs[i].pcolormesh(xaxis, yaxis, f.reshape(X.shape), cmap=cmap, shading='auto')
            cbar = fig.colorbar(im, ax=axs[i])
            cbar.set_label(labels[i])
            axs[i].set(xlabel=xname, ylabel=yname)
            # adding contour lines with labels
            axs[i].contour(X, Y, f.reshape(X.shape), levels=20,
                           linewidths=0.5, colors='black', linestyles='solid')
            # plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=6)
            axs[i].scatter(xtrials, ytrials, s=2, c='black', marker='o')
        plt.tight_layout()

        if filename is not None:
            plt.savefig(filename, dpi=300)
            print('Saving figure to', filename)
