import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
# Ax utilities for model building
from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import (
    GenerationStep, GenerationStrategy)
from ax.modelbridge.registry import Models
from ax.modelbridge.factory import get_GPEI
from ax.core.observation import ObservationFeatures


class AxModelManager(object):

    def __init__(self, df):
        """
        Initialize an AxModelManager object

        Parameter:
        ----------
        df: Pandas dataframe
            Full data with at least the varying parameters and the objective
            function values.

        """

        if isinstance(df, pd.DataFrame):
            self.df = df
        else:
            raise RuntimeError('A pandas DataFrame must be given.')

        self.parnames = []
        self.ax_client = None
        self.model = None

    def build_model(self, parnames=[], objname='f', minimize=True):
        """
        Initialize the AxClient using the given data, the model parameters and the metric,
        and fits a Gaussian Process model.

        Parameter:
        ----------
        parnames: list of string
            List with the names of the parameters of the model

        objname: string
            Name of the objective parameter

        minimize: bool
            Whether to minimize or maximize the objective.
            Only relevant to establish the best point and the orientation of the colormap.
        """

        if (not parnames) and self.parnames:
            parnames = self.parnames
        elif parnames:
            self.parnames = parnames
        else:
            raise RuntimeError('Parameter names list is not defined.')
            
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

    def evaluate_model(self, sample, p0=None):
        """
        Evaluate the model over the specified sample

        Parameter:
        ----------
        sample: Pandas DataFrame or numpy array
            If numpy array, it must contain the values of all the model parameres.
            If DataFrame, it can contain only those parameter to vary.
            The rest of parameters would be set to the model best point,
            unless they are further specified using `p0`.

        p0: dictionary
            Particular values of parameters to be fixed for the evaluation over the sample.

        minimize: bool
            Whether to minimize or maximize the objective
        """

        if self.model is None:
            raise RuntimeError('Model not present. Run `build_model_ax` first.')
        
        # Get optimum
        best_arm, best_point_predictions = self.model.model_best_point()
        parameters = best_arm.parameters
        parnames = list(parameters.keys())
        # user specific point
        if p0 is not None:
            for key in p0.keys():
                if key in parameters.keys():
                    parameters[key] = p0[key]
        
        if isinstance(sample, np.ndarray):
            # check the shape of the array
            if sample.shape[1] != len(parnames):
                raise RuntimeError('Second dimension of the sample array should match the number of parameters of the model')
        elif isinstance(sample, pd.DataFrame):
            # check if labels of the dataframe match parnames
            for col in sample.columns:
                if col not in parnames:
                    raise RuntimeError('Column %s does not match any of the parameter names' % col)
        else:
            raise RuntimeError('Wrong data type')

        obsf_list = []
        obsf_0 = ObservationFeatures(parameters=parameters)
        for i in range(sample.shape[0]):
            predf = deepcopy(obsf_0)
            if isinstance(sample, np.ndarray):
                for j, parname in enumerate(parameters.keys()):
                    predf.parameters[parname] = sample[i][j]
            elif isinstance(sample, pd.DataFrame):
                for col in sample.columns:
                    predf.parameters[col] = sample[col].iloc[i]
            obsf_list.append(predf)

        mu, cov = self.model.predict(obsf_list)
        metric_name = list(self.ax_client.experiment.metrics.keys())[0]
        f_array = np.asarray(mu[metric_name])
        sd_array = np.sqrt(cov[metric_name][metric_name])

        return f_array, sd_array

    def plot_model(self, xname=None, yname=None, p0=None, filename=None, npoints=200, stddev=False, **kw):
        """
        Plot model in the two selected variables, while others are fixed to the optimum.

        Parameter:
        ----------
        xname: string
            Name of the variable to plot in x axis.
        
        yname: string
            Name of the variable to plot in y axis.

        p0: dictionary
            Particular values of parameters to be fixed for the evaluation over the sample.

        filename: string, optional
            When defined, it saves the figure to the specified file.

        npoints: int, optional
            Number of points in each axis

        stddev: bool,
            when true also the standard deviation is shown

        kw: optional arguments to pass to `pcolormesh`.
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

        # Get grid sample of points where to evalutate the model
        xaxis = np.linspace(experiment.parameters[xname].lower,
                            experiment.parameters[xname].upper, npoints)
        yaxis = np.linspace(experiment.parameters[yname].lower,
                            experiment.parameters[yname].upper, npoints)
        X, Y = np.meshgrid(xaxis, yaxis)
        xarray = X.flatten()
        yarray = Y.flatten()

        sample = pd.DataFrame({xname: xarray, yname: yarray})
        f_plt, sd_plt = self.evaluate_model(sample, p0=p0)
        
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

        # f_plots = [f_plt, sd_plt]
        # labels = ['value', 'std. deviation']
        f_plots = [f_plt]
        metric_name = list(self.ax_client.experiment.metrics.keys())[0]
        labels = [metric_name]
        if stddev:
            f_plots.append(sd_plt)
            labels.append('std. deviation')

        nplots = len(f_plots)
        fig, axs = plt.subplots(nplots, figsize=(6.4, nplots * 4.8), dpi=100)
        fig.suptitle('Model for metric %s' % metric_name)
        for i, f in enumerate(f_plots):
            if nplots == 1:
                ax = axs
            else:
                ax = axs[i]
            cmap = 'Spectral'
            if (i == 0) and (not minimize):
                cmap = 'Spectral_r'
            im = ax.pcolormesh(xaxis, yaxis, f.reshape(X.shape), cmap=cmap, shading='auto', **kw)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(labels[i])
            ax.set(xlabel=xname, ylabel=yname)
            # adding contour lines with labels
            ax.contour(X, Y, f.reshape(X.shape), levels=20,
                       linewidths=0.5, colors='black', linestyles='solid')
            # plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=6)
            ax.scatter(xtrials, ytrials, s=2, c='black', marker='o')
        plt.tight_layout()

        if filename is not None:
            plt.savefig(filename, dpi=300)
            print('Saving figure to', filename)
