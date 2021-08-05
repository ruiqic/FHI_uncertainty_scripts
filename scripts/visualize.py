"""
While in jupyter notebook, use `%matplotlib inline` to make plots show up.
"""

import ase
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union

# Set figure defaults
width = 6  # Because it looks good
figsize = (width, width)
fontsize = 12
rc = {'figure.figsize': (width, width),
      'font.size': fontsize,
      'axes.labelsize': fontsize,
      'axes.titlesize': fontsize,
      'xtick.labelsize': fontsize,
      'ytick.labelsize': fontsize,
      'legend.fontsize': fontsize}
sns.set(rc=rc)
sns.set_style('ticks')


def rms_dict(x_ref, x_pred):
    """ Takes two datasets of the same shape and returns a dictionary containing RMS error data"""

    x_ref = np.array(x_ref)
    x_pred = np.array(x_pred)

    if np.shape(x_pred) != np.shape(x_ref):
        raise ValueError('WARNING: not matching shapes in rms')

    error_2 = (x_ref - x_pred) ** 2

    average = np.sqrt(np.average(error_2))
    std_ = np.sqrt(np.var(error_2))

    return {'rmse': average, 'std': std_}

def energy_plot(in_atoms, out_atoms, ax, title='Plot of energy'):
    """ Plots the distribution of energy per atom on the output vs the input"""

    # list energies
    ener_in = [at.info["energy"] / len(at.get_chemical_symbols()) for at in in_atoms]
    ener_out = [at.info["energy"] / len(at.get_chemical_symbols()) for at in out_atoms]
    # scatter plot of the data
    ax.scatter(ener_in, ener_out)
    # get the appropriate limits for the plot
    for_limits = np.array(ener_in +ener_out)   
    elim = (for_limits.min() - 0.05, for_limits.max() + 0.05)
    ax.set_xlim(elim)
    ax.set_ylim(elim)
    # add line of slope 1 for refrence
    ax.plot(elim, elim, c='k')
    # set labels
    ax.set_ylabel('energy by MTP / eV')
    ax.set_xlabel('energy by DFT / eV')
    #set title
    ax.set_title(title)
    # add text about RMSE
    _rms = rms_dict(ener_in, ener_out)
    rmse_text = 'RMSE:\n' + str(np.round(_rms['rmse'], 3)) + ' +- ' + str(np.round(_rms['std'], 3)) + 'eV/atom'
    ax.text(0.9, 0.1, rmse_text, transform=ax.transAxes, fontsize='large', horizontalalignment='right', 
            verticalalignment='bottom')
    
def force_plot(in_atoms, out_atoms, ax, symbol='HO', title='Plot of force'):
    """ Plots the distribution of firce components per atom on the output vs the input 
        only plots for the given atom type(s)"""
    
    # extract data for only one species
    in_force, out_force = [], []
    for at_in, at_out in zip(in_atoms, out_atoms):
        # get the symbols
        sym_all = at_in.get_chemical_symbols()
        # add force for each atom
        for j, sym in enumerate(sym_all):
            if sym == symbol:
                in_force.append(at_in.arrays["forces"][j])
                #out_force.append(at_out.get_forces()[j]) \  
                out_force.append(at_out.arrays['forces'][j])
    # convert to np arrays, much easier to work with
    #in_force = np.array(in_force)
    #out_force = np.array(out_force)
    # scatter plot of the data
    ax.scatter(in_force, out_force)
    # get the appropriate limits for the plot
    for_limits = np.array(in_force + out_force)   
    flim = (for_limits.min() - 1, for_limits.max() + 1)
    ax.set_xlim(flim)
    ax.set_ylim(flim)
    # add line of 
    ax.plot(flim, flim, c='k')
    # set labels
    ax.set_ylabel('force by MTP / (eV/Å)')
    ax.set_xlabel('force by DFT / (eV/Å)')
    #set title
    ax.set_title(title)
    # add text about RMSE
    _rms = rms_dict(in_force, out_force)
    rmse_text = 'RMSE:\n' + str(np.round(_rms['rmse'], 3)) + ' +- ' + str(np.round(_rms['std'], 3)) + 'eV/Å'
    ax.text(0.9, 0.1, rmse_text, transform=ax.transAxes, fontsize='large', horizontalalignment='right', 
            verticalalignment='bottom')
    
    
def scatter_parity(train_true, train_predict, valid_true, valid_predict, elements=["Cu", "C"]):
    """
    plot a series of parity scatter plots on training and validation data.
    The input atoms are expected to have fields:
        .info["energy"]
        .arrays["forces"]
    
    train_true: ase.Atoms
        training set labeled with target true values
    train_predict: ase.Atoms
        training set labeled with predicted values
    valid_true: ase.Atoms
        validation set labeled with target true values
    valid_predict: ase.Atoms
        validation set labeled with predicted values
    elements:
        list of elements to have separate forces plots for
        
    returns: None
    
    """
    
    nrows = 1 + len(elements)
    fig, ax_list = plt.subplots(nrows=nrows, ncols=2, gridspec_kw={'hspace': 0.3})
    fig.set_size_inches(13, 20)
    ax_list = ax_list.flat[:]
    
    energy_plot(train_true, train_predict, ax_list[0], 'Energy on training data')
    energy_plot(valid_true, valid_predict, ax_list[1], 'Energy on validation data')
    i = 2
    for elem in elements:
        force_plot(train_true, train_predict, ax_list[i], elem, f'Force on training data - {elem}')
        force_plot(valid_true, valid_predict, ax_list[i+1], elem, f'Force on validation data - {elem}')
        i += 2

    

def get_proportion_lists_vectorized(residuals, y_std, num_bins):
    """
    Return lists of expected and observed proportions of points falling into
    intervals corresponding to a range of quantiles.
    """
    
    residuals = residuals + (residuals == 0) * 1e-12
    y_std = y_std + (y_std == 0) * 1e-12
    
    # Compute proportions
    exp_proportions = np.linspace(0, 1, num_bins)
    in_exp_proportions = exp_proportions

    norm = stats.norm(loc=0, scale=1)
    gaussian_lower_bound = norm.ppf(0.5 - in_exp_proportions / 2.0)
    gaussian_upper_bound = norm.ppf(0.5 + in_exp_proportions / 2.0)

    normalized_residuals = (residuals.flatten() / y_std.flatten()).reshape(-1, 1)
    above_lower = normalized_residuals >= gaussian_lower_bound
    below_upper = normalized_residuals <= gaussian_upper_bound

    within_quantile = above_lower * below_upper
    obs_proportions = np.sum(within_quantile, axis=0).flatten() / len(residuals)

    return exp_proportions, obs_proportions

def plot_calibration(residuals, y_std, num_bins=100):
    """
    Plot calibration curve
    
    residuals: array-like
    y_std: array-like
        uncertainties
    num_bins: int
        number of points used for plotting.
        
    return: None
    """
    
    predicted_pi, observed_pi = get_proportion_lists_vectorized(residuals, y_std, num_bins)
    # Plot the calibration curve
    fig_cal = plt.figure(figsize=figsize)
    ax_ideal = sns.lineplot(x=[0, 1], y=[0, 1])
    _ = ax_ideal.lines[0].set_linestyle('--')
    ax_gp = sns.lineplot(x=predicted_pi, y=observed_pi)
    ax_fill = plt.fill_between(predicted_pi, predicted_pi, observed_pi,
                               alpha=0.2, label='miscalibration area')
    _ = ax_ideal.set_xlabel('Expected cumulative distribution')
    _ = ax_ideal.set_ylabel('Observed cumulative distribution')
    _ = ax_ideal.set_xlim([0, 1])
    _ = ax_ideal.set_ylim([0, 1])

    # Calculate the miscalibration area.
    polygon_points = []
    for point in zip(predicted_pi, observed_pi):
        polygon_points.append(point)
    for point in zip(reversed(predicted_pi), reversed(predicted_pi)):
        polygon_points.append(point)
    polygon_points.append((predicted_pi[0], observed_pi[0]))
    polygon = Polygon(polygon_points)
    x, y = polygon.exterior.xy # original data
    ls = LineString(np.c_[x, y]) # closed, non-simple
    lr = LineString(ls.coords[:] + ls.coords[0:1])
    mls = unary_union(lr)
    polygon_area_list =[poly.area for poly in polygonize(mls)]
    miscalibration_area = np.asarray(polygon_area_list).sum()

    # Annotate the plot with the miscalibration area
    plt.text(x=0.95, y=0.05,
             s='Miscalibration area = %.3f' % miscalibration_area,
             verticalalignment='bottom',
             horizontalalignment='right',
             fontsize=12)
    
    
def plot_sharpness(stdevs, units="eV/atom"):
    """
    Plot histogram of uncertainties.
    Calculates sharpness and dispersion
    
    stdevs: array-like
        list of uncertainties
    units: string
        units of uncertainty measurement. Used in label text.
        
    return: None
    
    """
    
    # Plot sharpness curve
    #xlim = [0, 0.006]
    fig_sharp = plt.figure(figsize=figsize)
    ax_sharp = sns.histplot(stdevs, kde=False)
    
    #ax_sharp.set_xlim(xlim)
    ax_sharp.set_xlabel(f'Predicted standard deviation ({units})')
    ax_sharp.set_ylabel('Normalized frequency')
    ax_sharp.set_yticklabels([])
    ax_sharp.set_yticks([])
    xlim = ax_sharp.get_xlim()
    # Calculate and report sharpness/dispersion
    sharpness = np.sqrt(np.mean(stdevs**2))
    _ = ax_sharp.axvline(x=sharpness, label='sharpness')
    dispersion = np.sqrt(((stdevs - stdevs.mean())**2).sum() / (len(stdevs)-1)) / stdevs.mean()
    if sharpness < (xlim[0] + xlim[1]) / 2:
        text = '\n  Sharpness = %.4f %s\n  C$_v$ = %.3f' % (sharpness, units, dispersion)
        h_align = 'left'
    else:
        text = '\nSharpness = %.4f %s  \nC$_v$ = %.3f  ' % (sharpness, units, dispersion)
        h_align = 'right'
    _ = ax_sharp.text(x=sharpness, y=ax_sharp.get_ylim()[1],
                      s=text,
                      verticalalignment='top',
                      horizontalalignment=h_align,
                      fontsize=fontsize)
    
def plot_energy_parity(energy_df, subset="train"):
    """
    create a energy parity plot with histplot showing densities of points
    
    energy_df: pandas.DataFrame
        columns: "dft value", "predicted value"
    subset: string
        "set" column filter: "train" or "valid"
        
    return: None
    """

    data = energy_df[energy_df["set"]==subset]
    g = sns.jointplot(x="predicted value", y="dft value", data=data, alpha=0.7, kind="hist", hue="structure", height=6,
                      hue_order=["small", "large"], bins=20, marginal_kws=dict(bins=20, multiple="stack", hue_order=["small", "large"]))
    #g.plot_marginals(sns.histplot, color="r", clip_on=True, hue="structure")
    ax = g.ax_joint
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    ax.set(xlabel="energy by model (eV)", ylabel="energy by DFT (eV)", xlim=lims, ylim=lims)
    # Draw a line of x=y 
    ax.plot(lims, lims, '-k', zorder=0)
    # RMSE
    _rms = rms_dict(energy_df[energy_df["set"]==subset]["dft value"], energy_df[energy_df["set"]==subset]["predicted value"])
    rmse_text = 'RMSE:\n' + str(np.round(_rms['rmse'], 3)) + ' +- ' + str(np.round(_rms['std'], 3)) + 'eV/atom'
    ax.text(0.9, 0.1, rmse_text, transform=ax.transAxes, fontsize='large', horizontalalignment='right', 
                verticalalignment='bottom')
    
    
def plot_forces_parity(forces_df, subset="train", species="C"):
    """
    create a forces parity plot with histplot showing densities of points
    
    energy_df: pandas.DataFrame
        columns: "dft value", "predicted value"
    subset: string
        "set" column filter: "train" or "valid"
    species: string
        element name to plot
        
    return: None
    """
    data = forces_df[(forces_df["set"]==subset) & (forces_df["species"]==species)]
    g = sns.jointplot(x="predicted value", y="dft value", data=data, alpha=0.7, kind="hist", hue="structure", height=6,
                      hue_order=["small", "large"], bins=20, marginal_kws=dict(bins=20, multiple="stack", hue_order=["small", "large"]))
    #g.plot_marginals(sns.histplot, color="r", clip_on=True, hue="structure")
    ax = g.ax_joint
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    ax.set(xlabel="force by model (eV/Å)", ylabel="force by DFT (eV/Å)", xlim=lims, ylim=lims)
    # Draw a line of x=y 
    ax.plot(lims, lims, '-k', zorder=0)
    # RMSE
    _rms = rms_dict(data["dft value"], data["predicted value"])
    rmse_text = 'RMSE:\n' + str(np.round(_rms['rmse'], 3)) + ' +- ' + str(np.round(_rms['std'], 3)) + 'eV/Å'
    ax.text(0.9, 0.1, rmse_text, transform=ax.transAxes, fontsize='large', horizontalalignment='right', 
            verticalalignment='bottom')
    
    
def uncertainty_residual_scatter(res_df, units="eV", xlim=None, ylim=None):
    """
    side by side scatter plot for train and validation set, residual vs uncertainty
    hue on large vs small structure for visual
    
    res_df: pandas.DataFrame
        expected columns: "residual", "uncertainty", "structure", "set"
    units: string
        units of measurement for text label
    {x,y}lim: pair-tuple-like
        cutoff limits on the axes
    
    return: None
    """

    g = sns.FacetGrid(res_df, col="set", height=6, xlim=xlim, ylim=ylim)
    g.map_dataframe(sns.scatterplot, "uncertainty", "residual", hue="structure", linewidth=0, alpha=0.7, hue_order=["small", "large"])
    g.set_axis_labels(f"uncertainty ({units})", f"residual ({units})")
    g.set_titles(col_template="{col_name} set")
    g.add_legend()
    