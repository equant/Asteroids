import sys, os, glob, re
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
#sys.path.append("../asteroids")
#import gaps
import emcee

matplotlib.rcParams['font.size'] = 14

annotation_font = {
        'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 8
}

timestamp = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')

######################################
#### BEGIN user defined variables ####

save_figures = False
save_dir     = '.'

#hexbin_x_size = 0.05  # au
hexbin_x_size = 0.02  # au
#hexbin_x_size = 0.01  # au
#hexbin_x_size = 0.002 # au

# objects should be a dataframe that contains 'H' and 'a' columns
objects = pd.read_csv('sample_data.csv')

regions = {
    'Hungarias'         : [1.78, 2],
    #'Main Belt'         : [2.12, 3.25],
    #'Inner Belt'        : [2.12, 2.5],
    #'Middle Belt'       : [2.5, 2.96],
    #'Inner-Middle Belt' : [2.5, 2.83],
    #'Outer-Middle Belt' : [2.83, 2.96],
    #'Outer Belt'        : [2.96, 3.25],
    #'Hildas'            : [3.92, 4.004],
    #'Trojans'           : [5.095, 5.319],
}

#### END user defined variables ####
####################################


def completion_limit_model(xs, C):
    ymodel = -5*np.log10(xs * (xs-1))
    ymodel += C
    return ymodel

def lnprior(p):
    C = p
    try:
        if (-50 < C < 50):
            return 0.0
    except:
        print("lnprior: Something is wrong with C: {}".format(C))
        print("    it is a {}".format(type(C)))
        return -np.inf
    return -np.inf

def lnlike(p, xdata, ydata, yerrors):
    C = p
    ymodel = completion_limit_model(xdata, C)
    #noise  = H_bin_uncertainty
    lnlikelihood = (-0.5 * ((ymodel - ydata) / yerrors)**2).sum()
    #print("lnlikelihood: {}".format(lnlikelihood))
    #print("fitting: {}".format(C))
    return lnlikelihood

def lnprob(p, xdata, ydata, yerrors):
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(p, xdata, ydata, yerrors)

fiducial_error = (objects['H'].max() - objects['H'].min())/2

most_populated_H_at_distance = []
most_populated_H_at_distance_errors = []
aus        = []
nH_per_bin = []
min_H_at_distance = []
max_H_at_distance = []

_H  = objects['H'].max() - objects['H'].min()
_au = objects['a'].max() - objects['a'].min()
au_bins = pd.cut(objects['a'], int(_au/hexbin_x_size))

au_groups = objects.groupby(au_bins)
for interval, au_group in au_groups:
    if len(au_group) == 0:
        continue
    H_bins   = pd.cut(au_group['H'], int(_H*2))
    H_groups = au_group.groupby(H_bins)

    most_pop_H = H_groups.count()['H'].idxmax().mid
    nH         = sum(H_groups.count()['H'])
    au         = interval.mid

    aus.append(interval.mid)
    most_populated_H_at_distance.append(most_pop_H)
    nH_per_bin.append(nH)

    error = fiducial_error / np.sqrt(nH)
    most_populated_H_at_distance_errors.append(error)

most_populated_H_at_distance = np.array(most_populated_H_at_distance)
most_populated_H_at_distance_errors = np.array(most_populated_H_at_distance_errors)
xdata = np.array(aus)
ydata = most_populated_H_at_distance

fig, ax = plt.subplots(1,1, figsize=(15,6))
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

## Now we loop through different regions that we want to fit separately...

for_df_regions  = list(regions.keys())
for_df_inner = [x[0] for x in regions.values()]
for_df_outer = [x[1] for x in regions.values()]
for_df_C        = []
for_df_err      = []
for_df_Cerr     = []   # Combined string of C +/- error
for_df_e        = []
for_df_e_err    = []
for_df_eerr     = []   # Combined string of e +/- error
for_df_n        = []
for_df_Hlim     = []

interactive_df = {}

for name, (fit_min, fit_max) in regions.items():
    print(name)
    region_objects = objects[ (objects['a'] >= fit_min) & (objects['a'] <= fit_max)]

    # interactive_df is useful for interactive data analysis in ipython
    interactive_df[name] = {}
    interactive_df[name]['objects'] = region_objects

    color = '#002480'
    error_color = '#7D96D3'

    fitmask = (xdata>fit_min) & (xdata<fit_max)    # au

    ndim, nwalkers, nsteps = 1, 14, 10000
    pos = [[np.random.uniform(low=-50, high=50)] for i in range(nwalkers)]

    # Do the below so that our fiducial errors are confined to the region we are looking at.
    # as opposed to some of the fiducial error code above which considers the entire "objects"
    # region.
    region_errors = np.array((region_objects['H'].max() - region_objects['H'].min()) / np.sqrt(nH_per_bin))

    # Again, this df is made for interactive exploration
    # of data in ipython
    interactive_df[name]['errors'] = region_errors[fitmask]
    interactive_df[name]['x'] = xdata[fitmask]
    interactive_df[name]['y'] = ydata[fitmask]
    interactive_df[name]['mask'] = fitmask

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(xdata[fitmask], ydata[fitmask], region_errors[fitmask]))
    sampler.run_mcmc(pos, nsteps)

    #########################
    ## -- Plot Fitting

    burn_in_steps = 120
    samples = sampler.chain[:, burn_in_steps:, :].reshape((-1, ndim))

    if name == "Main Belt":
        ax.plot(xdata, completion_limit_model(xdata, np.median(samples)), color=color, alpha=1, label='model', zorder=0)
    else:

        # Plot the median model...
        ax.plot(xdata[fitmask], completion_limit_model(xdata[fitmask], np.median(samples)), color=color, alpha=1, label='model', zorder=11)

        # Plot the upper and lower confience interval...
        percentile = 99.7
        upper_C = np.percentile(samples, percentile)
        lower_C = np.percentile(samples, 100-percentile)
        ax.fill_between(
                xdata[fitmask],
                completion_limit_model(xdata[fitmask], upper_C),
                completion_limit_model(xdata[fitmask], lower_C),
                facecolor=error_color,
                alpha=0.8,
                label='model',
                zorder=10,
        )
    print("    Fit Constant (C): {:.4g} +/- {:.4g}".format(np.median(samples), samples.std()))

    for_df_C.append("{:.4g}".format(np.median(samples)))
    for_df_err.append("{:.4g}".format(samples.std()))
    for_df_Cerr.append("{:.4g} \pm {:.2g}".format(np.median(samples), samples.std()))

    for_df_e.append("{:.4g}".format(np.median(region_objects['e'])))
    for_df_e_err.append("{:.4g}".format(np.median(region_objects['e'])))
    for_df_eerr.append("{:.2g} \pm {:.2g}".format(np.median(region_objects['e']), region_objects['e'].std()))
    for_df_n.append(len(region_objects))

    Hlim_for_region = completion_limit_model(np.median(region_objects['a']), np.median(samples))
    Hlim_for_region_error = completion_limit_model(np.median(region_objects['a']), samples).std()
    for_df_Hlim.append("{:.4g} \pm {:.2g}".format(Hlim_for_region, Hlim_for_region_error))
    print("    Completion Limit: {:.4g} +/- {:.2g} (mag)".format(Hlim_for_region, Hlim_for_region_error))


    result_string = """
    C: {:.4g} +/- {:.3g}
    objects fit:  {}
    fit annulus:  {}-{} au
    bin min/max:  {}/{}
    hexbin width: {:.2g} au
    """.format(
                np.median(samples),
                samples.std(),
                sum(nH_per_bin),
                fit_min,
                fit_max,
                min(nH_per_bin),
                max(nH_per_bin),
                hexbin_x_size,
              )
    ax.errorbar(xdata[fitmask], ydata[fitmask], yerr=region_errors[fitmask], label='data', zorder=1, color='#666666', alpha=1, fmt='.k')


# Plot error bars for all data.
#ax.errorbar(xdata, ydata, yerr=most_populated_H_at_distance_errors, label='data', zorder=0, color='black', alpha=0.3, fmt='o')

#ax.errorbar(x, y, yerr=yerr, fmt=".k")
ax.set_xlabel("a (au)", fontsize=18)
ax.set_ylabel("H (mag)", fontsize=18)

#ax.set_ylim([12.5,20])
#ax.set_xlim([1.9,3.5])
#ax.set_xlim([1.6,5.5])

plotFileName = "multi_completion_limit-fitting-{}binwidth".format(hexbin_x_size)
plotFilePath = save_dir + "/" + plotFileName
plt.show()
if save_figures:
    print("Saving figure/s to: {}".format(plotFilePath))
    fig.savefig(plotFilePath + ".pdf", bbox_inches='tight')
    fig.savefig(plotFilePath + ".png", dpi=300)
plt.close()



data = [
for_df_regions,
for_df_inner,
for_df_outer,
for_df_n,
for_df_Cerr,
for_df_Hlim,
for_df_eerr,
]

columns = [
"regions",
"inner",
"outer",
"N",
"Cerr",
"Hlim",
"e",
]

#df = pd.DataFrame().from_items(zip(columns, data)) # from_items depricated
#df = pd.DataFrame.from_dict((zip(columns, data)))
#print(df.to_latex(index=False))

