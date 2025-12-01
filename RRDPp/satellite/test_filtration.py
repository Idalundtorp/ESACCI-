import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import pearsonr

# === CONFIGURATION ===
path = "/dmidata/users/ilo/projects/RRDPp/satellite/Final_files"
SAT = 'CS2'
OBSID = 'SCICEX'
HS = 'NH'
var = 'SID'
file = f'ESACCIplus-SEAICE-RRDP2+-{var}-{OBSID}-{SAT}-{HS}-CCIp-v3p0-rc2.dat'
full_path = f"{path}/{file}"
valid_variables = ['SID']#, 'SD']

# Column names
ObsNames2 = [
    'date','lat','lon','obsSD','obsSIT','obsSIF','satSD','satSIT','satSIF',
    "obsSD_std","obsSD_ln","obsSD_unc","obsSIT_std","obsSIT_ln","obsSIT_unc",
    "obsFRB_std","obsFRB_ln","obsFRB_unc", "QFT", "QFS", "QFG","satSD_std","satSD_ln","satSD_unc",
    "satSIT_std","satSIT_ln","satSIT_unc","satFRB_std","satFRB_ln","satFRB_unc", "index"
]

ObsNames_SID = [
    'date','lat','lon','obsSID','satSID',
    "obsSID_std","obsSID_ln","obsSID_unc", "QFT", "QFS", "QFG",
    "satSID_std","satSID_ln","satSID_unc",
]

# === LOAD DATA ===

if var=='SID':
    data_raw = np.genfromtxt(full_path, dtype=None, skip_header=1, names=ObsNames_SID, encoding='utf-8')
else:
    data_raw = np.genfromtxt(full_path, dtype=None, skip_header=1, names=ObsNames2, encoding='utf-8')
#print(data_raw)
# Preprocess
if var == 'SID':
    data_raw['satSID'][data_raw['satSID'] < 0] = np.nan
    print(np.median(data_raw['satSID_ln']))
    data_raw['obsSID'] = data_raw['obsSID'] + 0.29
else:
    data_raw['satSIT'][data_raw['satSIT'] < 0] = np.nan
    #data_raw['satSIF'] = data_raw['satSIF'] + data_raw['satSD']
    data_raw['obsSIF'] = data_raw['obsSIF'] - data_raw['obsSD']
    data_raw['satSIF'][data_raw['satSIF'] < 0] = np.nan
    data_raw['obsSIF'][data_raw['obsSIF'] < 0] = np.nan
    data_raw['satSD'][data_raw['satSD'] < 0] = np.nan

if OBSID=='AEM-AWI':
    data_raw['satSIT'] = data_raw['satSIT'] + data_raw['satSD']

# === FILTERS ===
def apply_filters(data, filter_name, var):
    data = data.copy()

    if filter_name == "Unfiltered":
        return data

    # Apply global limits
    if filter_name in ["Global Limits", "Global + Obs Count Filtered"]:
        if 'obsSIT' in data.dtype.names:
            data['obsSIT'][data['obsSIT'] > 8] = np.nan
            data['satSIT'][data['satSIT'] > 8] = np.nan
        if 'obsSIF' in data.dtype.names:
            data['obsSIF'][data['obsSIF'] > 1] = np.nan
            data['satSIF'][data['satSIF'] > 1] = np.nan
        if 'obsSD' in data.dtype.names:
            data['obsSD'][data['obsSD'] > 2] = np.nan
            data['satSD'][data['satSD'] > 2] = np.nan
        if 'obsSID' in data.dtype.names:
            data['obsSID'][data['obsSID'] > 6] = np.nan
            data['satSID'][data['satSID'] > 6] = np.nan

    # Apply observation count filters
    if filter_name in ["Obs Count Filtered", "Global + Obs Count Filtered"]:
        if var=='SIT':
            lim = round(np.nanmedian(data['obsSIT_ln'][data['obsSIT_ln']>0])/2)
        elif var=='SD':
            lim = round(np.nanmedian(data['obsSD_ln'][data['obsSD_ln']>0])/2)
        else:
            lim = round(np.nanmedian(data['obsSID_ln'][data['obsSID_ln']>0])/2)
            #lim = data['QFT'] round(np.nanmedian(data['obsSID_ln'][data['obsSID_ln']>0])/2)
            if lim>50:
                lim=50

        #print(lim)
        if var!='SID':
            for var in ["SIT", "SIF", "SD"]:
                obs_ln_name = f'obs{var}_ln'
                sat_ln_name = f'sat{var}_ln'

                # Some variables use FRB for _ln fields (e.g., SIF)
                if var == "SIF":
                    obs_ln_name = 'obsFRB_ln'
                    sat_ln_name = 'satFRB_ln'

                # if obs_ln_name in data.dtype.names and sat_ln_name in data.dtype.names:
                #     bad = (data[obs_ln_name] < lim) | (data[sat_ln_name] < lim)
                #     if f'obs{var}' in data.dtype.names and f'sat{var}' in data.dtype.names:
                #         data[f'obs{var}'][bad] = np.nan
                #         data[f'sat{var}'][bad] = np.nan
        else:
            for var in ["SID"]:
                obs_ln_name = f'obs{var}_ln'
                sat_ln_name = f'sat{var}_ln'

                if obs_ln_name in data.dtype.names and sat_ln_name in data.dtype.names:
                    bad = (data[obs_ln_name] < lim) | (data[sat_ln_name] < lim)
                    #bad = data['QFS'] ==3
                    print(lim)
                    print(len(data[f'obs{var}'][bad] ))
                    print(len(data[f'obs{var}']))
                    if f'obs{var}' in data.dtype.names and f'sat{var}' in data.dtype.names:
                        data[f'obs{var}'][bad] = np.nan
                        data[f'sat{var}'][bad] = np.nan

    return data


# === METRICS FUNCTION ===
def compute_metrics(obs, sat):
    mask = np.isfinite(obs) & np.isfinite(sat)
    if np.sum(mask) == 0:
        return np.nan, np.nan, np.nan, np.nan
    bias = np.nanmedian(obs[mask] - sat[mask])
    corr, _ = pearsonr(obs[mask], sat[mask])
    #print(np.nanmedian(obs[mask]), np.nanstd(obs[mask]), np.nanmedian(sat[mask]), np.nanstd(sat[mask]), bias, corr)
    return np.nanmedian(obs[mask]), np.nanstd(obs[mask]), np.nanmedian(sat[mask]), np.nanstd(sat[mask]), bias, corr

# === MAIN ANALYSIS ===
filter_names = [
    "Unfiltered",
    "Global Limits",
    "Obs Count Filtered",
    "Global + Obs Count Filtered"
]
filtered_datasets = [apply_filters(data_raw, fn, var) for fn in filter_names]

for vv in valid_variables:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    medians_obs = []
    stds_obs = []
    medians_sat = []
    stds_sat = []
    biases = []
    corrs = []

    for filtered_data in filtered_datasets:
        obs = filtered_data[f'obs{vv}']
        sat = filtered_data[f'sat{vv}']
        med_obs, std_obs, med_sat, std_sat, bias, corr = compute_metrics(obs, sat)
        medians_obs.append(med_obs)
        stds_obs.append(std_obs)
        medians_sat.append(med_sat)
        stds_sat.append(std_sat)
        biases.append(bias)
        corrs.append(corr)
    
    x = np.arange(len(filter_names))
    width = 0.2

    ax.bar(x - 1.5*width, medians_obs, width, label='Obs Median')
    ax.bar(x + 0.5*width, medians_sat, width, label='Sat Median')
    ax.bar(x - 0.5*width, stds_obs, width, label='Obs Std')
    ax.bar(x + 0.5*width, stds_sat, width, label='Sat Std')
    ax.bar(x + 1.5*width, biases, width, label='Bias (Obs - Sat)')

    # Add correlation values as text
    for i, c in enumerate(corrs):
        ax.text(x[i], max(medians_obs[i], medians_sat[i]) + 0.05, f'R={c:.2f}, MD={biases[i]:.2f}', ha='center', fontsize=10, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(filter_names)
    ax.set_ylabel(vv)
    ax.set_title(f"{vv} Metrics under Different Filtering")
    ax.legend(loc='lower left')
    ax.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'{OBSID}_metrics_bar_{vv}.png')
    plt.close()

def plot_scatter_all_filters(datasets_by_filter, dataset_labels, variables):
    for var in variables:
        plt.figure(figsize=(10, 10))
        plt.title(f"Observed vs Satellite {var} — All Filters")

        for label, data in zip(dataset_labels, datasets_by_filter):
            if f'obs{var}' in data.dtype.names and f'sat{var}' in data.dtype.names:
                index = np.isfinite(data[f'obs{var}']) & np.isfinite(data[f'sat{var}'])
                if np.sum(index) > 2:
                    corr = np.round(pearsonr(data[f'obs{var}'][index], data[f'sat{var}'][index])[0], 2)
                    bias = np.nanmedian(data[f'obs{var}'][index] - data[f'sat{var}'][index])
                    plt.scatter(
                        data[f'obs{var}'][index],
                        data[f'sat{var}'][index],
                        label=f"{label} (R: {corr}, MD={bias:.2f})",
                        alpha=0.6,
                        s=5
                    )

        plt.xlabel(f"Observed {var}")
        plt.ylabel(f"Satellite {var}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{OBSID}_scatter_{var}.png')
        plt.close()
        #plt.show()

plot_scatter_all_filters(filtered_datasets, filter_names, valid_variables)
