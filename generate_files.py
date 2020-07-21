import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from time import time
from spectra_collection_functions import CollectionParameters, read_pkl_spectra
from sklearn.manifold import TSNE
# from MulticoreTSNE import MulticoreTSNE as TSNE

# --------------------------------------------------------
# ---------------- Read data -----------------------------
# --------------------------------------------------------
print('Reading input GALAH data')
date_string = '20190801'

galah_data_dir = '/shared/mari/cotar/'

# additional data and products about observed spectra
general_data = Table.read(galah_data_dir + 'sobject_iraf_53_reduced_'+date_string+'.fits')
params_data = Table.read(galah_data_dir + 'GALAH_DR3_main_200331.fits')

# auxiliary tsne list
tsne_classes = Table.read(galah_data_dir + 'tsne_classification_dr52_2018_04_09.csv', format='ascii.csv')

spectra_ccd3_pkl = 'galah_dr53_ccd3_6475_6745_wvlstep_0.060_ext4_'+date_string+'.pkl'
# parse interpolation and averaging settings from filename
ccd3_wvl = CollectionParameters(spectra_ccd3_pkl).get_wvl_values()
idx_read_ccd3 = np.where(np.logical_and(ccd3_wvl >= 6550,
                                        ccd3_wvl <= 6650))[0]
ccd3_wvl_use = ccd3_wvl[idx_read_ccd3]

print('Reading resampled GALAH spectra')
spectra_ccd3 = read_pkl_spectra(galah_data_dir + spectra_ccd3_pkl, read_cols=idx_read_ccd3)

# --------------------------------------------------------
# ---------------- Prepare data for students -------------
# --------------------------------------------------------
print('Creating list of spectra')
sobj_selection = list([])

n_per_class = 300
n_random_other = 3000

# select spectra determined by the Gregors' DR2 tSNE projection
for tsne_c in np.unique(tsne_classes['tsne_class']):
    sobj_ids = tsne_classes[tsne_classes['tsne_class'] == tsne_c]['sobject_id']

    if len(sobj_ids) > n_per_class:
        sobj_class = np.random.choice(sobj_ids, size=n_per_class, replace=False)
    else:
        sobj_class = sobj_ids

    print(' Class:', tsne_c, len(sobj_class))
    sobj_selection.append(sobj_class)

# select random spectra from the rest of the data pool
print(' Class: random')
sobj_ids = general_data[np.in1d(general_data['sobject_id'], tsne_classes['sobject_id'], invert=True)]['sobject_id']
sobj_rand = np.random.choice(sobj_ids, size=n_random_other, replace=False)
sobj_selection.append(sobj_rand)

# create a spectral subset
idx_sobj_selection = np.in1d(general_data['sobject_id'], np.hstack(sobj_selection))
final_sobj_selection = general_data['sobject_id'][idx_sobj_selection]
spectra_ccd3_selected = spectra_ccd3[idx_sobj_selection, :]

print('Shape of the final selection', spectra_ccd3_selected.shape)

# --------------------------------------------------------
# ---------------- Save final output ---------------------
# --------------------------------------------------------
# final outputs
print('Saving output file')
np.savez('GALAH_spektri_vaja',
         valovne_dolzine=ccd3_wvl_use,
         galah_spektri=spectra_ccd3_selected)

# TODO: Export stellar parameters

# --------------------------------------------------------
# ---------------- Test run by t-SNE ---------------------
# --------------------------------------------------------
print('Running tSNE projection')

perp = 50
theta = 0.5

tsne_class = TSNE(n_components=2,
                  perplexity=perp,
                  angle=theta,
                  metric='euclidean',
                  method='barnes_hut',
                  n_iter=1000,
                  n_iter_without_progress=350,
                  init='random',
                  # n_jobs=30,  # new in scikit-learn version 0.22
                  verbose=1)

tsne_start = time()
tsne_res = tsne_class.fit_transform(spectra_ccd3_selected)
print(f'Total tSNE time: {(time() - tsne_start):.1f}s')  # f-string notation

# export tsne coordinates
np.savez('GALAH_spektri_tsne_transformacija',
         tsne_koordinate=tsne_res)

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.scatter(tsne_res[:, 0], tsne_res[:, 1], lw=0, s=2, alpha=0.3, c='grey', label='')

# add colours for identified tSNE classes
for tsne_c in np.unique(tsne_classes['tsne_class']):
    sobj_ids = tsne_classes[tsne_classes['tsne_class'] == tsne_c]['sobject_id']

    idx_sobj_tsne = np.in1d(final_sobj_selection, sobj_ids)
    if np.sum(idx_sobj_tsne):
        ax.scatter(tsne_res[idx_sobj_tsne, 0], tsne_res[idx_sobj_tsne, 1], lw=0, s=2, alpha=1., label=tsne_c)

ax.set(xlabel='t-SNE axis 1', ylabel='t-SNE axis 2')
lgnd = ax.legend()
# increase size of dots in legend
for lgnd_item in lgnd.legendHandles:
    try:
        lgnd_item.set_sizes([8.0])
    except:
        pass
fig.tight_layout()
fig.savefig('tsne_test.png', dpi=300)
plt.close(fig)
