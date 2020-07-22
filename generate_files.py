import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, join
from time import time
from spectra_collection_functions import CollectionParameters, read_pkl_spectra
from sklearn.decomposition import PCA
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
general_data = join(general_data, params_data, keys='sobject_id', join_type='left')

# auxiliary tsne list
tsne_classes = Table.read(galah_data_dir + 'tsne_classification_dr52_2018_04_09.csv', format='ascii.csv')

spectra_ccd3_pkl = 'galah_dr53_ccd3_6475_6745_wvlstep_0.060_ext4_'+date_string+'.pkl'
# parse interpolation and averaging settings from filename
ccd3_wvl = CollectionParameters(spectra_ccd3_pkl).get_wvl_values()
idx_read_ccd3 = np.where(np.logical_and(ccd3_wvl >= 6550,
                                        ccd3_wvl <= 6675))[0]
ccd3_wvl_use = ccd3_wvl[idx_read_ccd3]

print('Reading resampled GALAH spectra')
spectra_ccd3 = read_pkl_spectra(galah_data_dir + spectra_ccd3_pkl, read_cols=idx_read_ccd3)

# --------------------------------------------------------
# ---------------- Prepare data for students -------------
# --------------------------------------------------------
print('Creating list of spectra')
sobj_selection = list([])

n_per_class = 300
n_random_other = 4000

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
# ---------------- Save final outputs --------------------
# --------------------------------------------------------
# final spectroscopic outputs
print('Saving output file')
np.savez('GALAH_spektri_vaja',
         valovne_dolzine=ccd3_wvl_use,
         galah_spektri=spectra_ccd3_selected)

# prepare only the most essential stellar parameters
params_data_out = general_data['sobject_id', 'teff', 'fe_h', 'logg', 'vbroad', 'vmic', 'alpha_fe', 'flag_sp'][idx_sobj_selection]
params_data_out['class'] = '                        '

# mark a subset of selected spectra with the training class flags
# at the same time create plots for selected examples
n_flag_p_class = 10
for tsne_c in np.unique(tsne_classes['tsne_class']):
    fig, ax = plt.subplots(1, 1, figsize=(11, 3))

    sobj_ids = tsne_classes[np.logical_and(tsne_classes['tsne_class'] == tsne_c,
                                           np.in1d(tsne_classes['sobject_id'], params_data_out['sobject_id']))]['sobject_id']

    sobj_class = np.random.choice(sobj_ids, size=n_flag_p_class, replace=False)
    idx_class_mark = np.in1d(params_data_out['sobject_id'], sobj_class)

    params_data_out['class'][idx_class_mark] = tsne_c

    for idx_spec in np.where(idx_class_mark)[0]:
        ax.plot(ccd3_wvl_use, spectra_ccd3_selected[idx_spec, :], lw=0.2)

    ax.set(xlabel=u'Valovna dolzina [$\AA$]', ylabel=u'Normaliziran fluks',
           xlim=[ccd3_wvl_use[0], ccd3_wvl_use[-1]], ylim=[0.4, 1.1])
    ax.grid(ls='--', color='black', alpha=0.2)
    fig.tight_layout()
    fig.savefig('primeri_razred_'+tsne_c+'.pdf')
    plt.close(fig)

# export stellar parameters
params_data_out.write('GALAH_spektri_vaja_parametri.fits', overwrite=True)

# --------------------------------------------------------
# ---------------- Test run by t-SNE ---------------------
# --------------------------------------------------------
print('Running tSNE projection')

perp = 80
theta = 0.5

tsne_class = TSNE(n_components=2,
                  perplexity=perp,
                  angle=theta,
                  metric='euclidean',
                  method='barnes_hut',
                  n_iter=1000,
                  n_iter_without_progress=350,
                  init='random',
                  n_jobs=8,  # new in scikit-learn version 0.22
                  verbose=1
                  )

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

ax.set(xlabel='t-SNE koordinata 1', ylabel='t-SNE koordinata 2')
lgnd = ax.legend()
# increase size of dots in legend
for lgnd_item in lgnd.legendHandles:
    try:
        lgnd_item.set_sizes([10.0])
    except:
        pass

fig.tight_layout()
fig.savefig('tsne_test.png', dpi=300)
plt.close(fig)

# --------------------------------------------------------
# ---------------- Test run by PCA -----------------------
# --------------------------------------------------------
print('Running PCA decomposition')

n_pca_comp = 6
pca_class = PCA(n_components=n_pca_comp)
pca_res = pca_class.fit_transform(spectra_ccd3_selected)

# export pca components
np.savez('GALAH_spektri_pca_komponente',
         tsne_koordinate=pca_res)

# plot all components
fig, ax = plt.subplots(n_pca_comp, n_pca_comp,
                       figsize=(15, 15), sharex='col', sharey='row')

for i_x in range(n_pca_comp):
    for i_y in range(n_pca_comp):
        ax[i_y, i_x].scatter(pca_res[:, i_x], pca_res[:, i_y], lw=0, s=2, alpha=0.3, c='black', label='')

        if i_x == 0:
            ax[i_y, i_x].set(ylabel=f'Komponenta {i_y+1:d}',
                             ylim=np.percentile(pca_res[:, i_y], [0.5, 99.5]))

        if i_y == n_pca_comp - 1:
            ax[i_y, i_x].set(xlabel=f'Komponenta {i_x+1:d}',
                         xlim=np.percentile(pca_res[:, i_x], [0.5, 99.5]))

fig.align_xlabels()
fig.align_ylabels()

fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)
fig.savefig('pca_test_vse.png', dpi=250)
plt.close(fig)

# plot only the first two components
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.scatter(pca_res[:, 0], pca_res[:, 1], lw=0, s=2, alpha=0.3, c='grey', label='')

# add colours for identified tSNE classes
for tsne_c in np.unique(tsne_classes['tsne_class']):
    sobj_ids = tsne_classes[tsne_classes['tsne_class'] == tsne_c]['sobject_id']
    idx_sobj_tsne = np.in1d(final_sobj_selection, sobj_ids)

    if np.sum(idx_sobj_tsne):
        ax.scatter(pca_res[idx_sobj_tsne, 0], pca_res[idx_sobj_tsne, 1], lw=0, s=3, alpha=1., label=tsne_c)

ax.set(xlabel='PCA komponenta 1', ylabel='PCA komponenta 2',
       xlim=np.percentile(pca_res[:, 0], [0.5, 99.5]),
       ylim=np.percentile(pca_res[:, 1], [0.5, 99.5])
       )
lgnd = ax.legend()
# increase size of dots in legend
for lgnd_item in lgnd.legendHandles:
    try:
        lgnd_item.set_sizes([10.0])
    except:
        pass

fig.tight_layout()
fig.savefig('pca_test_2d.png', dpi=300)
plt.close(fig)
