from functions import *
pd.options.mode.chained_assignment = None  # default='warn'

### Set Working Directory
abspath = os.path.abspath(__file__)
dirname = os.path.dirname(abspath)
os.chdir(dirname)

dir_data = 'data'
dir_figures = 'figures'
dir_results = 'results'

### Load and prepare data
dict_freq_phase = loadmat(os.path.join(dir_data, 'Freq_Phase.mat'))
vec_freq = dict_freq_phase['freqs'][0]
vec_phase = dict_freq_phase['phases'][0]

# list_subject_data = loadData(dirname, '.mat')  # load all subject data
cca_mat_result = np.load(os.path.join(dir_results, 'cca_mat_result.npy'))
cca_mat_b = np.load(os.path.join(dir_results, 'cca_mat_b.npy'))
cca_mat_b_thresh = np.load(os.path.join(dir_results, 'cca_mat_b_thresh.npy'))
cca_mat_max = np.load(os.path.join(dir_results, 'cca_mat_max.npy'))
cca_mat_time = np.load(os.path.join(dir_results, 'cca_mat_time.npy'), allow_pickle=True)

fbcca_mat_result = np.load(os.path.join(dir_results, 'fbcca_mat_result.npy'))
fbcca_mat_b = np.load(os.path.join(dir_results, 'fbcca_mat_b.npy'))
fbcca_mat_b_thresh = np.load(os.path.join(dir_results, 'fbcca_mat_b_thresh.npy'))
fbcca_mat_max = np.load(os.path.join(dir_results, 'fbcca_mat_max.npy'))
fbcca_mat_time = np.load(os.path.join(dir_results, 'fbcca_mat_time.npy'), allow_pickle=True)

mat_result_ext_cca = np.load(os.path.join(dir_results, 'mat_result_ext_cca.npy'))
mat_time_ext_cca = np.load(os.path.join(dir_results, 'mat_time_ext_cca.npy'), allow_pickle=True)

## Convert to pandas dataframe
Ns = 35
Nb = 6
Nf = 40
fs = 250  # sampling frequency in hz

df_bcca = mk_df(cca_mat_result, cca_mat_b_thresh, cca_mat_time, cca_mat_max, vec_freq, Nf, Ns, Nb)
df_bfbcca = mk_df(fbcca_mat_result, fbcca_mat_b_thresh, fbcca_mat_time, fbcca_mat_max, vec_freq, Nf, Ns, Nb)

df_cca = make_df(cca_mat_result, vec_freq, vec_phase, Nf, Ns, Nb, cca_mat_time)
df_ext_cca = make_df(mat_result_ext_cca, vec_freq, vec_phase, Nf, Ns, Nb, mat_time_ext_cca)
df_fbcca = make_df(fbcca_mat_result, vec_freq, vec_phase, Nf, Ns, Nb, fbcca_mat_time)

# convert to subject wise representation
df_subject = pd.DataFrame()

df_subject['Accuracy CCA'] = df_cca.groupby(['Subject']).sum()['Compare'] / (Nb * Nf) * 100
df_subject['Accuracy ext CCA'] = df_ext_cca.groupby(['Subject']).sum()['Compare'] / (Nb * Nf) * 100
df_subject['Accuracy FBCCA'] = df_fbcca.groupby(['Subject']).sum()['Compare'] / (Nb * Nf) * 100

df_subject['Time CCA'] = df_cca.groupby(['Subject']).mean()['Time'] / 1000
df_subject['Time FBCCA'] = df_fbcca.groupby(['Subject']).mean()['Time'] / 1000
df_subject['Time Ad CCA'] = df_ext_cca.groupby(['Subject']).mean()['Time'] / 1000

df_subject['ITR CCA'] = df_subject[['Accuracy CCA', 'Time CCA']].apply(itr, axis=1)
df_subject['ITR FBCCA'] = df_subject[['Accuracy FBCCA', 'Time FBCCA']].apply(itr, axis=1)
df_subject['ITR Ad CCA'] = df_subject[['Accuracy ext CCA', 'Time Ad CCA']].apply(itr, axis=1)

# Plot
palette = sns.color_palette('Greys')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
sns.barplot(ax=ax1, data=df_subject[['Accuracy CCA', 'Accuracy FBCCA', 'Accuracy ext CCA']], ci=95, palette='Greys',
            capsize=.1, orient='h')
ax1.set_yticklabels(['CCA', 'FBCCA', 'Extended \n CCA'])
ax1.set_xlabel('Accuracy in %')
set_style(fig1, ax1)
set_size(fig1, 3, 2.2)
# plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
sns.barplot(ax=ax2, data=df_subject[['Time CCA', 'Time FBCCA', 'Time Ad CCA']], ci=95, palette='Greys', capsize=.1,
            orient='h')
ax2.set_yticklabels(['CCA', 'FBCCA', 'Extended \n CCA'])
ax2.set_xlabel('Time elapsed in s')
set_style(fig2, ax2)
set_size(fig2, 3, 2.2)
# plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
sns.barplot(ax=ax3, data=df_subject[['ITR CCA', 'ITR FBCCA', 'ITR Ad CCA']], ci=95, palette='Greys', capsize=.1,
            orient='h')
ax3.set_yticklabels(['CCA', 'FBCCA', 'Extended \n CCA'])
ax3.set_xlabel('ITR')
set_style(fig3, ax3)
set_size(fig3, 3, 2.2)
# plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

fig1.savefig(os.path.join(dir_figures, 'accuracy.pdf'), dpi=300)
fig1.savefig(os.path.join(dir_figures, 'accuracy.png'), dpi=300)
fig2.savefig(os.path.join(dir_figures, 'time.pdf'), dpi=300)
fig2.savefig(os.path.join(dir_figures, 'time.png'), dpi=300)
fig3.savefig(os.path.join(dir_figures, 'itr.pdf'), dpi=300)
fig3.savefig(os.path.join(dir_figures, 'itr.png'), dpi=300)

print("=====================================")
print(
    "Accuracy CCA Mean: " + str(df_subject['Accuracy CCA'].mean()) + ", Std: " + str(df_subject['Accuracy CCA'].std()))
print("Accuracy FBCCA Mean: " + str(df_subject['Accuracy FBCCA'].mean()) + ", Std: " + str(
    df_subject['Accuracy FBCCA'].std()))
print("Accuracy Extended CCA Mean: " + str(df_subject['Accuracy ext CCA'].mean()) + ", Std: " + str(
    df_subject['Accuracy ext CCA'].std()))
print("=====================================")

print(
    "Time CCA Mean: " + str(df_subject['Time CCA'].mean()) + ", Std: " + str(df_subject['Time CCA'].std()))
print("Time FBCCA Mean: " + str(df_subject['Time FBCCA'].mean()) + ", Std: " + str(
    df_subject['Time FBCCA'].std()))
print("Time Extended CCA Mean: " + str(df_subject['Time Ad CCA'].mean()) + ", Std: " + str(
    df_subject['Time Ad CCA'].std()))
print("=====================================")

print(
    "ITR CCA Mean: " + str(df_subject['ITR CCA'].mean()) + ", Std: " + str(df_subject['ITR CCA'].std()))
print("ITR FBCCA Mean: " + str(df_subject['ITR FBCCA'].mean()) + ", Std: " + str(
    df_subject['ITR FBCCA'].std()))
print("ITR Extended CCA Mean: " + str(df_subject['ITR Ad CCA'].mean()) + ", Std: " + str(
    df_subject['ITR Ad CCA'].std()))
print("=====================================")

fig4, ax4 = plot_trial(cca_mat_result)
fig4.savefig(os.path.join(dir_figures, 'cca_freq.pdf'), dpi=300)
fig4.savefig(os.path.join(dir_figures, 'cca_freq.png'), dpi=300)

fig5, ax5 = plot_trial(fbcca_mat_result)
fig5.savefig(os.path.join(dir_figures, 'fbcca_freq.pdf'), dpi=300)
fig5.savefig(os.path.join(dir_figures, 'fbcca_freq.png'), dpi=300)

fig6, ax6 = plot_trial(mat_result_ext_cca)
fig6.savefig(os.path.join(dir_figures, 'ad_cca_freq.pdf'), dpi=300)
fig6.savefig(os.path.join(dir_figures, 'ad_cca_freq.png'), dpi=300)
