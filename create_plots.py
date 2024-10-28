#%%
import pandas as pd

ADCN_XIIoT = pd.read_csv('./results/strategy=ADCN_dataset=XIIoT_num_experiments=10_normalize=True_ADCN_label_mode=random_.csv')
ADCN_WUST = pd.read_csv('./results/strategy=ADCN_dataset=WUST_num_experiments=10_normalize=True_ADCN_label_mode=random_.csv')

ADCN_AE_WUST = pd.read_csv('./results/strategy=ADCN-AE_dataset=WUST.csv')
ADCN_AE_XIIoT = pd.read_csv('./results/strategy=ADCN-AE_dataset=XIIoT.csv')

ADCN_PCA_WUST = pd.read_csv('./results/strategy=ADCN-PCA_dataset=WUST.csv')
ADCN_PCA_XIIoT = pd.read_csv('./results/strategy=ADCN-PCA_dataset=XIIoT.csv')

#%%

ADCN_PCA_WUST = ADCN_PCA_WUST[ADCN_PCA_WUST['Threshold Method'] == 'best_f_score']
ADCN_PCA_XIIoT = ADCN_PCA_XIIoT[ADCN_PCA_XIIoT['Threshold Method'] == 'best_f_score']

ADCN_WUST = ADCN_WUST[ADCN_WUST['Experiment Number'] == 0]
ADCN_XIIoT = ADCN_XIIoT[ADCN_XIIoT['Experiment Number'] == 0]

ADCN_PCA_WUST = ADCN_PCA_WUST[ADCN_PCA_WUST['Threshold Method'] == 'best_f_score']
ADCN_PCA_XIIoT = ADCN_PCA_XIIoT[ADCN_PCA_XIIoT['Threshold Method'] == 'best_f_score']

#%%
def plot_experiences(ADCN_df, UCON_df, name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    nExperiences = int(ADCN_df['Train Experience'].max() + 1)
    ADCN_df['F1'] = ADCN_df['F1'].astype(float) * 100
    UCON_df['F1'] = UCON_df['F1'].astype(float) * 100
    color_palette = ['#db4242', '#59bd6b']

    fig, ax = plt.subplots(1, nExperiences, figsize=(15, 3))
    for i in range(nExperiences):
        ADCN_experience = ADCN_df[ADCN_df['Test Experience'] == i]
        UCON_experience = UCON_df[UCON_df['Test Experience'] == i]
        #Plot F1 vs Accuracy accross train experiences
        sns.lineplot(x='Train Experience', y='F1', data=ADCN_experience, ax=ax[i], color=color_palette[0])
        sns.lineplot(x='Train Experience', y='F1', data=UCON_experience, ax=ax[i], color=color_palette[1])
        ax[i].set_title(f'Test Experience {i}')
        ax[i].set_xticks(range(nExperiences))
        #set y range to 0-100
        ax[i].set_ylim(0, 100)
        #Convert y axis to percentage
        # ax[i].set_yticklabels([f'{int(x)}%' for x in ax[i].get_yticks()])
        ax[i].set_xlabel('Training experience')
        if i == 0:
            ax[i].set_ylabel('F1 Score (%)')
        else:
            ax[i].set_ylabel('')
    

    #Move legend to bottom of plot

    #increase font size
    font_size = 15
    for ax in fig.get_axes():
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.set_title(ax.get_title(), fontsize=font_size)
        ax.set_xlabel(ax.get_xlabel(), fontsize=font_size)
        ax.set_ylabel(ax.get_ylabel(), fontsize=font_size)
     #Add legend
    lgd = fig.legend(['ADCN','UCON-IDS'], bbox_to_anchor=(0, -0.2, 1.05, 0), ncol=2, fontsize=font_size, loc='lower center')

    plt.tight_layout()
    plt.savefig(f'./plots/{name}.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


plot_experiences(ADCN_WUST, ADCN_PCA_WUST, 'WUSTL-IIoT-experiences')
plot_experiences(ADCN_XIIoT, ADCN_PCA_XIIoT, 'X-IIoTID-experiences')

# %%
