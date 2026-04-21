plt.cm.viridis_r

plt.cm.coolwarm

sns.color_palette('viridis_r', as_cmap=True)

sns.color_palette('flare', as_cmap=True)


#%%

modelo.rsquared # R²

modelo.nobs # número de observações

modelo.params # parâmetros

modelo.params.iloc[0] # alpha
modelo.params.iloc[1] # beta

modelo.ess # SQModelo
modelo.ssr # SQResíduos
