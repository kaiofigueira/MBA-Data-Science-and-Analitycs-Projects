# Diferença entre Padronização (Z-scores) e Normalização (Box-Cox)

# Padronização pelo Z-scores da variável 'comprimento'

from scipy.stats import zscore

df_bebes['zcomprimento'] = zscore(df_bebes['comprimento'])
df_bebes['zcomprimento'].describe()

#%%
#Teste de Shapiro-Francia na variável 'comprimento'

teste_sf = shapiro_francia(df_bebes['comprimento']) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')
    
#%%
#Teste de Shapiro-Francia na variável 'zcomprimento'

teste_sf = shapiro_francia(df_bebes['zcomprimento']) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')
    
#%% Histograma da variável original 'comprimento'

plt.figure(figsize=(15,10))
hist1 = sns.histplot(data=df_bebes['comprimento'], kde=True, bins=25,
                     color = 'aqua', alpha=0.4, edgecolor='silver',
                     line_kws={'linewidth': 3})
hist1.get_lines()[0].set_color('navy')
plt.ylabel('Frequência', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()

#%% Histograma da variável padronizada 'zcomprimento'

plt.figure(figsize=(15,10))
hist1 = sns.histplot(data=df_bebes['zcomprimento'], kde=True, bins=25,
                     color = 'aqua', alpha=0.4, edgecolor='silver',
                     line_kws={'linewidth': 3})
hist1.get_lines()[0].set_color('navy')
plt.ylabel('Frequência', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()