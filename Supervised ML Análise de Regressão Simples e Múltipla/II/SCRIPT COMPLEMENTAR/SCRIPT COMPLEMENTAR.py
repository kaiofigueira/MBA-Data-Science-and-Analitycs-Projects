from statsmodels.stats.anova import anova_lm

anova_lm(modelo)

# Cálculo manual da estatística F de Fisher-Snedecor

Fcalc = (1638.851351)/(45.143581)
Fcalc

#%%

# Função densidade de probabilidade da estatística F

df1 = 10
df2 = 80

f_values = np.random.f(df1, df2, 100000)

plt.hist(f_values, bins=100, edgecolor='black', color='bisque')

#%%

# Cálculo do F crítico:
    
from scipy.stats import f

float(f.ppf(q=0.95, dfn=1, dfd=8))

# Cálculo do p-value associado ao F calculado:

float(1 - f.cdf(36.30308705461359, 1, 8))

#%%

# Distribuição t de Student (Gosset)

df = 40
t_values = np.random.standard_t(df,100000)
plt.hist(t_values, bins = 100, edgecolor='black',color='papayawhip')

plt.hist(t_values, bins = 100, color='lightgrey')

from scipy.stats import t

float(t.sf(6.025, 8)*2)

float(t.sf(1.96, 8))


6.025**2

#%%

# Cálculo do R² ajustado - EXEMPLO 2

r2_ajust_manual = (1 - (50-1)/(50-1-2)*(1-0.324))
r2_ajust_manual

# Diretamente...
float(modelo_paises.rsquared_adj)

#%%
# EXEMPLO 3: cpi médio por região

cpi_medio = df_corrupcao.groupby('regiao')['cpi'].mean().reset_index()
cpi_medio

