## EXEMPLO 7:

df_saeb_rend['codigo'] = df_saeb_rend['codigo'].astype('str')

df_saeb_rend.info()

df_saeb_rend.describe()

#%%
# Teste de Breusch-Pagan

df_saeb_rend['up'] = ((df_saeb_rend['residuos'])**2)/\
    (((df_saeb_rend['residuos'])**2).sum()/25530)

modelo_aux = sm.OLS.from_formula('up ~ fitted',
                                 df_saeb_rend).fit()
modelo_aux.summary()

from statsmodels.stats.anova import anova_lm
anova_table = anova_lm(modelo_aux, typ=2)
anova_table

SQReg = anova_table.sum_sq.iloc[0]/2 # chi2 com 1 grau de liberdade
SQReg

p_value = stats.chi2.pdf(SQReg, 1)*2
p_value
