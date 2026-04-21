# Log-likelihood:

modelo_atrasos.llf

# Cálculo do LLR p-value (chi2): teste de razão de verossimilhança

modelo_nulo = sm.Logit.from_formula('atrasado ~ 1',
                                    df_atrasado).fit()

# Parâmetros do 'modelo_nulo'
modelo_nulo.summary()

modelo_nulo.llf

# chi2 é análogo ao teste F do modelo de regressão estimado por OLS
chi2 = -2*(modelo_nulo.llf - modelo_atrasos.llf)
chi2

pvalue = stats.distributions.chi2.sf(chi2, 2)
pvalue

#%%
# Cálculo do pseudo R² de McFadden

pseudoR2MF = ((-2*modelo_nulo.llf)-(-2*modelo_atrasos.llf))/\
    (-2*modelo_nulo.llf)
pseudoR2MF

#%%
# Cálculo manual do predict da célula [1.4]

(1)/(1 + np.exp(-(-26.1665 + 0.1904*7 + 2.3629*10)))

#%%
# Cálculo da Patrícia
modelo_atrasos.predict(pd.DataFrame({'dist':[13.3], 'sem':[10]}))

(1)/(1 + np.exp(-(-26.1665 + 0.1904*13.3 + 2.3629*10)))
