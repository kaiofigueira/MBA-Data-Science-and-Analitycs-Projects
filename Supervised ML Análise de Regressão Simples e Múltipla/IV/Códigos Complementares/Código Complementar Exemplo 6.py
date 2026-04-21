## EXEMPLO 6:
    
# Modelo 1 Auxiliar 1, somente com a preditora 'rh1'

modelo1_aux1 = sm.OLS.from_formula('salario ~ rh1', df_salarios).fit()

modelo1_aux1.summary()

modelo1_aux1.rsquared**0.5

#%%
# Modelo 1 Auxiliar 2, somente com a preditora 'econometria1'

modelo1_aux2 = sm.OLS.from_formula('salario ~ econometria1', df_salarios).fit()

modelo1_aux2.summary()

modelo1_aux2.rsquared**0.5

#%%
#Procedimento Stepwise no 'modelo1'

from statstests.process import stepwise

modelo1_step = stepwise(modelo1, pvalue_limit=0.05)

#%%
# Modelo 1 Auxiliar 3, rodando 'rh1 ~ econometria1'

modelo1_aux3 = sm.OLS.from_formula('rh1 ~ econometria1', df_salarios).fit()

modelo1_aux3.summary()

modelo1_aux3.rsquared**0.5

#%% Cálculo da Tolerancee VIF

tolerance1 = 1 - modelo1_aux3.rsquared
tolerance1

VIF1 = 1/tolerance1
VIF1

#%%
# Modelo 2 Auxiliar , rodando 'rh2 ~ econometria2'

modelo2_aux1 = sm.OLS.from_formula('rh2 ~ econometria2', df_salarios).fit()

modelo2_aux1.summary()

#%% Cálculo da Tolerancee VIF

tolerance2 = 1 - modelo2_aux1.rsquared
tolerance2

VIF2 = 1/tolerance2
VIF2

#%%
# Procedimento Stepwise ao 'modelo2'

from statstests.process import stepwise

modelo2_step = stepwise(modelo2, pvalue_limit=0.05)

# BOA NOTÍCIA! STEPWISE TRATA DESSE FENÔMENO!

