# MODELO SÓ COM O ENDIVIDAMENTO

modelo_auxiliar1 = sm.OLS.from_formula('retorno ~ endividamento',
                                       df_empresas).fit()

# Parâmetros do 'modelo_auxiliar1'
modelo_auxiliar1.summary()

#%%
# MODELO COM A EXCLUSÃO MANUAL DO ENDIVIDAMENTO

modelo_auxiliar2 = sm.OLS.from_formula('retorno ~ disclosure +\
                                       ativos + liquidez',
                                       df_empresas).fit()

# Parâmetros do 'modelo_auxiliar2'
modelo_auxiliar2.summary()

#%%
# MODELO SÓ COM O DISCLOSURE

modelo_auxiliar3 = sm.OLS.from_formula('retorno ~ disclosure',
                                       df_empresas).fit()

# Parâmetros do 'modelo_auxiliar3'
modelo_auxiliar3.summary()

#%%
# MODELO COM A EXCLUSÃO DO DISCLOSURE

modelo_auxiliar4 = sm.OLS.from_formula('retorno ~ ativos + liquidez',
                                       df_empresas).fit()

# Parâmetros do 'modelo_auxiliar4'
modelo_auxiliar4.summary()
