## EXEMPLO 8:

# Estimação do modelo só com 'renda' como preditora

modelo_renda = sm.OLS.from_formula('despmed ~ renda',
                                   df_planosaude).fit()

# Parâmetros do modelo
modelo_renda.summary()

# RENDA PASSA!