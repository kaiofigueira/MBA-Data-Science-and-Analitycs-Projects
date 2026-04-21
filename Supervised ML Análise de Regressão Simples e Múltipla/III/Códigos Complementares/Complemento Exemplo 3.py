# cpi médio por região:
    
cpi_medio = df_corrupcao.groupby('regiao')['cpi'].mean().reset_index()
cpi_medio

#%%
# Mudança da categoria de referência

df_corrupcao_dummies = pd.get_dummies(df_corrupcao, columns=['regiao'],
                                      dtype=int,
                                      drop_first=False)

df_corrupcao_dummies

#%%
# Escolher a categoria de referência e estimar o modelo

# Definição da fórmula utilizada no modelo
lista_colunas =list(df_corrupcao_dummies.drop(columns=['cpi','pais',
                                                        'regiao_numerico',
                                                        'regiao_Oceania']).columns)
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "cpi ~ " + formula_dummies_modelo
print("Fórmula utilizada: ",formula_dummies_modelo)

# Estimação
modelo_corrupcao_dummies = sm.OLS.from_formula(formula_dummies_modelo,
                                               df_corrupcao_dummies).fit()

# Parâmetros do 'modelo_corrupcao_dummies'
modelo_corrupcao_dummies.summary()

