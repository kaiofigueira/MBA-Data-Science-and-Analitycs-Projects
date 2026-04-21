# Histograma da distribuição Poisson

pois = np.random.poisson(lam=2, size=1000)
plt.hist(pois, bins=8, edgecolor='white')
plt.show()

#%%
# Média da variável 'violations' antes e depois do enforcement legal

df_corruption.groupby('post')['violations'].mean()

#%%
# Cálculo manual do fit do 'modelo_poisson' antes da vigência da lei

modelo_poisson.params

np.exp(2.212739 -4.296762*0 + 0.021870*23 + 0.341765*0.5)

# Cálculo manual do fit do 'modelo_poisson' depois da vigência da lei

np.exp(2.212739 -4.296762*1 + 0.021870*23 + 0.341765*0.5)

#%%
# Histograma da distribuição binomial negativa

theta = 2 # parâmetro de forma da distribuição Poisson-Gama
delta = 2 # taxa de decaimento da distribuição Poisson-Gama

nbinom = np.random.negative_binomial(n= delta, p=delta/(theta + delta), size =1000)
plt.hist(nbinom,bins=10, edgecolor='white')
plt.show()

#%%
# Cálculo manual do LR test
-2*(-2071.79 - (-567.40))

#%%
# Cálculo manual do fit do 'modelo_bneg' antes da vigência da lei

modelo_bneg.params

np.exp(1.946894 -4.274618*0 + 0.040018*23 + 0.452662*0.5)

# Cálculo manual do fit do 'modelo_bneg' depois da vigência da lei

np.exp(1.946894 -4.274618*1 + 0.040018*23 + 0.452662*0.5)
