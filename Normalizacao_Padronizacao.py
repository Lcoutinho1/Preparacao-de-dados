import pandas as pd
from sklearn.preprocessing import RobustScaler , MinMaxScaler, StandardScaler


pd.set_option('display.width', None)
pd.set_option('display.max_colwidth',None)

df=pd.read_csv('clientes-v2-tartados.csv')

print(df.head())

df= df.drop(['data','estado','nivel_educacao','numero_filhos','estado_civil','area_atuacao'],axis=1)

#normalizaçao - minmaxscaler
scaler=MinMaxScaler()
df['idadeMinMaxScaler']= scaler.fit_transform(df[['idade']])
df['salarioMinMaxScaler']= scaler.fit_transform(df[['salario']])

min_max_scaler=MinMaxScaler(feature_range=(-1,1))
df['idadeMinMaxScaler_mm']= min_max_scaler.fit_transform(df[['idade']])
df['salarioMinMaxScaler-mm']= min_max_scaler.fit_transform(df[['salario']])

#padronizaçao - standardscaler
scaler= StandardScaler()
df['idadeStandardScaler']= scaler.fit_transform(df[['idade']])
df['salarioStandardScaler']= scaler.fit_transform(df[['salario']])

#padronizaçao - robustscaler
scaler= RobustScaler()
df['idadeRobusScaler']= scaler.fit_transform(df[['idade']])
df['salarioRobustScaler']= scaler.fit_transform(df[['salario']])

print(df.head())