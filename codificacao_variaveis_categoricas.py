import pandas as pd
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.width',None)

df= pd.read_csv('clientes-v2-tratados.csv')

print(df.head())

#codificaçao one-hot para  'estado_civil'
df=pd.concat([df,pd.get_dummies(df['estado_civil'], prefix='estado_civil')],axis=1)

print("\nDataFrame apos codificaçao one-hot para 'estado_civil':\n",df.head())

#codificaçao ordinal para 'nivel_educaçao'
educacao_ordem={'Ensino Fundamental':1,'Ensino Medio':2,'Ensino Superior':3,'Pos-graduaçao':4}
df['nivel_educacao_ordinal']= df['nivel_educacao'].map(educacao_ordem)

print("\nDataFrame apos transformar 'area_atuacao' em codigos numericos:\n",df.head())

#tranformar 'area_atuacao' em categorias codificadas usando o metodo .cat.codes
df['area_atuacao_cod']= df['area_atuacao'].astype('category').cat.codes

print("\nDataFrame apos transformar 'area_atuacao'  em codigos numericos:\n",df.head())

#LabelEncoder para 'estado'
#LabelEncoder converter cada valor unico em numeros de 0 a n_classes_1
label_encoder= LabelEncoder()
df['estado_cod']= label_encoder.fit_transform(df['estado'])
print("\nDataFrame apos aplicar LabelEncoder  em 'estado':\n",df.head())
