import numpy as np
import pandas as pd

#genereer een 3x4 pandas DataFrame met kolommen Eleanor, Chidi, Tahani en Jason
data = np.random.randint(101, size=(3,4))
KolomNamen = ['Eleanor','Chidi','Tahani','Jason']
DataFrame = pd.DataFrame(data=data,columns=KolomNamen)

#print voll dataframe:
print(DataFrame)

#print waarde van 1ste rij van de Eleanor kolom:
print(DataFrame['Eleanor'].iloc[[1]])

#maak een 5de kolom Janet die de waarde heeft van Tahani+Jason:
DataFrame['Janet'] = DataFrame['Tahani'] + DataFrame['Jason']
