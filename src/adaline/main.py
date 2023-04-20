import pandas as pd

from src.adaline.Adaline import Neuron, ActivationFunctions

databaseTreino = pd.read_excel('database/Dados_Adaline.xls', sheet_name='Treinamento')
databaseTeste = pd.read_excel('database/Dados_Adaline.xls', sheet_name='Teste')

train_x = databaseTreino.drop(columns=['d'])
train_y = databaseTreino.drop(columns=['x1', 'x2', 'x3', 'x4'])

adaline = Neuron(fg = ActivationFunctions.SigmoidBipolar, fixed_value = 10 ** -6, teta= -1)

adaline.printW()

adaline.trainWithLog(train_x, train_y, times=1, path='C:/Users/Faria/Repos/Estudo-RNA/src/adaline/testes/teste_ad.csv', max_epochs = 999, lerning_rate= 0.0025, verbose= False)

adaline.predict(databaseTeste)

