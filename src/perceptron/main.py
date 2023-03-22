import pandas as pd

from src.perceptron.Perceprton import Neuron, ActivationFunctions

databaseTreino = pd.read_excel('database/Dados.ods', sheet_name='Treinamento')
databaseTeste = pd.read_excel('database/Dados.ods', sheet_name='Teste')

train_x = databaseTreino.drop(columns=['d'])
train_y = databaseTreino.drop(columns=['x1', 'x2', 'x3'])

perceptron = Neuron(fg = ActivationFunctions.SigmoidBipolar, teta= -1)

perceptron.printW()

perceptron.trainWithLog(train_x, train_y, times=10, path='C:/Users/Guilherme/Repos/Estudo-RNA/src/perceptron/testes/teste.csv', max_epochs = 999, lerning_rate= 0.01, verbose= False)

perceptron.predict(databaseTeste)

