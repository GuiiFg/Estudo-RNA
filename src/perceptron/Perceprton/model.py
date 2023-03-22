import pandas as pd
import random as rd

class Neuron:
  def __init__(self, fg, teta, wteta = rd.random()):
    self.__wi = []
    self.__initWi = []
    self.__epoch = 0
    self.__teta = teta
    self.__wteta = wteta
    self.__initWteta = wteta
    self.__fg = fg
    self.__lerning_rate = 0

  def trainWithLog(self, train_x:pd.DataFrame, train_y:pd.DataFrame, times:int, path:str = None, max_epochs = 0, lerning_rate = rd.random(), verbose= False):
    initialW = []
    finalW = []
    totalEpochs = []
    dfJson = {}

    for i in range(times):
      self.__epoch = 0
      newWTeta = rd.random()
      self.__wteta = newWTeta
      self.__initWteta = newWTeta
      try:
        _test = dfJson['wT']
        dfJson['wT'].append(self.__initWteta)
      except:
        dfJson['wT'] = [self.__initWteta]
      self.train(train_x, train_y, max_epochs, lerning_rate, verbose)
      for index, w in enumerate(self.__initWi):
        try:
          _test = dfJson[f'w{index}']
          dfJson[f'w{index}'].append(w)
        except:
          dfJson[f'w{index}'] = [w]

      for index, w in enumerate(self.__wi):
        try:
          _test = dfJson[f'w{index}f']
          dfJson[f'w{index}f'].append(w)
        except:
          dfJson[f'w{index}f'] = [w]

      try:
        _test = dfJson['wTf']
        dfJson['wTf'].append(self.__wteta)
      except:
        dfJson['wTf'] = [self.__wteta]

      try:
          _test = dfJson['epochs']
          dfJson['epochs'].append(self.__epoch)
      except:
          dfJson['epochs'] = [self.__epoch]

    df = pd.DataFrame(dfJson)

    if path != None:
      df.to_csv(path, ';', encoding='utf8', index=False)

    return df


  def train(self, train_x:pd.DataFrame, train_y:pd.DataFrame, max_epochs = 0, lerning_rate = rd.random(), verbose= False):

    if verbose:
      print(f'\nstart training: \n - smples: {len(train_y)}\n - features: {len(train_x.columns)}')
    
    self.__lerning_rate = lerning_rate
    num_features = len(train_x.columns)
    randomW = [rd.random() for x in range(num_features)]
    self.__wi = list(randomW)
    self.__initWi = list(randomW)
    erros = 1
    expected_result = list(train_y[train_y.columns[0]])
    epoch_accuracy = 0

    # print(num_features, self.__wi, lerning_rate)

    while((self.__epoch < max_epochs if max_epochs > 0 else True) and erros > 0):
      self.__epoch += 1
      results = self.__predict(train_x, expected_result)

      erros = 0
      erros = [results[i] == expected_result[i] for i in range(len(results))].count(False)

      epoch_accuracy = 1 - (((erros * 100) / len(expected_result)) / 100)

      if verbose:
        print(f'epoch: {self.__epoch} | erros: {erros} | acc: {epoch_accuracy}')

    if verbose:
      print(f'Finish: epochs: {self.__epoch} | converg: {True if erros == 0 else False} | acc: {epoch_accuracy}')

  def __updateWi(self, lerning_rate, expected, result, inputs):
    for i in range(len(self.__wi)):
      self.__wi[i] = float(self.__wi[i] + lerning_rate * (expected - result) * inputs[i])

    self.__wteta = float(self.__wteta + lerning_rate * (expected - result) * self.__teta)

  def __predict(self, train_x:pd.DataFrame, train_y:list = []):
    results = []
    values = train_x.values
    for i in range(len(train_x.values)):
      result = sum([values[i][x] * self.__wi[x] for x in range(len(self.__wi))]) + (self.__teta * self.__wteta)
      resultActivation = self.__fg(result)
      results.append(resultActivation)

      if len(train_y) > 0:
        if resultActivation != train_y[i]:
          # print(f'rate: {self.__lerning_rate} | expec: {train_y.values[i]} | result: {resultActivation} | input: {values[i]}')
          self.__updateWi(self.__lerning_rate, train_y[i], resultActivation, values[i])

    return results
  
  def predict(self, test_x:pd.DataFrame):
    return self.__predict(test_x)

  def printW(self):
    print(self.__wi)
    print(self.__wteta)


if __name__ == '__main__': 

  train_x = pd.DataFrame({
    'x1':[1,2,3,4],
    'x2':[1,2,3,4],
    'x3':[1,2,3,4],
    'x4':[1,2,3,4]
  })

  train_y = pd.DataFrame([1,2,3,4])

  def SigmoidBipolar(x:int):
    if x < 0:
      return -1

    if x > 0:
      return 1

  neuron = Neuron(fg=SigmoidBipolar, teta= -1)

  neuron.train(train_x, train_y, max_epochs= 300, verbose = True)



