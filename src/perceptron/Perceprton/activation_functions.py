
class ActivationFunctions():

  @staticmethod
  def SigmoidBipolar(x:int):
    if x < 0:
      return -1

    if x > 0:
      return 1

if __name__ == '__main__':
  funcs = ActivationFunctions
  a = funcs.SigmoidBipolar
  type(a)
  # a(2) == 1
  