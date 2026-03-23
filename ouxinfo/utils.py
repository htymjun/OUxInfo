import matplotlib.pyplot as plt


def myParams(fontsize=20):
  plt.rcParams['mathtext.fontset'] = 'stix'
  plt.rcParams['xtick.direction']  = 'in'
  plt.rcParams['ytick.direction']  = 'in'
  plt.rcParams['font.size']        = fontsize
  try:
    plt.rcParams['font.family']    = 'Times New Roman'
  except:
    plt.rcParams['font.family']    = 'Liberation Serif'

