from math import sqrt
from scipy.stats import norm
import numpy as np
from config import config

import matplotlib.pyplot as plt

def simulate_coupling():

  X = np.ones(50)
  c_max = 0.0
  c_time = 0.0 
  c_sum = 0.0
  c = 0 # index
  tau=[0.0]
  B = [0.0]
  S = [0.0]
  N = [0.0]
  A = [0.0]
  Z = [0.0]
  for x in X:
    found=False
    while not found:
      c+=1
      c_time += config.dt
      B.append(B[c-1]+np.random.normal(scale=np.sqrt(config.dt)))
      if c_max < B[c]:
        c_max=B[c]
      elif B[c]<=c_max-x:
        tau.append(c_time)
        c_max = B[c]
        c_sum += x
        S.append(c_max)
        N.append(c_sum)
        break
      S.append(c_max)
      N.append(c_sum)


  S = np.array(S)
  B=np.array(B)
  Z=np.array(Z)
  N=np.array(N)
  A = S+N


  t_last=0
  # sigma = np.zeros(len(B),dtype=np.int8)
  sigma = []
  for t1 in range(0,len(B)):
    for t2 in range(0,len(B)):
      if A[t2]>= config.m_1*t1*config.dt:
        sigma.append(t2)
        # sigma[t1]=int(t2)
        # t_last = t2
        break


  # print('len of sigma')
  # print(len(sigma))
  # print(sigma)

  theline = [i*config.dt*config.m_1 for i in range(0,len(B))]
  Z = np.array([S[i] for i in sigma])
  # Z = -Z
  # B = -np.array(B)



  t = np.linspace(0,len(B)*config.dt,num=len(B))

  # splot = plt.plot(t[0:len(Z)],S,label='S')
  bplot = plt.plot(t[0:len(Z)],-B[0:len(Z)],label='B')
  zplot = plt.plot(t[0:len(Z)],-Z[0:len(Z)],label='Z')
  # Aplot = plt.plot(t[0:len(Z)],A)
  # Nplot = plt.plot(t[0:len(Z)],N,label='N')
  # thelineplot = plt.plot(theline)
  print('len of B')
  print(len(B))
  # print('Is A non-decreasing')
  # print(np.all(np.diff(A) >= 0))

  plt.legend()
  plt.show()
  # print(len(Z))







  

if __name__ == '__main__':
  simulate_coupling()
  # A = [1,2,3]
  # print([A[i] for i in (1,2)])



