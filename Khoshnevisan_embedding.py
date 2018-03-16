from math import sqrt
from scipy.stats import norm
import numpy as np
from config import config

import matplotlib.pyplot as plt
import pickle


def simulate_coupling():

  X = np.ones(50)
  c_max = 0.0
  c_time = 0.0 
  c_sum = 0.0
  c = 0 # index
  tau=[0.0]
  B = np.zeros(config.ndt)
  S = np.zeros(config.ndt)
  N = np.zeros(config.ndt)
  A = np.zeros(config.ndt)
  Z = np.zeros(config.ndt)
  while True:
    x = np.random.pareto(2.2)
    found=False
    while not found:
      c+=1
      if c>=config.ndt-1:
        break
      c_time += config.dt
      B[c] = B[c-1]+np.random.normal(scale=np.sqrt(config.dt))
      if c_max < B[c]:
        c_max=B[c]
      elif B[c]<=c_max-x:
        tau.append(c_time)
        c_max = B[c]
        c_sum += x
        S[c] = c_max
        N[c] = c_sum
        break
      S[c] = c_max
      N[c] = c_sum
    if c>=config.ndt-1:
      break

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



  B = -B
  Z=-Z
  t = np.linspace(0,len(B)*config.dt,num=len(B))

  # splot = plt.plot(t[0:len(Z)],S,label='S')
  # bplot = plt.plot(t[0:len(Z)],B[0:len(Z)],label='B')
  # zplot = plt.plot(t[0:len(Z)],Z[0:len(Z)],label='Z')
  # Aplot = plt.plot(t[0:len(Z)],A)
  # Nplot = plt.plot(t[0:len(Z)],N,label='N')
  # thelineplot = plt.plot(theline)
  # print('Is A non-decreasing')
  # print(np.all(np.diff(A) >= 0))

  # plt.legend()
  # plt.show()
  # print(len(Z))
  return [B[0:len(Z)],Z[0:len(Z)]]


def expected_cost(nsamples):
  samples = np.zeros(nsamples)
  for i in range(0,nsamples):
    rtuple = simulate_coupling()
    samples[i] = max(abs(rtuple[0]-rtuple[1]))
  return np.mean(samples)

def clt(nsamples,nmeans):
  means = np.zeros(nmeans)
  samples = np.zeros(nmeans,nsamples)
  for i in range(0,nmeans):
    for j in range(0,nsamples):
      rtuple = simulate_coupling()
      samples[i,j] = max(abs(rtuple[0]-rtuple[1]))
      means[i] = np.mean(samples[i,:])
  print(np.shape(means))
  meansdict = {"means": means, "samples": samples}
  pickle_out = open("vals.pickle","wb")
  pickle.dump(meansdict,pickle_out)
  pickle_out.close()


# def plothist(data):
#   plt.open()


if __name__ == '__main__':
  # rtuple = simulate_coupling()
  # ctuple = expected_cost(100)
  # 
  # clt(100,100)

  # pickle_in = open("vals.pickle","rb")
  # test = pickle.load(pickle_in)
  # means = test["means"]
  # print(test["samples"])
  # pickle_in.close()

  # plt.hist(means,10)
  # plt.show()
  # A = [1,2,3]
  # print([A[i] for i in (1,2)])



