import pandas as pd

def A1(sm):
  dummyA1 = pd.DataFrame()
  for j in range(0,k):
    dummyA1[j] = np.prod(sm[sm.columns[0:j+1]], axis = 1)*(-1)**(j+1)
  return dummyA1.sum(axis=1)
    
def A2(sm):
  dummyA2 = pd.DataFrame()
  for j in range(0,k):
    dummyA2[j] = (abs(4*sm[sm.columns[j]]-2)+a2[j])/(1+a2[j])
  return (dummyA2.product(axis=1))
    
def B1(sm):
  dummyB1 = pd.DataFrame()
  for j in range(0,k):
    dummyB1[j] = (k-sm[sm.columns[j]])/(k-0.5)
  return dummyB1.product(axis=1)
    
def B2(sm):
  dummyB2 = pd.DataFrame()
  for j in range(0,k):
    dummyB2[j] = (sm[sm.columns[j]])**(1/k)
  return dummyB2.product(axis=1)*((1+1/k)**k)
    
def B3(sm):
  dummyB3 = pd.DataFrame()
  for j in range(0,k):
    dummyB3[j] = (abs(4*sm[sm.columns[j]]-2)+b3[j])/(1+b3[j])
  return (dummyB3.product(axis=1))
    
def C1(sm):
  dummyC1 = pd.DataFrame()
  for j in range(0,k):
    dummyC1[j] = pd.Series(abs(4*sm[sm.columns[j]]-2))
  return (dummyC1.product(axis=1))
    
def C2(sm):
  return (sm.product(axis=1)*2**k)

functions = [A1, A2, B1, B2, B3, C1, C2]
