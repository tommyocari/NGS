import numpy as np
import pandas as pd

import sys
sys.path.append('/home/tommaso/Downloads/PGM-master/source/')
import bm

print(float(sys.argv[1])+1)

data = pd.read_csv('../../data/Byrne.csv',index_col=0).query('T0 > 0').sort_values('T0', ascending=False).iloc[1:]

counts = data['T0'].to_numpy()
data = data.iloc[:,:7].to_numpy()

model = bm.BM(N=7, nature='Potts', n_c=20, random_state=0)
data = data.astype('int16')
counts = counts.astype('float32')

model.fit(data, weights=counts, l2=float(sys.argv[1]), optimizer='ADAM',n_iter=10)
np.save('ByrneT0_fields_%.2e.npy'%float(sys.argv[1]),model.layer.fields)
np.save('ByrneT0_couplings_%.2e.npy'%float(sys.argv[1]),model.layer.couplings)