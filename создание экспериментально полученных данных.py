import numpy as np


alpha_exp = np.array([0.13958110, 0.27407124, 0.00445478, 0.02223744])
I_exp = np.array([[0], 
                  [19.2683955], 
                  [6.42211315], 
                  [6.98511124]])
a_init_exp = np.array([[-26.5671234 + 10.4641286*1j],
                       [0 + 0*1j],
                       [0 + 0*1j],
                       [0 + 0*1j]])
np.savetxt('альфа_эксп.csv', alpha_exp)
np.savetxt('I_эксп.csv', I_exp)
np.savetxt('a_init_эксп.csv', a_init_exp)