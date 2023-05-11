# -*- coding: utf-8 -*-
import numpy as np
import nnoptic as nn


"""alpha_0 = np.array([[0.0424, 0.0449, 0.0395, 0.0400, 0.0635],
                   [0.0884, 0.0580, 0.0857, 0.0560, 0.0397],
                   [0.0496, 0.0523, 0.0902, 0.0784, 0.0386]])
f_0 = np.array([[-0.014, -2.930, 0.292, 0.320, -2.473, 0.012],
               [-0.628, -3.270, -0.080, 0.002, -3.040, 0.012],
               [-0.928, 0.025, -0.536, -2.395, -0.016, -0.031]])"""
u1 = nn.generator_unitary_matrix(4)
print('u1: ', u1)
u2 = nn.generator_unitary_matrix(4)
print('u2: ', u2)
I_coeff = np.random.randint(1, 200, (4, 3))
print('I: ', I_coeff)
a_in = np.random.randint(1, 15, (4,)) + 1j * np.random.randint(1, 15, (4,))
for i in range(1, len(a_in)):
    a_in[i] = 0
np.savetxt('излучение.csv', a_in)
np.savetxt('первая_u.csv', u1)
np.savetxt('вторая_u.csv', u2)
np.savetxt('cила_тока.csv', I_coeff)
