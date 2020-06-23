# This program solves a standard 2D truss with three spans pinned to the ground

# 1.1: Import useful libraries.

import numpy as np
from subfunc import mem_length, axial_f, mem_res, evaluation, bounds

# 1.2: Define the geometry of the truss.

L = np.array([[20], [20]])
h = np.array([[1.5], [1.5]])
print(L, h)

# 2.1: Calculate the members' length.

S = mem_length(L, h)
print(S)

# print(S)

# 1.3: Define the load of the truss.

load = np.ones((np.shape(L)[0], 3))
load *= np.array([[100], [50]])
P = np.array(load)

# 2.2: Calculate the members' internal forces.

F = axial_f(P, S, L, h)
print(F)

# 1.4: Define the properties of the steel used.

fy = 275            # MPa
k = 0.80
E = 210             # GPa

# print(F)

# 1.5: Define the dimensions of the sections.

b_lst = [[140, 100, 80, 170, 80, 170, 80, 170, 80, 100, 140],
         [140, 100, 80, 170, 80, 170, 80, 170, 80, 100, 140]]
b = np.array(b_lst)     # width in mm.
print(np.shape(b))

t_lst = [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]
t = np.array(t_lst)     # thickness in mm.

# # 3.2: Test the variables b and t for input error.

bounds(b, t)

# 2.3: Calculate the members' resistance.

T_for_cr, C_for_cr = mem_res(S, b, t, fy, E, k)
# print(T_for_cr)
# print(C_for_cr)

# 2.4: Evaluate whether the member fails or not.
Failure = evaluation(F, T_for_cr, C_for_cr)
print(Failure)