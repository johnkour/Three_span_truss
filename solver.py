# This program solves a standard 2D truss with three spans pinned to the ground

# 1.1: Import useful libraries.

import numpy as np
import subfunc as sf
import generator as gen
import pandas as pd

# 1.2: Define the geometry of the truss.

m = input(
    'Please enter the number of training examples ' +
    '(or press enter fordefault value: 500k)\n')

if len(m) < 1:  m = 500 * 10**3

m = sf.ex_num(m)

L = np.ones((m, 1))
L *= 20
h = np.ones((m, 1))
h *= 1.5
# print(L, h)

# 2.1: Calculate the members' length.

S = sf.mem_length(L, h)
# print(S)

# print(S)

# 1.3: Define the load of the truss.

load = np.ones((m, 3))
load *= 100
P = np.array(load)

# 2.2: Calculate the members' internal forces.

F = sf.axial_f(P, S, L, h)
# print(F)

# 1.4: Define the properties of the steel used.

fy = 275            # MPa
k = 0.80
E = 210             # GPa

# print(F)

# 1.5: Define the dimensions of the sections.

b, t, temp = gen.gen(m)

del temp

# b_lst = [[140, 100, 80, 170, 80, 170, 80, 170, 80, 100, 140],
#          [140, 100, 80, 170, 80, 170, 80, 170, 80, 100, 140]]
# b = np.array(b_lst)     # width in mm.
# print(np.shape(b))

# t_lst = [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
#          [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]
# t = np.array(t_lst)     # thickness in mm.

# # 3.2: Test the variables b and t for input error.

sf.bounds(b, t)

# 2.3: Calculate the members' resistance.

T_for_cr, C_for_cr = sf.mem_res(S, b, t, fy, E, k)
# print(T_for_cr)
# print(C_for_cr)

# 2.4: Evaluate whether the member fails or not.
Failure = sf.evaluation(F, T_for_cr, C_for_cr)
print(Failure)

n_fail = np.sum(Failure, axis = 0)
print(n_fail, m-n_fail)

# 4: Export results to csvs.

pth = 'D:/jkour/Documents/Σχολή/2ο έτος/Εαρινό Εξάμηνο/Προγραμματισμός Η.Υ'
pth += '/Εργασία εξαμήνου/MyCode'

variables = np.concatenate((b, t, P), axis = 1)
# print(variables)
df1 = pd.DataFrame(variables)
# print(df)

pth1 = pth + '/variables.csv'
df1.to_csv(pth1, float_format = '%.2f', header = False, index = False)

df2 = pd.DataFrame(Failure)
# print(df)

pth2 = pth + '/results.csv'
df2.to_csv(pth2, float_format = '%d', header = False, index = False)
