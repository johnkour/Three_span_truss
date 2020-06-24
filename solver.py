# This program solves a standard 2D truss with three spans pinned to the ground

# 1.1: Import useful libraries.

import numpy as np
import subfunc as sf

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

# 1.3 & 1.5: Define the load of the truss and the dimensions of it's sections.

# load = np.ones((m, 3))
# load *= 100
# P = np.array(load)

b, t, P = sf.generator(m)

# 3.2: Test the variables b and t for input error.

sf.bounds(b, t)

# 2.2: Calculate the members' internal forces.

F = sf.axial_f(P, S, L, h)
# print(F)

# 1.4: Define the properties of the steel used.

fy = 275            # MPa
k = 0.80
E = 210             # GPa

# print(F)

# 2.3: Calculate the members' resistance.

T_for_cr, C_for_cr = sf.mem_res(S, b, t, fy, E, k)
# print(T_for_cr)
# print(C_for_cr)

# 2.4: Evaluate whether the member fails or not.
Failure = sf.evaluation(F, T_for_cr, C_for_cr)
# print(Failure)

n_fail = np.sum(Failure, axis = 0)
# print(n_fail, m-n_fail)

# 4: Export results to csvs.

message = 'Please enter the path where the file will be stored'
message += ' (Press enter to set path: '
message += 'D:/jkour/Documents/Σχολή/2ο έτος/Εαρινό Εξάμηνο/Προγραμματισμός Η.Υ'
message += '/Εργασία εξαμήνου/MyCode)\n'

pth = input(message)

if len(pth) < 1:
    
    pth = 'D:/jkour/Documents/Σχολή/2ο έτος/Εαρινό Εξάμηνο/Προγραμματισμός Η.Υ'
    pth += '/Εργασία εξαμήνου/MyCode'

tup = (b, t, P)

sf.exporter(pth, 'variables', tup, '%.2f')

tup = (Failure, )

sf.exporter(pth, 'results', tup, '%d')
