from itertools import product
import random
import numpy as np

bp_dict = {
  'AA': 0,
  'AC': 1,
  'AG': 2,
  'AT': 3,
  'CA': 4,
  'CC': 5,
  'CG': 6,
  'CT': 7,
  'GA': 8,
  'GC': 9,
  'GG': 10,
  'GT': 11,
  'TA': 12,
  'TC': 13,
  'TG': 14,
  'TT': 15
}

# seq = "ATGCATGC"
# prod = [bp_dict["".join(i)] for i in product(seq, repeat=2)]
# print(prod)

# map2D = []
# for i in range(len(seq)):
#   map2D.append(np.array(prod[i*len(seq): (i + 1)*len(seq)]))

# a = np.reshape(prod, (len(seq), len(seq)))
# print(a)

# print((np.arange(a.max()) == a[...,None]-1).astype(int))

def one_hot_2D(seqs, bp_dict):
  result = []
  for i, seq in enumerate(seqs):
    interactions = [bp_dict["".join(i)] for i in product(seq, repeat=2)]
    interactions_2D = np.reshape(interactions, (len(seq), len(seq)))
    result.append(np.array((np.arange(interactions_2D.max()) == interactions_2D[...,None]-1)))
  return np.asarray(result)

random_seqs = [random.choices(['A', 'T', 'G', 'C'], k=50) for i in range(200000)]

# repr2D = one_hot_2D(random_seqs, bp_dict)
# print(repr2D[0])

# Assumes all sequences in seqs are the same length
def structure_2D(seqs):
  result = []
  l = len(seqs[0])
  diag_zeros = np.ones((l, l))
  for i in range(-3, 4):
    diag_zeros *= np.diag(-1 * np.ones(l - abs(i)), i) + 1
  bp = (('A', 'T'), ('T', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'T'), ('T', 'G'))
  for i, seq in enumerate(seqs):
    bp_potential = [int(pair in bp) for pair in product(seq, repeat=2)]
    bp_2D = np.reshape(bp_potential, (l, l)) * diag_zeros
    result.append(bp_2D)
  return np.asarray(result)

# print(structure_2D([['A', 'T', 'G', 'G', 'T', 'A', 'C', 'T', 'C', 'A', 'T']]))
print(len(structure_2D(random_seqs)))