import random
import numpy as np
from scipy import spatial
from sklearn.metrics import pairwise_distances
from timeit import default_timer as timer

def hamming(seqs1, seqs2):
  dists = []
  for seq in seqs1:
    for seq2 in seqs2:
      dists.append(sum(c1 != c2 for c1, c2 in zip(seq, seq2)))
  return dists

def np_hamming(seqs1, seqs2):
  return pairwise_distances(X=seqs1, Y=seqs2, metric='hamming')

def np_split(seqs):
  return np.array([np.fromiter(seq, (np.compat.unicode,1)) for seq in seqs])

def py_split(seqs):
  return [np.ndarray([char for char in seq]) for seq in seqs]

def compare_speed(funcs, *args):
  for func in funcs:
    start = timer()
    func(*args)
    end = timer()
    print(f"Time for {func.__name__}: {end - start}")

random_seqs = np_split([random.choices(['A', 'T', 'G', 'C'], k=50) for i in range(500)])
random_seqs2 = np_split([random.choices(['A', 'T', 'G', 'C'], k=50) for i in range(500)])

uniques = np.unique(random_seqs)
X = np.searchsorted(uniques, random_seqs)
Y = np.searchsorted(uniques, random_seqs2)


compare_speed([hamming, np_hamming], X, Y)
