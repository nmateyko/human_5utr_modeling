from itertools import product
import numpy as np
import scipy
from sklearn.metrics import pairwise_distances
from collections import Counter

def hamming(seqs1, seqs2):
  return pairwise_distances(X=seqs1, Y=seqs2, metric='hamming')

def hamming_distribution(b, n):
  all = [np.array(i) for i in product(range(b), repeat=n)]
  dists = np.rint(hamming(all, all).flatten() * n)
  return Counter(dists)

counts = hamming_distribution(4, 6)
total = sum(counts.values())
print(total)
print(counts.items())
print([item[1]/total for item in counts.items()])
print(scipy.stats.binom.pmf(range(7), 6, 0.75))
