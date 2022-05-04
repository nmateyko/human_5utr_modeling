from itertools import product
from itertools import groupby
import numpy as np
from sklearn.metrics import pairwise_distances
from collections import Counter
import random
from tqdm import tqdm
from timeit import default_timer as timer

def hamming(seqs1, seqs2):
  return pairwise_distances(X=seqs1, Y=seqs2, metric='hamming')

def hamming_distribution(b, n):
  all = [np.array(i) for i in product(range(b), repeat=n)]
  dists = np.rint(hamming(all, all).flatten() * n)
  return Counter(dists)

# counts = hamming_distribution(4, 6)
# total = sum(counts.values())
# print(total)
# print(counts.items())
# print([item[1]/total for item in counts.items()])
# print(scipy.stats.binom.pmf(range(7), 6, 0.75))

# random_seqs = [random.choices(range(4), k=50) for i in range(5000)]
# random_seqs2 = [random.choices(range(4), k=50) for i in range(5000)]

def max_run_py(seqs1, seqs2, min_len):
  similar = []
  for seq1 in seqs1:
    for seq2 in seqs2:
      equal = [int(i == j) for i, j in zip(seq1, seq2)]
      if max(sum(1 for _ in l if n == 1) for n, l in groupby(equal)) >= min_len:
        similar.append((seq1, seq2))
  return similar


def compare_speed(funcs, *args):
  for func in funcs:
    start = timer()
    func(*args)
    end = timer()
    print(f"Time for {func.__name__}: {end - start}")

def max_run(seqs1, seqs2, min_len):
  rows = len(seqs1)
  cols = len(seqs2)
  seq_len = len(seqs1[0])
  a = (seqs1[:, None, :] == seqs2[None, :, :]).reshape(cols * rows, seq_len)
  m,n = a.shape
  A = np.zeros((m,n+2), dtype=bool)
  A[:m,1:-1] = a

  dA = np.diff(A)
  nz = np.nonzero(dA)
  start = (nz[0][::2], nz[1][::2])
  end = (nz[0][1::2], nz[1][1::2])

  run_lengths = end[1]-start[1]
  argmax_run = np.where(run_lengths >= min_len)
  row = start[0][argmax_run]
  return (list(seqs1[row // cols]), list(seqs2[row % cols]), list(run_lengths[argmax_run]))

seqs1 = np.array([random.choices(['A', 'T', 'G', 'C'], k=50) for i in range(5000)])
seqs2 = np.array([random.choices(['A', 'T', 'G', 'C'], k=50) for i in range(500)])

def find_long_runs(train, test, chunk_size, min_len):
  runs_train = []
  runs_test = []
  lengths = []
  for i in range(0, len(train), chunk_size):
    for j in range(0, len(test), chunk_size):
      seqs1 = train[i:i + chunk_size]
      seqs2 = test[j:j + chunk_size]
      res = max_run(seqs1, seqs2, min_len)
      runs_train += res[0]
      runs_test += res[1]
      lengths += res[2]
      for seq1, seq2 in zip(res[0], res[1]):
        print(f"{''.join(seq1)}\n{''.join(seq2)}\n\n")
  return (runs_train, runs_test, lengths)

res = find_long_runs(seqs1, seqs2, 500, 12)
print(res[2])
for seq1, seq2 in zip(res[0], res[1]):
  print(f"{''.join(seq1)}\n{''.join(seq2)}\n\n")