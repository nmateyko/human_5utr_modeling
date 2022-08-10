import numpy as np
import pandas as pd
from itertools import product

test_df = pd.DataFrame({'utr': ['ATGC', 'GTCA', 'TTCG'], 'rl': [3, 4, 5]})

def one_hot_encode(df, col='utr'):

  seq_len = len(df['utr'][0])
  # Dictionary returning one-hot encoding of nucleotides. 
  nuc_d = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0]}
  
  # Create empty matrix.
  vectors=np.empty([len(df),seq_len,4])
  
  # Iterate through UTRs and one-hot encode
  for i, seq in enumerate(df[col].astype(str)): 
    seq = seq.lower()
    a = np.array([nuc_d[x] for x in seq])
    vectors[i] = a
  return vectors

# seq_onehot = one_hot_encode(test_df)
# print(seq_onehot[2][0])

def kmer_dict(n):
  kmer_encoding = {}
  for i, kmer in enumerate(product(['a', 'c', 'g', 't'], repeat=n)):
    zeros = [0] * 4 ** n
    zeros[i] = 1
    kmer_encoding["".join(kmer)] = zeros
  return kmer_encoding

def featurize(df, k, col='utr'):
  nuc_d = kmer_dict(k)
  seq_len = len(df['utr'][0])
  vectors = np.empty([len(df), seq_len - k  + 1, 4 ** k])
  
  for i, seq in enumerate(df[col].astype(str)):
    print(i)
    seq = seq.lower()
    encoding = []
    for j in range(seq_len - k + 1):
      print(j)
      encoding.append(nuc_d[seq[j:j + k]])
    vectors[i] = np.array(encoding)
  return vectors

print(one_hot_encode(test_df))
print(featurize(test_df, 3))