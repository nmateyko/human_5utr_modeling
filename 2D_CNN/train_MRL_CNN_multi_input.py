import argparse
import numpy as np
import pandas as pd
from itertools import product
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--epochs", type=int)
parser.add_argument("--history", type=str)
parser.add_argument("--no_uAUG", action='store_true')
args = parser.parse_args()

def train_model(data, y, valid, num_filters_1D=(128,128,128), num_filters_2D=(32,64,128),
                kernel_sizes_1D=(8,8,8), kernel_sizes_2D=((3,3),(3,3),(3,3)),
                number_nodes_1D=(64,), number_nodes_2D=(64,), number_nodes_end=(128,),
                dropout_1D=(0.2,), dropout_2D=(0.2,), dropout_end=(0.2,),
                max_pooling=True, batch_size=32, epochs=3,
                checkpoint_path='weights.{epoch:02d}-{val_loss:.2f}.hdf5'):

    inputs_1D = keras.Input(shape=(50,4), name='sequence')
    x = inputs_1D
    for i, j in zip(num_filters_1D, kernel_sizes_1D):
        x = layers.Conv1D(filters=i, kernel_size=j, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    for i, j in zip(number_nodes_1D, dropout_1D):
        x = layers.Dense(i, activation='relu')(x)
        x = layers.Dropout(j)(x)
    x_1D = x


    inputs_2D = keras.Input(shape=(50,50,1), name='structure')
    x = inputs_2D
    for i, j in zip(num_filters_2D, kernel_sizes_2D):
        x = layers.Conv2D(filters=i, kernel_size=j, padding='same', activation='relu')(x)
        if max_pooling:
            x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    for i, j in zip(number_nodes_2D, dropout_2D):
        x = layers.Dense(i, activation='relu')(x)
        x = layers.Dropout(j)(x)
    x_2D = x

    x = layers.concatenate([x_1D, x_2D])
    for i, j in zip(number_nodes_end, dropout_end):
        x = layers.Dense(i, activation='relu')(x)
        x = layers.Dropout(j)(x)
    outputs = layers.Dense(1)(x)
    
    adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model = keras.Model(inputs=[inputs_1D, inputs_2D], outputs=outputs, name="MRL_predict")
    model.compile(loss='mean_squared_error', optimizer=adam)

    checkpoint_filepath = checkpoint_path
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    print(model.summary())

    history = model.fit({'sequence': data[0], 'structure': data[1]}, y,
                        batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=({'sequence': valid[0],
                                          'structure': valid[1]},
                                         valid[2]),
                        callbacks=[model_checkpoint_callback])
    return(model, history)


def one_hot_encode(df, col='utr', seq_len=50):
    # Dictionary returning one-hot encoding of nucleotides. 
    nuc_d = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0]}
    
    # Create empty matrix.
    vectors=np.empty([len(df),seq_len,4])
    
    # Iterate through UTRs and one-hot encode
    for i,seq in enumerate(df[col].str[:seq_len]): 
        seq = seq.lower()
        a = np.array([nuc_d[x] for x in seq])
        vectors[i] = a
    return vectors


def structure_2D(seqs):
    '''
    Creates a matrix of sequence x sequence with a 1 for any nucleotide pair that could
    potentially base pair and a 0 for any pair that could not base pair. Pairs must be AT,
    GC, or GU wobble (GT here because sequences contain T). The pair must be separated by
    at least 3 nucleotides to be considered a potential base pair interaction.
    '''
    result = []
    l = len(seqs.iloc[0])
    diag_zeros = np.ones((l, l))
    for i in range(-3, 4):
        diag_zeros *= np.diag(-1 * np.ones(l - abs(i)), i) + 1
        bp = (('A', 'T'), ('T', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'T'), ('T', 'G'))
    for i, seq in enumerate(seqs):
        bp_potential = [int(pair in bp) for pair in product(seq, repeat=2)]
        bp_2D = np.reshape(bp_potential, (l, l)) * diag_zeros
        result.append(bp_2D)
    return np.asarray(result)

if args.no_uAUG:
    train = pd.read_csv(f'{args.data_path}/train_no_uAUG.csv')
    valid = pd.read_csv(f'{args.data_path}/valid_no_uAUG.csv')
    train_one_hot = np.load(f'{args.data_path}/onehot_train_no_uAUG.npy')
    valid_one_hot = np.load(f'{args.data_path}/onehot_valid_no_uAUG.npy')
    train_structure = np.load(f'{args.data_path}/structure_train_no_uAUG.npy')
    valid_structure = np.load(f'{args.data_path}/structure_valid_no_uAUG.npy')

else:
    train = pd.read_csv(f'{args.data_path}/train.csv')
    valid = pd.read_csv(f'{args.data_path}/valid.csv')
    train_one_hot = np.load(f'{args.data_path}/onehot_train.npy')
    valid_one_hot = np.load(f'{args.data_path}/onehot_valid.npy')
    train_structure = np.load(f'{args.data_path}/structure_train.npy')
    valid_structure = np.load(f'{args.data_path}/structure_valid.npy')

train_structure = np.expand_dims(train_structure, axis=3)
valid_structure = np.expand_dims(valid_structure, axis=3)

model, history = train_model((train_one_hot, train_structure), np.asarray(train['scaled_rl']),
                             valid=(valid_one_hot, valid_structure, np.asarray(valid['scaled_rl'])),
                             num_filters_1D=(128,128,128), num_filters_2D=(32,64,128),
                             kernel_sizes_1D=(8,8,8), kernel_sizes_2D=((3,3),(3,3),(3,3)),
                             number_nodes_1D=(64,), number_nodes_2D=(128,), number_nodes_end=(128,),
                             dropout_1D=(0.2,), dropout_2D=(0.2,), dropout_end=(0.2,),
                             max_pooling=True, batch_size=32, epochs=args.epochs, checkpoint_path=args.checkpoint)

with open(args.history, 'w') as f:
  f.write(str(history.history['loss']))
  f.write('\n')
  f.write(str(history.history['val_loss']))