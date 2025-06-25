import numpy as np
import torch


def grokking_data(p: int, op: str = '/', train_fraction: float = 0.5):
    operations = {
        '*': lambda a, b: (a * b) % p,
        '/': lambda a, b: (a * pow(int(b), p-2, p)) % p,
        '+': lambda a, b: (a + b) % p,
        '-': lambda a, b: (a - b) % p
    }

    if op not in operations:
        raise ValueError(
            "Unsupported operation, choose from ['*', '/', '+', '-']")

    # generate all pairs more efficiently
    if op == '/':
        a_vals = np.repeat(np.arange(p), p - 1)
        b_vals = np.tile(np.arange(1, p), p)
    else:
        a_vals = np.repeat(np.arange(p), p)
        b_vals = np.tile(np.arange(p), p)
    
    X = np.column_stack((a_vals, b_vals))
    
    # vectorized operations where possible
    if op == '+':
        T = (a_vals + b_vals) % p
    elif op == '-':
        T = (a_vals - b_vals) % p
    elif op == '*':
        T = (a_vals * b_vals) % p
    else:  # division
        T = np.array([operations[op](a, b) for a, b in X])

    # create embeddings more efficiently
    embed_op = p
    embed_eq = p + 1
    X = np.column_stack((X[:, 0], np.full(len(X), embed_op), X[:, 1], np.full(len(X), embed_eq)))

    n_train = int(train_fraction * len(X))
    inds = np.random.permutation(len(X))
    Xtrain, Ttrain = X[inds[:n_train]], T[inds[:n_train]]
    Xtest, Ttest = X[inds[n_train:]], T[inds[n_train:]]

    return torch.tensor(Xtrain, dtype=torch.long), \
           torch.tensor(Ttrain, dtype=torch.long), \
           torch.tensor(Xtest, dtype=torch.long), \
           torch.tensor(Ttest, dtype=torch.long)


if __name__ == '__main__':
    Xtrain, Ttrain, Xtest, Ttest = grokking_data(
        11, op='/', train_fraction=0.5)
    print(Xtrain.shape, Ttrain.shape, Xtest.shape, Ttest.shape)
    print(Xtrain[0], Ttrain[0])
