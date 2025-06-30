
words = open('C:/Users/VedPrakash/Desktop/ml models/makemore/names.txt', 'r').read().splitlines()
print(words[:10])  # Print the first 10 words to verify

len(words)  # Check the total number of words
max(len(w) for w in words)
min(len(w) for w in words)

# print(words[:1])
print(words[1:5])

B = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        # print(ch1, ch2)
        bigram = (ch1, ch2)
        B[bigram] = B.get(bigram, 0) + 1

sorted(B.items(), key=lambda x: x[1])

import matplotlib
import torch

# arr = torch.zeros((3, 3), dtype=torch.int32)
# print(arr)

# create a bigram matrix
arr = torch.zeros((27, 27), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
len(chars)
chars

stoi = {ch:i+1 for i, ch in enumerate(chars)}
stoi['.'] = 0

itos = {i:ch for ch, i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]
        arr[idx1, idx2] += 1


import matplotlib.pyplot as plt

plt.figure(figsize=(16,16))
plt.imshow(arr, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, arr[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');

# Normalizing the bigram matrix
arr[0]

p = arr[0].float()
# doing normalization
p /= p.sum()
p.sum()

# doing sampling
g = torch.Generator().manual_seed(2147483647)
idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
itos[idx]

g = torch.Generator().manual_seed(2147483647)
x = torch.rand(3, generator=g)
x /= x.sum()
x

g = torch.Generator().manual_seed(2147483647)

torch.multinomial(x, num_samples=20, replacement=True, generator=g)

ix = 0
out = []

for i in range(50):

    while True:
        p = arr[ix].float()
        p /= p.sum()
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

# really bad bigram model need to do better

P = (arr+1).float()
p = P.sum(dim=1, keepdim=True)
P /= p
P[1].sum()

P.shape
p.shape

g = torch.Generator().manual_seed(2147483647)
l = []
for i in range(50):
    ix = 0
    out=[]
    while True:
        p = P[ix]
        # p = arr[ix].float()
        # p /= p.sum()
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    l.append(''.join(out))

len(l)
print(l[:50])

#now we need to know how good or bad this model is
max_likelihood = 0.0
cnt = 0

for w in ['jqyoti']:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]
        prob = P[idx1, idx2]
        log_prob = torch.log(prob)
        max_likelihood += log_prob
        cnt += 1
        print(f'{ch1}{ch2}: {prob: .4f} {log_prob: .4f}')

print(f'Maximum likelihood: {max_likelihood: .4f}')

nll = -max_likelihood
print(f'Negative log likelihood --> loss: {nll: .4f}')

loss = nll / cnt
print(f'Average loss per character: {loss}')

# now we have to create a neural network model to learn the bigram probabilities
#creating a training set for our model

xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]
        # print(f'{ch1}{ch2}')
        xs.append(idx1)
        ys.append(idx2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

num = len(xs)

print(f'Number of training examples: {num}')

xs
ys

# we have to encode the input as one-hot vectors
import torch.nn.functional as F


# plt.imshow(xenc)
# plt.show()


g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)  # weight matrix for bigram model
xenc = F.one_hot(xs, num_classes=27).float()

(W**2).mean().shape

for i in range(105):
    # we are doing forward pass
    # (5, 27) * (27, 27) --> (5, 27)
    logits = xenc @ W  # log-counts (matrix multiplication)
    counts = logits.exp()  # equivalent to arr
    probs = counts / counts.sum(dim=1, keepdim=True)  # predict probabilities
    # calculating the loss
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()  # regularization term
    print(f'Loss:{i + 1} {loss.item():.8f}')
    #doing backward pass
    W.grad = None
    loss.backward()
    # a magic happened here

    W.data += -50 * W.grad  # updating the weights


