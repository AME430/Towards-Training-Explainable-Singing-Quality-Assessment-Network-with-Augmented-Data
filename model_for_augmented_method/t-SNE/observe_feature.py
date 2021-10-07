import numpy as np
from sklearn.manifold import TSNE
import pylab

X = np.loadtxt(r'D:\tsne\txtfiles\tsne_CQT_ph\feature_test.txt')
scores = np.loadtxt(r'D:\tsne\txtfiles\tsne_CQT_ph\target_test.txt')

# get labels
labels = []
for i in range(len(scores)):
    if scores[i] >= 0.25:
        labels.append(1)
    else:
        labels.append(2)
labels = np.array(labels)

# reduce dimension of feature vector using tsne
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
Y = tsne.fit_transform(X)

# plot
pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
pylab.title('Hybrid CRNN model: CPH-CRNN')
pylab.savefig(r'D:\tsne\figures\CPH-CRNN-test.png')
pylab.show()