from matplotlib.pyplot import scatter as sc
from sklearn.manifold import TSNE
import torch
from matplotlib import pyplot as plt
import numpy as np
import random
import sys

def visualize_scatter(plt, data_2d, label_ids, figsize=(20,20)):
    plt.figure(figsize=figsize)
    plt.grid()
    nb_classes = len(np.unique(label_ids))

    for label_id in np.unique(label_ids):
        #if label_id % 3 != 0:
        #    continue
        plt.scatter(
                data_2d[np.where(label_ids == label_id), 0],
                data_2d[np.where(label_ids == label_id), 1],
                marker='o',
                color= plt.cm.Set1(label_id / float(nb_classes)),
                linewidth='1',
                alpha=0.8,
                label=label_id)
        plt.legend(loc='best')

#a = torch.rand(64,1024)
#TSNE(random_state=20181116).fit_transform(a)
#tt=TSNE(random_state=20181116).fit_transform(a)
features = np.load(sys.argv[1])
labels = np.load(sys.argv[2])

visualize_scatter(plt, features, labels)
#plt.show()
plt.savefig('%s.pdf' % sys.argv[3])



