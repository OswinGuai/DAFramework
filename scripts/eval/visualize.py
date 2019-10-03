import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

a = torch.rand(64,1024)
TSNE(random_state=20181116).fit_transform(a)
tt=TSNE(random_state=20181116).fit_transform(a)

f = plt.figure(figsize=(8,4))

c = ['r','b']
plt.scatter(x=tt[:,0], y=tt[:,1], label='A')

plt.show()
f.savefig('good.pdf')


def visualize_scatter(data_2d, label_ids, figsize=(20,20)):
    plt.figure(figsize=figsize)
    plt.grid()
    nb_classes = len(np.unique(label_ids))

    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
        data_2d[np.where(label_ids == label_id), 1],
        marker='o',
        color= plt.cm.Set1(label_id / float(nb_classes)),
        linewidth='1',
        alpha=0.8,
        label=id_to_label_dict[label_id])
        plt.legend(loc='best')

