from ctypes import *
import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision.models as models
from sklearn import preprocessing
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import images
def get_images(path):
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(path)
    tensors = []
    images = []
    for entry in dataset:
        img = np.array(entry[0])
        tensors.append(T(img))
        images.append(img)
    return np.array(tensors), np.array(images)

# Extract feature representation from model
def get_features(vgg, tensor, vgg_depth):
    output = vgg.features[:vgg_depth](tensor.view(1, 3, tensor.shape[1], tensor.shape[2]))[0]
    return output.permute(1, 2, 0).detach().to(torch.device('cpu')).numpy().astype(np.float64)

# Normalize n-by-d data such that each point (row) has unit length
def normalize(X):
    h, w, depth = X.shape
    mat = preprocessing.normalize(X.reshape((-1, depth)), axis=1).reshape((h, w, depth))
    return mat

# Use FSLIC to obtain a superpixel segmentation of a (feature) image
def get_superpixel_image(X, num_superpixels=50, compactness=0.05, max_iterations=10, f=0.0, g=2.0):
    h, w, d = X.shape
    shared_file = CDLL("./shared.so")
    shared_file.fslic.restype = POINTER(c_double * (h * w * d))
    l = py_object(X.astype(c_double).flatten().tolist())
    v = shared_file.fslic(l, c_int(w), c_int(h), c_int(d), c_int(num_superpixels), c_double(compactness), c_int(max_iterations), c_double(f), c_double(g))
    result = np.asarray([x for x in v.contents]).reshape((h, w, d))
    unique_superpixels, labels = np.unique(result.reshape((-1, d)), axis=0, return_inverse=True)
    labels = labels.reshape((h, w))
    return unique_superpixels, labels

# Nearest neighbor upsampling
def upsample(X, new_y, new_x):
    old_y, old_x = X.shape[:2]
    result = np.zeros((new_y, new_x), dtype=type(X[0][0]))
    for i in range(new_y):
        for j in range(new_x):
            result[i][j] = X[int(i/new_y*old_y)][int(j/new_x*old_x)]
    return result

# t-SNE wrapper
def get_tsne_result(X, n_components=2, perplexity=30):
    T_SNE = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=200)
    print('Starting t-SNE ...')
    tsne_result = T_SNE.fit_transform(X)
    print('Finished t-SNE')
    return tsne_result

# Create figure where each point in X_2d_data is overlayed with its corresponding patch
def overlay_tsne_patches(X_2d_data, patches, figsize=(45,45), image_zoom=1.0, title='', sparsity=1):
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for idx, (xy, i) in enumerate(zip(X_2d_data, patches)):
        if idx % sparsity == 0:
            x0, y0 = xy
            img = OffsetImage(i, zoom=image_zoom)
            ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    ax.axis('off')
    return fig