from utils import *
import sys

# Initialize variables
device = torch.device("cuda:0")
vgg = models.vgg16(pretrained=True, progress=True).to(device)
vgg_depth = 29

# Load images from data set
tensors, images = get_images(sys.argv[1])

# Obtain feature representations of images
features = []
for tensor in tqdm(tensors, "Passing images through model"):
    result = get_features(vgg, tensor.to(device), vgg_depth)
    result = normalize(result)
    features.append(result)

# Generate superpixel segmentation of feature images
unique_ = []
labels = []
lengths = []
for i, f in tqdm(enumerate(features), desc="Generating superpixel images"):
    unique, label = get_superpixel_image(f)
    unique_.extend(unique)
    shape = tensors[i].permute(1, 2, 0).numpy().shape[:2]
    label = upsample(label, shape[0], shape[1])
    labels.append(label)
    lengths.append(len(unique))
unique_ = np.asarray(unique_)
print('Total Number of Superpixels:', len(unique_))

# Embed feature superpixel centers using t-SNE
tsne_result = get_tsne_result(unique_, n_components=2, perplexity=30)

# Obtain bounding boxes of areas covered by feature superpixels with respect to the original images
patches = []
annotated_imgs = []
for i, l in tqdm(enumerate(lengths), desc='Generating Patches'):
    img = images[i]
    h, w = img.shape[:2]
    for j in range(l):
        ys, xs = np.where(labels[i] == j)
        left = max(np.min(xs) - 1, 0)
        right = min(np.max(xs) + 2, w)
        top = max(np.min(ys) - 1, 0)
        bottom = min(np.max(ys) + 2, h)
        patches.append(img[top:bottom, left:right])

fig = overlay_tsne_patches(tsne_result, patches, figsize=(80,60), image_zoom=0.8)
fig.savefig('tsne_thumbnails.pdf', format='pdf', bbox_inches='tight', transparent=True)
fig.savefig('tsne_thumbnails.png', format='png', bbox_inches='tight', transparent=True)
plt.close(fig)
print('Fin')
