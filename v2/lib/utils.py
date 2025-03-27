import matplotlib.pyplot as plt 
import numpy as np
import torch 
# helper function to show an image

def img_show(img):
    img = img / 2.0 + 0.5     # unnormalize
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    
def transform_show(img1, img2):
    fig = plt.figure(figsize=(16, 34))
    ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
    img_show(img1)
    ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
    img_show(img2)
    return fig



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
