import torch
import numpy as np
import matplotlib.pyplot as plt
import os
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0,0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0,0.02)
        m.bias.data.fill_(0)

def plotout(result,inputsampled,target,savepath,epochid):
    def denormalize(indata):
        indata = np.transpose(indata,[1,2,0])
        indata = (indata * 128 + 128).astype(np.uint8)
        return indata
    inputsampled = inputsampled[3,...]
    result_im = denormalize(result)
    target = denormalize(target)
    inputsampled = denormalize(inputsampled)

    fig, axs = plt.subplots(1,3)
    fig.set_size_inches(31.5, 10.5)
    axs[0].imshow(inputsampled)
    axs[1].imshow(target)
    axs[2].imshow(result_im)
    fig.tight_layout()
    plt.savefig(os.path.join(savepath,str(epochid)+'result.png'))
    plt.close()
