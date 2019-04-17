import time 
import mxnet as mx 
from mxnet import gluon
from data_loader import videoFolder
import utils

def train():
    print("--")
    root_dir = '/home/jiaming/Downloads/dataset/msr-vtt/TestVideo'
    dict_file = './annotation/test_50d.json'
    frames = 50
    gloveFile = '/home/jiaming/Downloads/dataset/glove.6B/glove.6B.50d.txt'
    caption_length = 50
    ctx = mx.cpu()
    transform = utils.Compose([utils.ToTensor(ctx),
                               utils.normalize(ctx),
                               ])
    test_dataset = videoFolder(root_dir,dict_file,frames,gloveFile,caption_length,transform=transform)

    test_loader = gluon.data.DataLoader(test_dataset, batch_size=32, last_batch='discard')

    for batch_id, (x, label) in enumerate(test_loader):
        print(x.shape,label.shape)
        break


if __name__ == "__main__":
    train()    