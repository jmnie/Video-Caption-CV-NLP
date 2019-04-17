from mxnet import gluon, nd
from mxnet.gluon.model_zoo import vision
from gluoncv import model_zoo
import mxnet as mx

def load_pretrain():
    model = model_zoo.get_mobilenet_v2(0.25)
    print(model)
    model.initialize()
    ctx = mx.cpu()
    X = nd.random.uniform(shape=(32, 50, 3, 128, 128),ctx=ctx)
    X = nd.transpose(X,(0,1,2,3,4))
    print(X.shape)
    output = model(X)
    print(output.shape)

class stack_lstm(gluon.Block)
    def __init__


if __name__ == '__main__':
    load_pretrain()