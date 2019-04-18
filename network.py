from mxnet import gluon, nd
from mxnet.gluon import nn,rnn
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

def test_net():
    model = mx.gluon.nn.Sequential()
    with model.name_scope():
        #model.add(mx.gluon.nn.Embedding(7, 10))
        model.add(mx.gluon.rnn.LSTM(20))
        model.add(mx.gluon.nn.Dense(50, flatten=False))
    model.initialize()

    net = lstm_net(50,50)
    net.initialize()
    output = net(mx.nd.ones((6, 50, 3, 224, 224)))
    print(output.shape)


class lstm_net(gluon.Block):
    def __init__(self,frames,caption_length):
        super(lstm_net,self).__init__()
        self.frames = frames
        self.caption_length = caption_length
        self.lstm_1 = rnn.LSTM(hidden_size=200,num_layers=2,layout='NTC',bidirectional=False)
        self.lstm_2 = rnn.LSTM(hidden_size=100,num_layers=2,layout='NTC',
        bidirectional=False)
        self.dense = nn.Dense(self.caption_length,flatten=False)
    
    def forward(self, x):
        input_ = x.reshape(x.shape[0],x.shape[1],x.shape[2]*x.shape[3]*x.shape[4])
        output_1 = self.lstm_1(input_)
        output_2 = self.lstm_2(output_1)
        dense_1 = self.dense(output_2)
        return dense_1


# class 3d_resnet(gluon.Block):
#     def __init__(self,img_size):
#         super(3d_resnet,self).__init__()
        
#     def forward(self,x):
#         return x




if __name__ == '__main__':
    #load_pretrain()
    test_net()