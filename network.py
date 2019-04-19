from mxnet import gluon, nd, ndarray
from mxnet.gluon import nn,rnn
from mxnet.gluon.model_zoo import vision
from gluoncv import model_zoo
import mxnet as mx
import mxnet.ndarray as F 

from mxnet.gluon.loss import Loss

def _apply_weighting(F, loss, weight=None, sample_weight=None):

    if sample_weight is not None:
        loss = F.broadcast_mul(loss, sample_weight)
    if weight is not None:
        assert isinstance(weight, numeric_types), "weight must be a number"
        loss = loss * weight
    return loss

def _reshape_like(F, x, y):
    return x.reshape(y.shape) if F is ndarray else F.reshape_like(x, y)


class L2Loss(Loss):
    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(L2Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = _reshape_like(F, label, pred)
        loss = F.square(pred - label)
        loss = _apply_weighting(F, loss, self._weight/2, sample_weight)
        return F.sqrt(F.mean(loss, axis=self._batch_axis, exclude=True))

class L2Loss_2(Loss):

    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None, batch_axis=1, **kwargs):
        super(L2Loss_2, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits
        self.weight = weight

    def hybrid_forward(self, F, pred, label, sample_weight=None):

        loss = F.sum(F.sqrt(F.square(pred - label)), axis=self._batch_axis)
        #return F.mean(loss, axis=self._batch_axis, exclude=True)
        return loss

class L2Loss_cos(Loss):
    def __init__(self, ctx, weight=None, batch_axis=0, margin=0, **kwargs):
        super(L2Loss_cos, self).__init__(weight, batch_axis, **kwargs)
        self._margin = margin
        self.ctx = ctx

    
    def hybrid_forward(self, F, pred, label,sample_weight=None):

        cos_sim_matrix = mx.ndarray.zeros((pred.shape[0],pred.shape[1]),ctx=ctx)

        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                cos_sim_matrix[i][j] = cosine_sim(F,pred[i][j],label[i][j])
        
        loss = F.abs(1-cos_sim_matrix)
        return loss

    def cosine_sim(self,F,input_1,input_2):
        cosine_sim = F.dot(input_1,input_2)/F.norm(input_1)/F.norm(input_2)
        return cosine_sim[0]

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

def _cosine_similarity(pred, label, ctx, F=mx.ndarray):
    cos_sim_matrix = mx.ndarray.zeros((pred.shape[0],pred.shape[1]),ctx=ctx)

    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            cos_sim_matrix[i][j] = cosine_sim(F,pred[i][j],label[i][j])

    #cos_sim_matrix = mx.nd.array(cos_sim_matrix)
    loss = F.abs(1-cos_sim_matrix)
    return loss

def cosine_sim(F,input_1,input_2):
    cosine_sim = F.dot(input_1,input_2)/F.norm(input_1)/F.norm(input_2)
    return cosine_sim[0]


if __name__ == '__main__':
    #load_pretrain()
    #test_net()
    ctx = mx.cpu()
    pred = nd.random.uniform(shape=(16, 50, 50),ctx=ctx)
    label = nd.random.uniform(shape=(16, 50, 50),ctx=ctx)
    result = F.sum(F.sqrt(F.square(pred - label)),axis=1)
    #print(pred[0].shape)
    #result = F.dot(pred[0],pred[0])/F.norm(pred[0])/F.norm(pred[0])
    #result = _cosine_similarity(pred,pred,ctx=ctx)
    print(result.shape)
