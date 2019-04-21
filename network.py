from mxnet import gluon, nd, ndarray
from mxnet.gluon import nn,rnn
from mxnet.gluon.model_zoo import vision
from gluoncv import model_zoo
import mxnet as mx
import mxnet.ndarray as F 
import numpy as np

def load_pretrain():
    model = vision.vgg16_bn(pretrained=True)

    new_net = model.features[:-1]
    #print(type(new_net))

    ctx = mx.cpu()
    X = nd.random.uniform(shape=(100, 50, 3, 224, 224),ctx=ctx)
    one_output = new_net(X[0])
    print(one_output.shape)

    print(X.shape)
    result = nd.zeros(shape=(X.shape[0],X.shape[1],one_output.shape[-1]))
    
    for i in range(X.shape[0]):
        result[i] = new_net(X[i])

    print(result.shape)

def test_net():
    model = mx.gluon.nn.Sequential()
    with model.name_scope():
        #model.add(mx.gluon.nn.Embedding(7, 10))
        model.add(mx.gluon.rnn.LSTM(20))
        model.add(mx.gluon.nn.Dense(50, flatten=False))
    model.initialize()

    net = lstm_net(50,50)
    net.initialize()
    output = net(mx.nd.ones((6, 3, 224, 224)))
    print(output.shape)


class lstm_net(gluon.Block):
    def __init__(self,frames,caption_length,ctx,pretrained=False):
        super(lstm_net,self).__init__()

        self.frames = frames
        self.caption_length = caption_length
        self.pretrained = pretrained
        self.ctx = ctx
        
        self.lstm_1 = rnn.LSTM(hidden_size=100,num_layers=1,layout='NTC',bidirectional=False)
        self.lstm_2 = rnn.LSTM(hidden_size=100,num_layers=1,layout='NTC',bidirectional=False)
        self.dense = nn.Dense(self.caption_length*self.caption_length,flatten=True)

    def forward(self, x):
        if not self.pretrained:
            input_ = x.reshape(x.shape[0],x.shape[1],x.shape[2]*x.shape[3]*x.shape[4])
        else:
            input_ = x
            
        output_1 = self.lstm_1(input_)
        output_2 = self.lstm_2(output_1)
        dense_1 = self.dense(output_2)
        output = F.reshape(dense_1,(dense_1.shape[0],self.caption_length,self.caption_length))
        return output


"""Change directly from the ResNet code"""

def _conv3x3(channels, stride, in_channels):
    return nn.Conv3D(channels, kernel_size=3, strides=stride, padding=1,use_bias=False, in_channels=in_channels)

# Blocks
class BasicBlockV1(gluon.HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BasicBlockV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels, 1, channels))
        self.body.add(nn.BatchNorm())
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv3D(channels, kernel_size=1, strides=stride,use_bias=False, in_channels=in_channels))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(residual+x, act_type='relu')

        return x

class BottleneckV1(gluon.HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BottleneckV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv3D(channels//4, kernel_size=1, strides=stride))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels//4, 1, channels//4))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv3D(channels, kernel_size=1, strides=1))
        self.body.add(nn.BatchNorm())
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv3D(channels, kernel_size=1, strides=stride,use_bias=False, in_channels=in_channels))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(x + residual, act_type='relu')
        return x

class BasicBlockV2(gluon.HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BasicBlockV2, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.conv1 = _conv3x3(channels, stride, in_channels)
        self.bn2 = nn.BatchNorm()
        self.conv2 = _conv3x3(channels, 1, channels)
        if downsample:
            self.downsample = nn.Conv3D(channels, 1, stride, use_bias=False, in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        return x + residual

class BottleneckV2(gluon.HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BottleneckV2, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.conv1 = nn.Conv3D(channels//4, kernel_size=1, strides=1, use_bias=False)
        self.bn2 = nn.BatchNorm()
        self.conv2 = _conv3x3(channels//4, stride, channels//4)
        self.bn3 = nn.BatchNorm()
        self.conv3 = nn.Conv3D(channels, kernel_size=1, strides=1, use_bias=False)
        if downsample:
            self.downsample = nn.Conv3D(channels, 1, stride, use_bias=False, in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        x = self.bn3(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv3(x)

        return x + residual

# Nets
class ResNetV1(gluon.HybridBlock):
    def __init__(self, block, layers, channels, classes=1000, thumbnail=False,caption_length=50,**kwargs):
        super(ResNetV1, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.caption_length=caption_length
            self.features = nn.HybridSequential(prefix='')
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0))
            else:
                self.features.add(nn.Conv3D(channels[0], 7, 2, 3, use_bias=False))
                self.features.add(nn.BatchNorm())
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool3D(3, 2, 1))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],stride, i+1, in_channels=channels[i]))
            self.features.add(nn.GlobalAvgPool3D())
            #self.features.add(nn.Dense(classes, in_units=in_channels))
            self.features.add(nn.Dense(caption_length*caption_length,in_units=in_channels))


    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        #x = self.output(x)
        return x

class ResNetV2(gluon.HybridBlock):
    def __init__(self, block, layers, channels, classes=1000, thumbnail=False, caption_length=50,**kwargs):
        super(ResNetV2, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.caption_length=caption_length
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.BatchNorm(scale=False, center=False))
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0))
            else:
                self.features.add(nn.Conv3D(channels[0], 7, 2, 3, use_bias=False))
                self.features.add(nn.BatchNorm())
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool3D(3, 2, 1))

            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],stride, i+1, in_channels=in_channels))
                in_channels = channels[i+1]
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.GlobalAvgPool3D())
            self.features.add(nn.Flatten())
            #self.features.add(nn.Dense(classes, in_units=in_channels))
            self.features.add(nn.Dense(caption_length*caption_length,in_units=in_channels))
            

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        #x = self.output(x)
        x = F.reshape(x,(x.shape[0],self.caption_length,self.caption_length))
        return x

resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

resnet_net_versions = [ResNetV1, ResNetV2]
resnet_block_versions = [{'basic_block': BasicBlockV1, 'bottle_neck': BottleneckV1},
                         {'basic_block': BasicBlockV2, 'bottle_neck': BottleneckV2}]

def get_resnet(version, num_layers, pretrained=False, ctx=mx.cpu(), caption_length=50, **kwargs):

    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s"%(
            num_layers, str(resnet_spec.keys()))
    block_type, layers, channels = resnet_spec[num_layers]
    assert version >= 1 and version <= 2, \
        "Invalid resnet version: %d. Options are 1 and 2."%version

    resnet_class = resnet_net_versions[version-1]
    block_class = resnet_block_versions[version-1][block_type]
    net = resnet_class(block_class, layers, channels, caption_length=caption_length,**kwargs)
    return net

def resnet18_v2(caption_length=50,ctx=mx.cpu(),**kwargs):
    return get_resnet(2, 18, caption_length=caption_length, ctx=ctx,**kwargs)

def resnet34_v2(caption_length=50,**kwargs):
    return get_resnet(2, 34, caption_length=50, **kwargs)

def resnet50_v2(caption_length=50, **kwargs):
    return get_resnet(2, 50, caption_length=50, **kwargs)

def resnet101_v2(caption_length=50, **kwargs):
    return get_resnet(2, 101, caption_length=50, **kwargs)

def resnet152_v2(caption_length=50, **kwargs):
    return get_resnet(2, 152, caption_length=50, **kwargs)


if __name__ == '__main__':
    ctx = mx.cpu()
    net = lstm_net(40,50,ctx=ctx)
    #net = resnet18_v2(50)
    net.initialize(ctx=ctx)
    #print(net.output)
    X = nd.random.uniform(shape=(233,50,3,224,224),ctx=ctx)
    output = net(X)
    print(output.shape)

    #load_pretrain()
