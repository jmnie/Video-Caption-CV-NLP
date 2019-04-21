import time 
import mxnet as mx 
from mxnet import gluon,autograd,nd
import mxnet.ndarray as F
from mxnet.gluon.model_zoo import vision
from data_loader import videoFolder
import utils
from option import Options, args_
from multiprocessing import cpu_count
from network import lstm_net,resnet18_v2
from metrics import L2Loss_2, L2Loss_cos
import sys

def train(args):
    frames = args.frames
    caption_length = args.caption_length
    glove_file = args.glove_file
    
    #CPU_COUNT = multiprocessing.cpu_count()
    if args.cuda:
        ctx = mx.gpu()
    else:
        ctx = mx.cpu()
    
    if args.load_pretrain:
        pretrain_model = vision.vgg16_bn(pretrained=True,ctx=ctx)
        transform = utils.Compose([utils.ToTensor(ctx),
                               utils.normalize(ctx),
                               utils.extractFeature(ctx,pretrain_model)
                             ])
    else:
        pretrain_model = None
        transform = utils.Compose([utils.ToTensor(ctx),
                                   utils.normalize(ctx),
                                 ])
    
    target_transform = utils.targetCompose([utils.WordToTensor(ctx)])

    train_dataset = videoFolder(args.train_folder,args.train_dict, frames, glove_file, 
                    caption_length, ctx, transform=transform, target_transform=target_transform)

    test_dataset = videoFolder(args.test_folder,args.test_dict, frames, glove_file, 
                        caption_length, ctx, transform=transform, target_transform=target_transform)

    train_loader = gluon.data.DataLoader(train_dataset,batch_size=args.batch_size,
                                last_batch='discard',shuffle=True)

    test_loader = gluon.data.DataLoader(test_dataset,batch_size=args.batch_size,
                                    last_batch='discard',shuffle=False)

    #loss = L2Loss_cos()
    loss = L2Loss_2()
    net = lstm_net(frames,caption_length,ctx,pretrained=args.load_pretrain)
    #net = resnet18_v2(caption_length=caption_length,ctx=ctx)
    
    net.collect_params().initialize(init=mx.initializer.MSRAPrelu(), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': args.lr})
    
    smoothing_constant = 0.01
    
    for e in range(args.epochs):
        epoch_loss = 0
        
        for batch_id, (x,_) in enumerate(train_loader):
            with autograd.record():
                pred = net(x)
                batch_loss = loss(pred,_)
            
            trainer.step(x.shape[0],ignore_stale_grad=True)
            batch_loss.backward()
            mx.nd.waitall()
            
            batch_loss = F.mean(batch_loss).asscalar()
            
            if batch_id % 100 == 0:
                print("Train Batch:{}, batch_loss:{}".format(batch_id+1, batch_loss))
                  
            epoch_loss = (batch_loss if ((batch_id == 0) and (e == 0))
                          else (1 - smoothing_constant)*epoch_loss + smoothing_constant*batch_loss)
        
        epoch_loss_1 = 0
        for batch_id, (x,_) in enumerate(test_loader):
            with autograd.predict_mode():
                predict = net(x)
                batch_loss_1 = loss(pred,_)
            
            batch_loss_1 = F.mean(batch_loss_1).asscalar()
            
            if batch_id % 100 == 0:
                print("Test Batch:{}, batch_loss:{}".format(batch_id+1, batch_loss_1))
                
            epoch_loss_1 = (batch_loss_1 if ((batch_id == 0) and (e == 0))
                          else (1 - smoothing_constant)*epoch_loss_1 + smoothing_constant*batch_loss_1)
            
 
        
        print("Epoch {}, train_loss:{}, test_loss:{}".format(e+1, epoch_loss, epoch_loss_1))
    
    if args.save_model == True:
        file_name = "./saved_model/" + "lstm_pretrain.params"
        net.save_parameters(file_name)

def eval(args):
    frames = args.frames
    caption_length = args.caption_length
    glove_file = args.glove_file

    if args.cuda:
        ctx = mx.gpu()
    else:
        ctx = mx.cpu()
    
    if args.load_pretrain:
        pretrain_model = vision.vgg16_bn(pretrained=True,ctx=ctx)
        transform = utils.Compose([utils.ToTensor(ctx),
                               utils.normalize(ctx),
                               utils.extractFeature(ctx,pretrain_model)
                             ])
    else:
        pretrain_model = None
        transform = utils.Compose([utils.ToTensor(ctx),
                                   utils.normalize(ctx),
                                 ])
    
    target_transform = utils.targetCompose([utils.WordToTensor(ctx)])

    val_dataset = videoFolder(args.val_folder,args.val_dict, frames, glove_file, caption_length, ctx, transform=transform, target_transform=target_transform)

    val_loader = gluon.data.DataLoader(val_dataset, batch_size=args.batch_size,last_batch='discard',shuffle=True)

    
    
def main():
    args = args_()
    train(args)

if __name__ == "__main__":
    main()  