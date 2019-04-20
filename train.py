import time 
import mxnet as mx 
from mxnet.gluon.model_zoo import vision
from mxnet import gluon, autograd, nd
from data_loader import videoFolder
import utils
from option import Options, args_
from network import lstm_net
from metrics import L2Loss_2, L2Loss_cos, cal_bleu_batch, cal_meteor_batch

def train(args):
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

    train_dataset = videoFolder(args.train_folder,args.train_dict, frames, glove_file, 
                    caption_length, ctx, transform=transform, target_transform=target_transform)

    #test_dataset = videoFolder(args.test_folder,args.test_dict, frames, glove_file, caption_length, transform=transform)

    train_loader = gluon.data.DataLoader(train_dataset,batch_size=args.batch_size,
                                         last_batch='keep',shuffle=True)

    #test_loader = gluon.data.DataLoader(test_dataset,batch_size=args.batch_size,last_batch='keep',shuffle=False)

    #loss = L2Loss_cos()
    loss = L2Loss_2()
    net = lstm_net(frames,caption_length,ctx,pretrained=args.load_pretrain)
    
    net.initialize(init=mx.initializer.MSRAPrelu(), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': args.lr})
    
    epoch_loss = nd.array([])
    
    for e in range(args.epochs):
        for batch_id, (x,_) in enumerate(train_loader):
            #print(x.shape)
            with autograd.record():
               pred = net(x)
               batch_loss = loss(pred,_)
               batch_loss.backward()

            #print(train_loss.shape)
            trainer.step(args.batch_size)
            mx.nd.waitall()
            
            epoch_loss = (batch_loss if ((batch_id == 0) and (e == 0))
                          else epoch_loss + batch_loss)
        
        print("Epoch {}, train_loss:{}".format(e+1, epoch_loss.shape))

def main():
    args = args_()
    train(args)

if __name__ == "__main__":
    main()  