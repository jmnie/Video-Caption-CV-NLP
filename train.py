import time 
import mxnet as mx 
from mxnet import gluon, autograd
from data_loader import videoFolder
import utils
from option import Options
from multiprocessing import cpu_count
from network import lstm_net, L2Loss_2, L2Loss_cos

def train(args):
    frames = args.frames
    caption_length = args.caption_length
    glove_file = args.caption_length

    CPU_COUNT = cpu_count()

    if args.cuda:
        ctx = mx.gpu()
    else:
        ctx = mx.cpu()
    
    transform = utils.Compose([utils.ToTensor(ctx),
                               utils.normalize(ctx),
                               ])

    train_dataset = videoFolder(args.train_folder,args.train_dict,glove_file,transform=transform)

    test_dataset = videoFolder(args.test_folder,args.test_dict,glove_file,transform=transform)

    train_loader = gluon.data.DataLoader(train_dataset,batch_size=args.batch_size,last_batch='keep',shuffle=True, num_workers=CPU_COUNT)

    test_loader = gluon.data.DataLoader(train_dataset,batch_size=args.batch_size,last_batch='keep',shuffle=False,num_workers=CPU_COUNT)

    loss = L2Loss_cos()
    net = lstm_net(frames,caption_length)
    net.initialize(init=mx.initializer.MSRAPrelu(), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': args.lr})

    for e in range(args.epochs):
        for batch_id, (x,_) in enumerate(train_loader):
            with autograd.record():
                pred = net(x)
                train_loss = loss(pred,_)
                train_loss.backward()
            
            trainer.step(args.batch_size)
            mx.nd.waitall()
        
        print("Epoch {}, train_loss: {:.4f}".format(e+1, train_loss))




def main():
    args = Options().parse()
    train(args)


if __name__ == "__main__":
    main()  