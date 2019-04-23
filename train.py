#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import numpy as np
import os
import sys
#import multiprocessing


# In[2]:


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
                                last_batch='keep',shuffle=True)

    test_loader = gluon.data.DataLoader(test_dataset,batch_size=args.batch_size,
                                    last_batch='keep',shuffle=False)

    loss = L2Loss_2()
    net = lstm_net(frames,caption_length,ctx,pretrained=args.load_pretrain)
    #net = resnet18_v2(caption_length=caption_length,ctx=ctx)
                            
            
    net.collect_params().initialize(init=mx.initializer.MSRAPrelu(), ctx=ctx)
        
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': args.lr})
    
    train_loss = []
    test_loss = []
    train_loss_batch = []
    test_loss_batch = []
    
    smoothing_constant = 0.01
    
    for e in range(args.epochs):
        
        epoch_loss = 0.
        for batch_id, (x,_) in enumerate(train_loader):
      
            with autograd.record():
                pred = net(x)
                batch_loss = loss(pred,_)
            
            trainer.step(x.shape[0],ignore_stale_grad=True)
            batch_loss.backward()
            mx.nd.waitall()
            
            batch_loss = np.mean(batch_loss.asnumpy())

            if ((batch_id == 0) and (e == 0)):
                epoch_loss = batch_loss
            else:
                epoch_loss = (1 - smoothing_constant)*epoch_loss + smoothing_constant*batch_loss
            
            train_loss_batch.append(batch_loss)
            
            if (batch_id+1) % 200 == 0:
                print("Train Batch:{}, batch_loss:{}".format(batch_id+1, batch_loss))
                
            if ((e+1)*(batch_id + 1)) % (2*args.log_interval) == 0:
                # save model
                save_model_filename = "Epoch_" + str(e) + "_iters_" + str(batch_id + 1) + '_'  + str(time.ctime()).replace(' ', '_') + "_" + ".params"
                
                save_model_path = os.path.join(args.model_path, save_model_filename)
                net.save_parameters(save_model_path)
                print("\nCheckpoint, trained model saved at", save_model_path)
                
                train_loss_filename = "Epoch_" + str(e) + "_iters_" + str(batch_id + 1) + str(time.ctime()).replace(' ', '_') + "_train_loss" + ".txt"
                
                train_loss_path = os.path.join(args.log_path, train_loss_filename)
                np.savetxt(train_loss_path,np.array(train_loss_batch))
        
                
        epoch_loss_1 = 0.
        for batch_id, (x,_) in enumerate(test_loader):
                            
            with autograd.predict_mode():
                predict = net(x)
                batch_loss_1 = loss(pred,_)
                
            batch_loss_1 = np.mean(batch_loss_1.asnumpy())
            #if (batch_id+1) % 30 == 0:
            #    print("Test Batch:{}, batch_loss:{}".format(batch_id+1, batch_loss_1))
                
            if ((batch_id == 0) and (e == 0)):
                epoch_loss_1 = batch_loss_1
            else:
                epoch_loss_1 = (1 - smoothing_constant)*epoch_loss_1 + smoothing_constant*batch_loss_1
            
            test_loss_batch.append(batch_loss_1)
            
            if ((e+1)*(batch_id + 1)) % (args.log_interval) == 0:
                
                test_loss_file_name = "Epoch_" + str(e) + "_iters_" + str(batch_id + 1) + str(time.ctime()).replace(' ', '_') + "_test_loss" + ".txt"
                test_loss_path = os.path.join(args.log_path, test_loss_file_name)
                np.savetxt(test_loss_path,np.array(test_loss_batch))
                
        
        train_loss.append(epoch_loss)
        test_loss.append(epoch_loss_1)
        
        print("Epoch {}, train_loss:{}, test_loss:{}".format(e+1, epoch_loss, epoch_loss_1))
     
    # save model
    save_model_filename = "Final_epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + ".params"
    save_model_path = os.path.join(args.model_path, save_model_filename)
    net.save_parameters(save_model_path)
    print("\nDone, trained model saved at", save_model_path)
    
    train_epoch_loss_file_name = 'train_epoch_loss.txt'
    test_epoch_loss_file_name = 'test_epoch_loss.txt'
    train_epoch_loss_path = os.path.join(args.log_path,train_epoch_loss_file_name)
    test_epoch_loss_path = os.path.join(args.log_path,test_epoch_loss_file_name)
    np.savetxt(train_epoch_loss_path,train_loss)
    np.savetxt(test_epoch_loss_path,test_loss)


# In[3]:


def main():
    args = args_()
#     args.set_data_path(
#         '/home/jiaming/Downloads/dataset/msr-vtt/TrainValVideo',
#         '/home/jiaming/Downloads/dataset/msr-vtt/TestVideo',
#         '/home/jiaming/Downloads/dataset/msr-vtt/TrainValVideo',
#     )
#     args.set_glove_file(
#         '/home/jiaming/Downloads/dataset/glove.6B/glove.6B.50d.txt'
#     )
    train(args)


# In[4]:


if __name__ == "__main__":
    main()  


# In[ ]:





# In[ ]:




