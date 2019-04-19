import argparse
import os

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Video Caption Project")
        #subparsers = self.parser.add_subparsers(title="subcommands", dest="subcommand")

        self.parser.add_argument('--train_folder', type=str,default='/HDD/dl_proj/msr_vtt/TrainValVideo')
        self.parser.add_argument('--val_folder', type=str, default='/HDD/dl_proj/msr_vtt/TrainValVideo')
        self.parser.add_argument('--test_folder', type=str, default='/HDD/dl_proj/msr_vtt/TestVideo')
        self.parser.add_argument('--glove_file', type=str,default='/HDD/dl_proj/glove/glove.6B.50d.txt')
        self.parser.add_argument('--train_dict', type=str,default='./annotation/train_50d.json')
        self.parser.add_argument('--test_dict', type=str,default='./annotation/test_50d.json')
        self.parser.add_argument('--val_dict.json', type=str, default='./annotation/val_50d.json')
        self.parser.add_argument('--frames', type=int, default=50)
        self.parser.add_argument('--caption_length', type=int, default=50)
        self.parser.add_argument('--cuda', type=int, default=0)
        self.parser.add_argument('--batch_size', type=int, default=16)
        self.parser.add_argument('--lr', type=float, default=1e-3, help='set the learning rate')
        self.parser.add_argument('--epochs', type=int, defualt=30, help='set the training epochs')

        #self.parser.add_argumen()

    def parse(self):
        return self.parser.parse_args()
    

# test = Options().parse()
# # print(test.add_argument('--batch_size',32))
# # print(test.parse_args().batch_size)
# print(test.batch_size)