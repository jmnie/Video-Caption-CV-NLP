import cv2
import mxnet.gluon.data as data
import json
import os
import numpy as np

WIDTH = 224
HEIGHT = 224

def make_dataset(rootdir,dict_file):
    with open(dict_file,'r') as fp:
        video_dict = json.load(fp)
    videos_ = []
    for key in video_dict:
        file = key + '.mp4'
        path = os.path.join(rootdir,file)
        caption = video_dict[key]['caption']
        item = (path,caption)
        videos_.append(item)
    return videos_

def load_glove_model(gloveFile):
    f = open(gloveFile,'r')
    model = {}
    dimension = 0
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
        dimension = len(embedding)
    return model,dimension


def word2embd(words,glove_model,caption_length,dimension):
    embd = np.zeros((caption_length,dimension))
    for i in range(caption_length):
        if i < len(words):
            embd[i] = glove_model[words[i]]
    return embd

def opencv_loader(path,frame_count,img_size=None):

    videocap = cv2.VideoCapture(path)
    
    total_frames = videocap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(total_frames/frame_count)
    
    frames = []
    for _ in range(frame_count):
        _f = _ * fps
        videocap.set(cv2.CAP_PROP_FRAME_COUNT,_f)
        hasFrames,frame = videocap.read()
        if hasFrames:
            '''
            Resize the frame
            '''
            if img_size is not None:
                frame = cv2.resize(frame, (img_size, img_size)) 
            frames.append(frame)
    return np.array(frames)


class videoFolder(data.Dataset):
    def __init__(self, rootdir, dict_file, frames, gloveFile, caption_length, img_size=240, transform=None,loader=opencv_loader, word2embd=word2embd):

        videos = make_dataset(rootdir,dict_file)

        if len(videos) == 0:
            raise(RuntimeError("Found 0 videos in directory of: " + rootdir + "\n"))
        
        self.videos = videos

        self.loader = loader
        self.word2embd = word2embd

        self.rootdir = rootdir
        self.dict_file = dict_file
        self.transform = transform
        self.img_size = img_size
        self.frames = frames
        glove_model,dimension = load_glove_model(gloveFile)
        self.glove_model = glove_model
        self.dimension = dimension
        self.caption_length = caption_length
       
        #return super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, caption = self.videos[idx]
        video_frame = self.loader(path,self.frames,self.img_size)
        embd = self.word2embd(caption,self.glove_model,self.caption_length,self.dimension)

        """
        Preprocess the frame
        """
        if self.transform is not None:
            video_frame = self.transform(video_frame)
        
        return video_frame,embd
    
    def __len__(self):
        return len(self.videos)

if __name__ == '__main__':
    video_path = '/home/jiaming/Downloads/dataset/msr-vtt/TestVideo/video8486.mp4'
    #video_path = '/HDD/dl_proj/msr_vtt/TestVideo/video7382.mp4'
    frame_count = 50
    size = 224
    frames = opencv_loader(video_path,frame_count,size)
    print(len(frames),frames.shape)
    #from PIL import Image
    #im = Image.fromarray(frames[0])
    #im.show()
  

