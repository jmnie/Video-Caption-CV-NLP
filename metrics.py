from mxnet.gluon.loss import Loss
import mxnet.ndarray as F 
from mxnet import gluon, nd, ndarray
import mxnet as mx

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

    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None, batch_axis=2, **kwargs):
        super(L2Loss_2, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits
        self.weight = weight

    def hybrid_forward(self, F, pred, label, sample_weight=None):

        loss = F.mean(F.sqrt(F.square(pred - label)), axis=self._batch_axis)
        #return F.mean(loss, axis=self._batch_axis, exclude=True)
        return loss

class L2Loss_cos(Loss):
    def __init__(self, weight=None, batch_axis=0, margin=0, eps=1e-12, **kwargs):
        super(L2Loss_cos, self).__init__(weight, batch_axis, **kwargs)
        self._margin = margin

    
    def hybrid_forward(self, F, pred, label,sample_weight=None):
        loss = F.sqrt(F.square(F.flatten(pred)-F.flatten(label)))
        loss = loss.reshape(loss.shape[0],pred.shape[1],pred.shape[2])
        return F.mean(loss,axis=1)

'''
Add the metrics here:
METEOR, CIDEr, BLEU, ROUGE_L
'''
import nltk
import nltk.translate.bleu_score as bleu
import nltk.translate.meteor_score as meteor

def get_bleu(label,pred):
    score = bleu.sentence_bleu([label], pred)
    return score

def get_meteor(label,pred):
    label = ' '.join(word for word in label)
    pred = ' '.join(word for word in pred)
    score = meteor.single_meteor_score(label,pred)
    return score

def cal_bleu_batch(label_batch,pred_batch):
    score = 0
    for i in range(len(label_batch)):
        score += get_bleu(label_batch[i],pred_batch[i])
    
    return float(score/len(label_batch)) 

def cal_meteor_batch(label_batch,pred_batch):
    score = 0
    for i in range(len(label_batch)):
        score += get_bleu(label_batch[i],pred_batch[i])
    
    return float(score/len(label_batch)) 

def embed_to_word(embd,model):
    bestWord = None
    distance = float('inf')
    for word in model.keys():
        e=model[word]
        d = 0
        for a,b in zip(e,embd):
            d+=(a-b)*(a-b)
        if d<distance:
            distance=d
            bestWord = word

    assert(bestWord is not None)
    return (bestWord, distance)

def embed_to_sentence(embd,model):
    setence=[]
    for i in range(len(embd)):
        word = embed_to_word(embd[i],model)[0]
        sentence.append(word)
    return sentence
        
def load_glove_model(gloveFile):
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    #print("Done.",len(model)," words loaded!")
    return model
    

def embed_to_sen_batch(batch_vector,gloveModel):
    batch_vector = batch_vector.asnumpy()
    senten_batch = []
    
    for i in range(batch_vector.shape[0]):
        temp = embed_to_sentence(batch_vetcor[i],gloveModel)
        senten_batch.append(temp)
        
    return senten_batch
    
if __name__ == '__main__':
    hyp = str('she read the book because she was interested in world history').split()
    ref_a = str('she read the book because she was interested in world history').split()
    ref_b = str('she was interested in world history because she read the book').split()
    #print(ref_a,ref_b)
    ctx = mx.cpu()
    x = nd.random.uniform(shape=(16,50),ctx=ctx)
    mean = F.mean(x).asscalar()
    print(mean)
    
    #print(get_meteor(ref_b,hyp))