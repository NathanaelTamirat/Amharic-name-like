import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words=pd.read_csv("data/amharic_names.csv")

###### Method-1 ######
# count method(counting)

am_words=words["in_am"].to_list()

b={}
for w in am_words:
    chs=["<S>"]+list(w)+["<E>"]
    for ch1,ch2 in zip(chs,chs[1:]):
        bigram=(ch1,ch2) # for counting we use bigram method
        b[bigram]=b.get(bigram,0)+1

word_sort=sorted(b.items(), key=lambda kv: -kv[1]) # decending

N=torch.zeros((188,188),dtype=torch.int32)
chars=sorted(list(set("".join(am_words))))  # set wont allow duplication
stoi={s:i+1 for i,s in enumerate(chars)}  #string to index
stoi["-"]=1
stoi["."]=0

for word in am_words:
    chs=["."]+list(word)+["."]
    for ch1,ch2 in zip(chs,chs[1:]):
        idx1=stoi[ch1]
        idx2=stoi[ch2]
        N[idx1,idx2]+=1

itos= {i:s for s,i in stoi.items()} #index to string

#this matrix has normalized pr of the N matix
P=(N+1).float() # we add 1 to the whole element in the vector to smooth(called model smoothing)   
P/=P.sum(1,keepdim=True)

g=torch.Generator().manual_seed(2147483647)   # generated the names in the index=0
for i in range(20):     
    out=[]
    ix=0
    while True:
        p=P[ix]
        ix=torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
        out.append(itos[ix])
        if ix==0:  # end token
            break
    print("".join(out)) # bigram language model is very terrible in the accuracy and learning

# to claculate the loss of the bigram model
log_likelihood=0
n=0
#finding the loss by counting
for word in am_words:
# for word in ["ናትናኤልዐድ"]:  # we dont get negative infinity 
    chs=["."]+list(word)+["."]
    for ch1,ch2 in zip(chs,chs[1:]):
        idx1=stoi[ch1]
        idx2=stoi[ch2]
        prob=P[idx1,idx2]
        logprob=torch.log(prob)
        log_likelihood+=logprob
        n+=1
        # print(f"{ch1} {ch2}: {prob: .4f} {logprob: .4f}")
print(f"{log_likelihood=}")
nll=-log_likelihood
print(f"{nll=}")
print(f"{nll/n}")


###### Method-2 ######
# simple Neural network with one layer

# create the training set of the bigrams
xs,ys=[],[]
for word in am_words:
    chs=["."]+list(word)+["."]
    for ch1,ch2 in zip(chs,chs[1:]):
        idx1=stoi[ch1]
        idx2=stoi[ch2]
        xs.append(idx1)
        ys.append(idx2)
        
xs=torch.tensor(xs)
ys=torch.tensor(ys)
num=xs.nelement()

g=torch.Generator().manual_seed(21447483647)
W=torch.randn((188,188),generator=g,requires_grad=True)

for i in range(1000):
    xenc=F.one_hot(xs,num_classes=188).float()
    logits=(xenc @ W) # predict the log count
    # softmax
    counts=logits.exp() # Counts (== N)
    prob=counts/counts.sum(1,keepdims=True) # prob for the next character
    loss=-prob[torch.arange(num),ys].log().mean()
    loss+=0.01*(W**2).mean()  #reularized loss(adding the numbers === adding more smoothing numbers)
    print(loss.item())
    #backward pass
    W.grad=None #same as zero
    loss.backward()
    W.data+=-50*W.grad  # update

# finally we generate some samples to see the result
g=torch.Generator().manual_seed(2147483647)   # generated the 20 names in the index =0
for i in range(200):     
    out=[]
    ix=4    
    while True:
        xenc=F.one_hot(torch.tensor([ix]),num_classes=188).float()
        logits= xenc @ W # predict the log count
        # softmax
        counts=logits.exp() # Counts (== N)
        prob=counts/counts.sum(1,keepdims=True) # prob 
        ix=torch.multinomial(prob,num_samples=1,replacement=True,generator=g).item()
        out.append(itos[ix])
        if ix==0:  # end token
            break
    print("".join(out))