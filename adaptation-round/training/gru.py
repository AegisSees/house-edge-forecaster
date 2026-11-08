import sys, os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
import torch, torch.nn as nn, time, argparse
ap=argparse.ArgumentParser(); ap.add_argument('--hid',type=int,default=64); ap.add_argument('--ep',type=int,default=30)
ap.add_argument('--lr',type=float,default=3e-3); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--layers',type=int,default=1)
ap.add_argument('--drop',type=float,default=0.1); ap.add_argument('--tag',default='g'); ap.add_argument('--trainsplits',default='train,validation,test')
a=ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(4)
S={s:np.load(f'seq_{s}.npz') for s in ['train','validation','test']}
def flat(s): z=S[s]; E,M,T,D=z['X'].shape; return z['X'].reshape(E*M,T,D), z['Y'].reshape(E*M,T), np.repeat(z['c'],M), z['q'].reshape(-1)
Xtr=np.concatenate([flat(s)[0] for s in a.trainsplits.split(',')]); Ytr=np.concatenate([flat(s)[1] for s in a.trainsplits.split(',')])
Xv,Yv,cv,qv=flat('validation')
va=load('validation')
class Net(nn.Module):
    def __init__(s,D,H):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D,H),nn.GELU(approximate='tanh'))
        s.rnn=nn.GRU(H,H,num_layers=a.layers,batch_first=True,dropout=a.drop if a.layers>1 else 0)
        s.head=nn.Sequential(nn.Dropout(a.drop),nn.Linear(2*H,H),nn.GELU(approximate='tanh'),nn.Linear(H,5))
    def forward(s,x):
        z=s.inp(x); o,_=s.rnn(z)
        # o_t has seen inputs up to t, input at t only uses ticks < t: safe
        return s.head(torch.cat([o,z],-1))
net=Net(Xtr.shape[-1],a.hid); opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-4)
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr,total_steps=a.ep*int(np.ceil(len(Xtr)/64)))
Xt=torch.tensor(Xtr); Yt=torch.tensor(Ytr); lossf=nn.CrossEntropyLoss(ignore_index=-1)
def evalv():
    net.eval()
    with torch.no_grad(): L=net(torch.tensor(Xv))
    P=torch.softmax(L,-1).numpy()[np.arange(len(Xv)),cv]
    pred=pd.DataFrame(P,columns=PC); pred['query_id']=qv; net.train(); return score(pred,va), pred
best=(0,None)
for e in range(a.ep):
    perm=torch.randperm(len(Xt)); tl=0; t0=time.time()
    for i in range(0,len(Xt),64):
        b=perm[i:i+64]; L=net(Xt[b]); loss=lossf(L.reshape(-1,5),Yt[b].reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sched.step(); tl+=loss.item()*len(b)
    (pts,rat),pred=evalv()
    print(e, round(tl/len(Xt),4), round(pts,3), rat, round(time.time()-t0,1), flush=True)
pred.to_csv(f'pred_val_{a.tag}.csv',index=False); torch.save(net.state_dict(),f'net_{a.tag}.pt')
