import sys, os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.environ.get('ENTRY',os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..')))
from common import *
from houseedge.features import build_split
import torch, torch.nn as nn, time, argparse
ap=argparse.ArgumentParser(); ap.add_argument('--hid',type=int,default=64); ap.add_argument('--ep',type=int,default=30)
ap.add_argument('--lr',type=float,default=3e-3); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--layers',type=int,default=1)
ap.add_argument('--drop',type=float,default=0.1); ap.add_argument('--tag',default='g'); ap.add_argument('--trainsplits',default='train,validation,test')
a=ap.parse_args()
ap2=a; torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(int(os.environ.get('THREADS','4')))
S={}
for s_ in ['train','validation','test']:
    X_,Y_,c_,q_=build_split(D/s_,extended=False); S[s_]=(X_.astype(np.float32),Y_.astype(np.int64),c_,q_)
def flat(s): return S[s]
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
    if e>=a.ep-2:
        (pts,rat),pred=evalv(); print(e, round(tl/len(Xt),4), round(pts,3), rat, round(time.time()-t0,1), flush=True)
    else: print(e, round(tl/len(Xt),4), round(time.time()-t0,1), flush=True)
OUTD=os.environ.get('OUTD','.'); pred.to_csv(f'{OUTD}/pv_{a.tag}.csv',index=False)
np.savez(f'{OUTD}/w_{a.tag}.npz',**{k:v.detach().numpy() for k,v in net.state_dict().items()})
