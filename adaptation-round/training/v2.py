import sys, os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.environ.get('ENTRY',os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')))
from common import load, score, PC, D as PUB
import os
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, time, argparse
from houseedge.features import load_split, PIDX, TYPES, T
ap=argparse.ArgumentParser()
ap.add_argument('--hid',type=int,default=128); ap.add_argument('--ep',type=int,default=28); ap.add_argument('--lr',type=float,default=3e-3)
ap.add_argument('--drop',type=float,default=0.1); ap.add_argument('--seed',type=int,default=1); ap.add_argument('--aug',type=float,default=0.0)
ap.add_argument('--mix',type=int,default=0); ap.add_argument('--threads',type=int,default=4); ap.add_argument('--tag',default='v2')
ap.add_argument('--bs',type=int,default=8); ap.add_argument('--wd',type=float,default=1e-4); ap.add_argument('--evalfrom',type=int,default=15)
a=ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads)
def raw(split):
    qs,hist=load_split(PUB/split)
    eps={}
    for q in qs: eps.setdefault(q['episode_id'],[]).append(q)
    C=[];TY=[];SG=[];CUT=[];Q=[]
    for ep in sorted(eps):
        e=sorted(eps[ep],key=lambda q:q['machine_id']); cut=int(e[0]['cutoff']); idx={q['machine_id']:i for i,q in enumerate(e)}
        c=np.full((len(e),T),5,dtype=np.int64)
        for m,t,p in hist.get(ep,[]):
            if t<cut: c[idx[m],t]=PIDX[p]
        C.append(c); TY.append([TYPES[q['machine_type']] for q in e]); SG.append([int(float(q['signal'])) for q in e]); CUT.append(cut); Q.append([q['query_id'] for q in e])
    return dict(C=torch.tensor(np.stack(C)),ty=torch.tensor(TY),sg=torch.tensor(SG,dtype=torch.float32),cut=torch.tensor(CUT),q=np.array(Q))
D={s:raw(s) for s in ['train','validation','test']}
ALL={k:(torch.cat([D[s][k] for s in D]) if k!='q' else None) for k in ['C','ty','sg','cut']}
SHORT=(ALL['C']<5).sum(-1).float().mean(-1)<100   # episodes already short
def feats(C,ty,sg):
    B,M,TT=C.shape
    oh=F.one_hot(C,6).float(); out=[]
    for k in range(1,9):
        sh=torch.zeros_like(oh); sh[...,5]=1; sh[:,:,k:]=oh[:,:,:-k]; out.append(sh)
    win=((C>0)&(C<5)).float(); obs=(C<5).float()
    for k in range(1,9):
        wk=torch.zeros_like(win); ok=torch.zeros_like(obs); wk[:,:,k:]=win[:,:,:-k]; ok[:,:,k:]=obs[:,:,:-k]
        sw=wk.sum(1,keepdim=True)-wk; so=ok.sum(1,keepdim=True)-ok
        out.append(torch.stack([sw/so.clamp(min=1), so/(M-1)],-1))
    st=torch.zeros(B,M,TT,5); st[...,:3]=F.one_hot(ty,3).float()[:,:,None,:]; st[...,3]=sg[:,:,None]; st[...,4]=torch.arange(TT).float()/256
    out.append(st); return torch.cat(out,-1)
def augment(C,cut,short):
    C=C.clone(); B,M,TT=C.shape; Y=C.clone(); Y[Y==5]=-1
    for b in range(B):
        if short[b] or np.random.rand()>=a.aug: continue
        c=int(np.random.randint(72,int(cut[b])+1))
        keep=torch.zeros(M,TT,dtype=torch.bool); keep[:,c-40:c]=True
        for m in range(M): keep[m,torch.randperm(c-40)[:24]]=True
        newC=torch.full((M,TT),5,dtype=torch.long); newC[keep]=C[b][keep]
        y=torch.full((M,TT),-1,dtype=torch.long); y[keep]=Y[b][keep]; y[:,c]=Y[b][:,c]
        C[b]=newC; Y[b]=y
    return C,Y
class Net(nn.Module):
    def __init__(s,Din,H):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(Din,H),nn.GELU(approximate='tanh'))
        s.g1=nn.GRU(H,H,batch_first=True); s.g2=nn.GRU(2*H if a.mix else H,H,batch_first=True); s.dp=nn.Dropout(a.drop)
        s.head=nn.Sequential(nn.Dropout(a.drop),nn.Linear(2*H,H),nn.GELU(approximate='tanh'),nn.Linear(H,5))
    def forward(s,x):
        B,M,TT,Din=x.shape; z=s.inp(x).reshape(B*M,TT,-1)
        h1,_=s.g1(z); h1=s.dp(h1)
        if a.mix:
            h=h1.reshape(B,M,TT,-1); mix=(h.sum(1,keepdim=True)-h)/(M-1); h1=torch.cat([h,mix],-1).reshape(B*M,TT,-1)
        h2,_=s.g2(h1)
        return s.head(torch.cat([h2,z],-1)).reshape(B,M,TT,5)
net=Net(69,a.hid); nE=len(ALL['C']); steps=a.ep*int(np.ceil(nE/a.bs))
opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=a.wd); sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr,total_steps=steps)
lossf=nn.CrossEntropyLoss(ignore_index=-1); va=load('validation'); V=D['validation']
Xv=feats(V['C'],V['ty'],V['sg'])
def evalv():
    net.eval()
    with torch.no_grad(): L=net(Xv)
    E,M=V['q'].shape; P=torch.softmax(L,-1)[torch.arange(E)[:,None],torch.arange(M)[None,:],V['cut'][:,None]].reshape(-1,5).numpy()
    pred=pd.DataFrame(P,columns=PC); pred['query_id']=V['q'].reshape(-1); net.train(); return score(pred,va),pred
for e in range(a.ep):
    perm=torch.randperm(nE); tl=0;t0=time.time()
    for i in range(0,nE,a.bs):
        b=perm[i:i+a.bs]; C,Y=augment(ALL['C'][b],ALL['cut'][b],SHORT[b])
        L=net(feats(C,ALL['ty'][b],ALL['sg'][b])); loss=lossf(L.reshape(-1,5),Y.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sched.step(); tl+=loss.item()
    msg=f'{e} {tl/np.ceil(nE/a.bs):.4f} {time.time()-t0:.0f}s'
    if e>=a.evalfrom or e==a.ep-1:
        (pts,rat),pred=evalv(); msg+=f' {pts:.3f} {rat}'
    print(msg,flush=True)
OUTD=os.environ.get('OUTD','.'); pred.to_csv(f'{OUTD}/pv_{a.tag}.csv',index=False)
np.savez(f'{OUTD}/w_{a.tag}.npz',**{k:v.detach().numpy() for k,v in net.state_dict().items()})
