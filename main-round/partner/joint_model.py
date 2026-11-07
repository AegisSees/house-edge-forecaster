"""Joint Bayesian prediction of win probability and positive payout sizes."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logsumexp,logit
from state_model import TYPES,pack,diffuse


def fit_size_relation(items,population,conditions):
    primary=population['primary']
    selected=[i for i in items if conditions[i['query'].query_id]=='stable']
    # Infer the size lag corresponding to each win pattern using train histories.
    relevance=[[] for _ in primary]
    groups=[]
    for item in selected:
        y=(item['y']>0).astype(float);x=item['x'][:,primary]
        cov=((x-x.mean(0))*(y-y.mean())[:,None]).mean(0)
        pattern=int(np.argmax(abs(cov)));m=item['y']>0
        if m.sum()<5:continue
        sx=item['x'][m,37:45];sy=(item['y'][m]==1).astype(float)
        centered=sx-sx.mean(0);yc=sy-sy.mean()
        score=np.mean(centered*yc[:,None],axis=0)**2/(np.mean(centered**2,axis=0)+.1)-np.var(sy)/len(sy)
        relevance[pattern].append(score)
        groups.append((item,pattern))
    size_lags=[int(np.argmax(np.mean(r,axis=0)))+1 for r in relevance]
    data=[]
    sums=np.zeros((3,2))
    for item,pattern in groups:
        m=item['y']>0;typ=TYPES.index(item['query'].machine_type)
        y=(item['y'][m]==1).astype(float);x=item['x'][m,36+size_lags[pattern]]
        data.append((typ,x,y));sums[typ]+=[y.sum(),len(y)]
    initial=np.r_[logit(sums[:,0]/sums[:,1]),.65]
    def objective(theta):
        value=0.;gradient=np.zeros(4)
        for typ,x,y in data:
            signed=x[:,None]*np.array([-1.,1.])[None,:]
            eta=theta[typ]+theta[3]*signed
            ll=np.sum(y[:,None]*eta-np.logaddexp(0,eta),axis=0)
            normal=logsumexp(ll)
            posterior=np.exp(ll-normal)
            value-=normal-np.log(2.)
            residual=expit(eta)-y[:,None]
            gradient[typ]+=np.sum(residual*posterior)
            gradient[3]+=np.sum(residual*signed*posterior)
        return value,gradient
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=[(-5,5)]*3+[(.05,2.)],options={'maxiter':100,'ftol':1e-11})
    return dict(lags=size_lags,bias=result.x[:3].tolist(),amplitude=float(result.x[3]),success=bool(result.success))


def pack_joint(items,population,size_model):
    p=pack(items,population)
    y=np.full_like(p['y'],-1)
    sx=np.zeros_like(p['x'])
    for k,item in enumerate(items):
        ticks=item['query'].cutoff-1-item['age']
        y[ticks,k]=item['y']
        columns=np.array(size_model['lags'])+36
        sx[ticks,k]=item['x'][:,columns]
        sx[item['query'].cutoff,k]=item['xnext'][columns]
    p.update(y=y,sx=sx,typ=np.array([TYPES.index(i['query'].machine_type) for i in items]))
    return p


def filter_joint(p,population,size_model,ratios,*,amplitude_noise=0.,bias_noise=.03,flip_rate=.001,
                 size_flip=0.,coupled_flip=False,amplitude_scale=1.,prior_width=1.,size_scale=1.,step=.5,
                 amplitude_nodes=0,bias_step=.4):
    n=len(p['cut']);j=len(population['primary'])
    amplitudes=np.arange(-6.,6.+step/2,step);offsets=np.arange(-2.,2.+bias_step/2,bias_step)
    center=np.array(population.get('amplitude_by_type',[population['amplitude']]*3))[p['typ']]*amplitude_scale
    asd=np.array(population.get('amplitude_sd_by_type',[population['amplitude_sd']]*3))[p['typ']]*prior_width
    bsd=np.array(population.get('bias_sd_by_type',[population['bias_sd']]*3))[p['typ']]*prior_width
    if amplitude_nodes:
        if amplitude_noise:raise ValueError('Quadrature amplitudes support sign reversals, not diffusion')
        nodes,weights=np.polynomial.hermite.hermgauss(amplitude_nodes)
        positive=center[:,None]+np.sqrt(2)*asd[:,None]*nodes[None,:]
        amplitudes=np.concatenate([-positive[:,::-1],positive],axis=1)
        ap=np.broadcast_to(np.r_[weights[::-1],weights]/(2*np.sqrt(np.pi)),amplitudes.shape).copy()
    else:
        ap=np.exp(-.5*((amplitudes[None,:]-center[:,None])/asd[:,None])**2)+np.exp(-.5*((amplitudes[None,:]+center[:,None])/asd[:,None])**2)
        amplitudes=np.broadcast_to(amplitudes,(n,len(amplitudes)))
    bp=np.exp(-.5*(offsets[None,:]/bsd[:,None])**2)
    prior=bp[:,:,None]*ap[:,None,:];prior/=prior.sum(axis=(1,2),keepdims=True)
    state=np.broadcast_to(prior[:,None,:,:,None],(n,j,len(offsets),amplitudes.shape[1],2)).copy()/(j*2)
    base=p['bias'][:,None,None,None]+offsets[None,None,:,None]
    sb=np.array(size_model['bias'])[p['typ']]
    ratio=np.array([ratios[TYPES[t]] for t in p['typ']])
    output=np.zeros((n,5));qa=min(.9,(amplitude_noise/step)**2);qb=min(.9,(bias_noise/bias_step)**2)
    for tick in range(int(min(p['first'].min(),p['cut'].min())),int(p['cut'].max())+1):
        active=(tick>p['first'])&(tick<=p['cut'])
        if active.any():
            transition=diffuse(diffuse(state[active],qa,3),qb,2)
            reverse=transition[:,:,:,::-1,::-1] if coupled_flip else transition[:,:,:,::-1,:]
            if flip_rate:transition=(1-flip_rate)*transition+flip_rate*reverse
            if size_flip:transition=(1-size_flip)*transition+size_flip*transition[:,:,:,:,::-1]
            state[active]=transition
        eta=base+p['x'][tick,:,:,None,None]*amplitudes[:,None,None,:]+population['secondary']*p['z'][tick,:,None,None,None]
        win=expit(eta)[:,:,:,:,None]
        one=expit(sb[:,None,None]+size_model['amplitude']*size_scale*p['sx'][tick,:,:,None]*np.array([-1.,1.])[None,None,:])[:,:,None,None,:]
        now=p['cut']==tick
        if now.any():
            z=state[now];pw=(z*win[now]).sum(axis=(1,2,3,4));p1=(z*win[now]*one[now]).sum(axis=(1,2,3,4))
            output[now,0]=1-pw;output[now,1]=p1;output[now,2:]=(pw-p1)[:,None]*ratio[now]
        observed=(p['y'][tick]>=0)&(tick<p['cut'])
        if observed.any():
            outcome=p['y'][tick,:,None,None,None,None]
            emission=np.where(outcome==0,1-win,np.where(outcome==1,win*one,win*(1-one)))
            update=state[observed]*emission[observed]
            state[observed]=update/update.sum(axis=(1,2,3,4),keepdims=True)
    return output
