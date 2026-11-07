"""Bayesian filtering of machine-specific temporal response and changing regimes."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from adaptive_model import learn_relevance, prior_for, positive_distribution

TYPES = ('steady','balanced','jackpot')


def fit_population(items, conditions):
    relevance = learn_relevance(items)
    primary = sorted((np.argsort(-relevance)[:3] + 1).tolist())
    selected = [i for i in items if conditions[i['query'].query_id] == 'stable' and len(i['y']) > 30]
    xx, yy, lag, group, typ, sig = [], [], [], [], [], []
    for k,item in enumerate(selected):
        x = item['x'][:,primary]
        y = (item['y'] > 0).astype(float)
        cov = np.mean((x-x.mean(0))*(y-y.mean())[:,None],axis=0)
        j = np.argmax(abs(cov))
        sign = 1 if cov[j] >= 0 else -1
        xx.extend(sign*x[:,j]);yy.extend(y);lag.extend(item['x'][:,7])
        group.extend([k]*len(y));typ.extend([TYPES.index(item['query'].machine_type)]*len(y));sig.extend([item['query'].signal]*len(y))
    x,y,z,g,t,s = map(np.array,(xx,yy,lag,group,typ,sig))
    g=g.astype(int);t=t.astype(int);n=len(selected)
    start=np.zeros(6+2*n);start[:6]=[.2,-.6,-1.6,.35,.25,2.5]
    def objective(beta):
        bias=beta[6:6+n]; amp=beta[6+n:]
        eta=beta[t]+beta[3]*s+beta[4]*z+bias[g]+(beta[5]+amp[g])*x
        residual=expit(eta)-y
        value=np.sum(np.logaddexp(0,eta)-y*eta)+.5*np.sum(bias*bias)/.6**2+.5*np.sum(amp*amp)/.7**2
        gradient=np.r_[np.bincount(t,weights=residual,minlength=3),residual@s,residual@z,residual@x,
                       np.bincount(g,weights=residual,minlength=n)+bias/.6**2,
                       np.bincount(g,weights=residual*x,minlength=n)+amp/.7**2]
        return value,gradient
    fit=minimize(objective,start,jac=True,method='L-BFGS-B',bounds=[(None,None)]*5+[(.5,6)]+[(None,None)]*(2*n),options={'maxiter':300,'ftol':1e-10})
    return dict(primary=primary,bias=fit.x[:3].tolist(),signal=float(fit.x[3]),secondary=float(fit.x[4]),
                amplitude=float(fit.x[5]),bias_sd=.6,amplitude_sd=.7,training_success=bool(fit.success))


def pack(items, population):
    n=len(items);maximum=max(i['query'].cutoff for i in items)
    y=np.full((maximum+1,n),-1,dtype=np.int8)
    x=np.zeros((maximum+1,n,len(population['primary'])))
    z=np.zeros((maximum+1,n))
    cut=np.array([i['query'].cutoff for i in items])
    first=cut.copy()
    bias=np.array([population['bias'][TYPES.index(i['query'].machine_type)]+population['signal']*i['query'].signal for i in items])
    episodes=np.array([i['query'].episode_id for i in items])
    _,episode_index=np.unique(episodes,return_inverse=True)
    for k,item in enumerate(items):
        ticks=item['query'].cutoff-1-item['age']
        y[ticks,k]=(item['y']>0)
        x[ticks,k]=item['x'][:,population['primary']]
        z[ticks,k]=item['x'][:,7]
        if len(ticks):first[k]=ticks[0]
        x[cut[k],k]=item['xnext'][population['primary']]
        z[cut[k],k]=item['xnext'][7]
    return dict(y=y,x=x,z=z,first=first,cut=cut,bias=bias,episode_index=episode_index)


def diffuse(state, strength, axis):
    if strength == 0:return state
    a=np.roll(state,1,axis=axis);b=np.roll(state,-1,axis=axis)
    index=[slice(None)]*state.ndim;index[axis]=0;a[tuple(index)]=state[tuple(index)]
    index[axis]=-1;b[tuple(index)]=state[tuple(index)]
    return (1-strength)*state+(strength/2)*(a+b)


def filter_win(packed,population,*,amplitude_noise=.15,bias_noise=.03,flip_rate=.003,shared_flip=0.,
               amplitude_scale=1.,bias_shift=0.,secondary_scale=1.,prior_width=1.,step=.5):
    n=len(packed['cut']);j=len(population['primary'])
    amps=np.arange(-6.,6.+step/2,step)
    offsets=np.arange(-2.,2.001,.4)
    amp=population['amplitude']*amplitude_scale
    ap=np.exp(-.5*((amps-amp)/(population['amplitude_sd']*prior_width))**2)+np.exp(-.5*((amps+amp)/(population['amplitude_sd']*prior_width))**2)
    bp=np.exp(-.5*(offsets/(population['bias_sd']*prior_width))**2)
    prior=bp[:,None]*ap[None,:];prior/=prior.sum()
    state=np.broadcast_to(prior,(n,j,len(offsets),len(amps))).copy()/j
    output=np.empty(n)
    base=packed['bias'][:,None,None,None]+bias_shift+offsets[None,None,:,None]
    qa=min(.9,(amplitude_noise/step)**2);qb=min(.9,(bias_noise/.4)**2)
    for tick in range(int(min(packed['first'].min(),packed['cut'].min())),int(packed['cut'].max())+1):
        active=(tick>packed['first'])&(tick<=packed['cut'])
        if active.any():
            transitioned=diffuse(diffuse(state[active],qa,3),qb,2)
            if flip_rate:transitioned=(1-flip_rate)*transitioned+flip_rate*transitioned[:,:,:,::-1]
            state[active]=transitioned
        eta=base+packed['x'][tick,:,:,None,None]*amps[None,None,None,:]+population['secondary']*secondary_scale*packed['z'][tick,:,None,None,None]
        probability=expit(eta)
        now=tick==packed['cut']
        if now.any():
            predictive=state[now]
            if shared_flip:predictive=(1-shared_flip)*predictive+shared_flip*predictive[:,:,:,::-1]
            output[now]=(predictive*probability[now]).sum(axis=(1,2,3))
        observed=(packed['y'][tick]>=0)&(tick<packed['cut'])
        if not observed.any():continue
        likelihood=np.where(packed['y'][tick,:,None,None,None]==1,probability,1-probability)
        likelihood[~observed]=1.
        updated=state*likelihood
        norm=updated.sum(axis=(1,2,3))
        updated/=norm[:,None,None,None]
        if shared_flip:
            alternate=state[:,:,:,::-1]*likelihood
            alt_norm=alternate.sum(axis=(1,2,3))
            alternate/=alt_norm[:,None,None,None]
            evidence=np.bincount(packed['episode_index'],weights=np.log(alt_norm)-np.log(norm))
            gate=expit(np.log(shared_flip/(1-shared_flip))+evidence)[packed['episode_index']]
            updated=(1-gate[:,None,None,None])*updated+gate[:,None,None,None]*alternate
        update_mask=((tick>=packed['first'])&(tick<packed['cut'])) if shared_flip else observed
        state[update_mask]=updated[update_mask]
    return output


def amount_priors(items):
    counts={t:np.ones(3)*.5 for t in TYPES}
    for item in items:
        counts[item['query'].machine_type]+=np.bincount(item['all_y'],minlength=5)[2:]
    return {t:(c/c.sum()).tolist() for t,c in counts.items()}


def factorized_amount(items,priors,ratios,half_life=64,strength=30):
    result=[]
    for item in items:
        old=positive_distribution(item,prior_for(item,priors),half_life,strength)
        result.append(np.r_[old[0],(1-old[0])*np.array(ratios[item['query'].machine_type])])
    return np.array(result)
