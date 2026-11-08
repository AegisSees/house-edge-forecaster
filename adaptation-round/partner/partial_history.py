"""Marginalize unobserved binary spins exactly for short-history queries."""
import numpy as np
from scipy.special import expit,logsumexp
from adaptive_model import PAIRS
from state_model import TYPES


def forecast_partial(items,population,size_model,ratios,*,amplitude_step=.5,bias_step=.4,prior_width=1.,batch_size=24,
                     amplitude_nodes=0,bias_nodes=0):
    out=[]
    for start in range(0,len(items),batch_size):
        out.append(_batch(items[start:start+batch_size],population,size_model,ratios,amplitude_step,bias_step,prior_width,amplitude_nodes,bias_nodes))
    return np.concatenate(out) if out else np.empty((0,5))


def _batch(items,population,size_model,ratios,amplitude_step,bias_step,prior_width,amplitude_nodes,bias_nodes):
    n=len(items);cut=np.array([i['query'].cutoff for i in items]);maximum=int(cut.max())
    observations=np.full((maximum+1,n),-1,dtype=np.int8)
    for k,item in enumerate(items):
        ticks=(item['query'].cutoff-1-item['all_age']).astype(int)
        observations[ticks,k]=item['all_y']
    typ=np.array([TYPES.index(i['query'].machine_type) for i in items])
    bias=np.array(population['bias'])[typ]+population['signal']*np.array([i['query'].signal for i in items])
    amps=np.arange(-6.,6.+amplitude_step/2,amplitude_step)
    offsets=np.arange(-2.,2.+bias_step/2,bias_step)
    center=np.array(population.get('amplitude_by_type',[population['amplitude']]*3))[typ]
    asd=np.array(population.get('amplitude_sd_by_type',[population['amplitude_sd']]*3))[typ]*prior_width
    bsd=np.array(population.get('bias_sd_by_type',[population['bias_sd']]*3))[typ]*prior_width
    if amplitude_nodes:
        nodes,weights=np.polynomial.hermite.hermgauss(amplitude_nodes)
        positive=center[:,None]+np.sqrt(2)*asd[:,None]*nodes[None,:]
        amps=np.concatenate([-positive[:,::-1],positive],axis=1)
        ap=np.broadcast_to(np.r_[weights[::-1],weights]/(2*np.sqrt(np.pi)),amps.shape).copy()
    else:
        ap=np.exp(-.5*((amps[None,:]-center[:,None])/asd[:,None])**2)+np.exp(-.5*((amps[None,:]+center[:,None])/asd[:,None])**2)
        amps=np.broadcast_to(amps,(n,len(amps)))
    if bias_nodes:
        nodes,weights=np.polynomial.hermite.hermgauss(bias_nodes)
        offsets=np.sqrt(2)*bsd[:,None]*nodes[None,:]
        bp=np.broadcast_to(weights/np.sqrt(np.pi),offsets.shape).copy()
    else:
        bp=np.exp(-.5*(offsets[None,:]/bsd[:,None])**2)
        offsets=np.broadcast_to(offsets,(n,len(offsets)))
    theta_prior=(bp[:,:,None]*ap[:,None,:]).reshape(n,-1);theta_prior/=theta_prior.sum(axis=1,keepdims=True)
    tb=np.broadcast_to(offsets[:,:,None],(n,offsets.shape[1],amps.shape[1])).reshape(n,-1)
    ta=np.broadcast_to(amps[:,None,:],(n,offsets.shape[1],amps.shape[1])).reshape(n,-1)
    evidence=[];wins=[];ones=[]
    for j,column in enumerate(population['primary']):
        if column<9:raise ValueError('Exact history mode requires pair features')
        lag_a,lag_b=PAIRS[column-9];lag_a+=1;lag_b+=1
        order=max(lag_a,lag_b);states=2**order;half=states//2
        bit=np.arange(states)
        feature=(2*((bit>>(lag_a-1))&1)-1)*(2*((bit>>(lag_b-1))&1)-1)
        chance=expit(bias[:,None,None]+tb[:,:,None]+ta[:,:,None]*feature[None,None,:])
        # Unknown prehistory is averaged over all binary lag states.
        state=np.broadcast_to(theta_prior[:,:,None]/states,(n,ta.shape[1],states)).copy()
        loglik=np.zeros(n);pw=np.empty(n);final_loglik=np.empty(n)
        for tick in range(maximum+1):
            now=cut==tick
            if now.any():
                pw[now]=(state[now]*chance[now]).sum(axis=(1,2));final_loglik[now]=loglik[now]
            if tick==maximum:break
            observed=(observations[tick]>=0)&(tick<cut)
            one=state*chance
            zero=state-one
            one[(observations[tick]==0)&observed]=0
            zero[(observations[tick]>0)&observed]=0
            nxt=np.empty_like(state)
            nxt[:,:,0::2]=zero[:,:,:half]+zero[:,:,half:]
            nxt[:,:,1::2]=one[:,:,:half]+one[:,:,half:]
            norm=nxt.sum(axis=(1,2))
            state=nxt/norm[:,None,None]
            loglik[observed]+=np.log(norm[observed])
        size_lag=size_model['lags'][j]
        size_log=np.zeros((n,2));size_next=np.zeros((n,2))
        for k,item in enumerate(items):
            rows=observations[:,k]
            ts=np.flatnonzero((rows>0)&(np.arange(len(rows))>=size_lag)&(np.arange(len(rows))<cut[k]))
            ts=ts[rows[ts-size_lag]>=0]
            previous=2.*(rows[ts-size_lag]>1)-1
            one_chance=expit(size_model['bias'][typ[k]]+size_model['amplitude']*previous[:,None]*np.array([-1.,1.])[None,:])
            size_log[k]=np.log(np.where(rows[ts,None]==1,one_chance,1-one_chance)).sum(0)
            previous_now=rows[cut[k]-size_lag]
            if previous_now<0:
                size_next[k]=expit(size_model['bias'][typ[k]])
            else:
                size_next[k]=expit(size_model['bias'][typ[k]]+size_model['amplitude']*(2.*(previous_now>1)-1)*np.array([-1.,1.]))
        size_evidence=logsumexp(size_log,axis=1)-np.log(2.)
        size_posterior=np.exp(size_log-logsumexp(size_log,axis=1)[:,None])
        q=(size_posterior*size_next).sum(1)
        evidence.append(final_loglik+size_evidence);wins.append(pw);ones.append(q)
    evidence=np.array(evidence).T;post=np.exp(evidence-logsumexp(evidence,axis=1)[:,None])
    win=(post*np.array(wins).T).sum(1);one=(post*np.array(wins).T*np.array(ones).T).sum(1)
    other=np.array([ratios[TYPES[t]] for t in typ])
    return np.column_stack([1-win,one,(win-one)[:,None]*other])
