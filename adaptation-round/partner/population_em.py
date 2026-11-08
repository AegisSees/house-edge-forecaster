"""Estimate the population distribution by integrating machine-level uncertainty."""
import copy
import numpy as np
from scipy.special import logsumexp
from state_model import TYPES


def fit_population_em(items,conditions,initial,iterations=100):
    selected=[i for i in items if conditions[i['query'].query_id]=='stable' and len(i['y'])>30]
    types=np.array([TYPES.index(i['query'].machine_type) for i in selected])
    signals=np.array([i['query'].signal for i in selected])
    bgrid=np.arange(-4.6,2.601,.2);agrid=np.arange(.5,5.501,.25);sign=np.array([-1.,1.])
    n=len(selected);patterns=initial['primary'];j=len(patterns)
    wins=np.zeros((n,j,4));counts=np.zeros_like(wins)
    for k,item in enumerate(selected):
        y=(item['y']>0).astype(float)
        for p,col in enumerate(patterns):
            context=2*(item['x'][:,col]>0).astype(int)+(item['x'][:,7]>0)
            counts[k,p]=np.bincount(context,minlength=4)
            wins[k,p]=np.bincount(context,weights=y,minlength=4)
    likelihood=np.zeros((n,j,2,len(bgrid),len(agrid)))
    for context in range(4):
        x=2*(context//2)-1;z=2*(context%2)-1
        eta=(bgrid[None,None,:,None]+initial['signal']*signals[:,None,None,None]
             +x*sign[None,:,None,None]*agrid[None,None,None,:]+initial['secondary']*z)
        likelihood+=wins[:,:,context,None,None,None]*eta[:,None,:,:,:]-counts[:,:,context,None,None,None]*np.logaddexp(0,eta[:,None,:,:,:])
    mb=np.array(initial['bias']);ma=np.full(3,initial['amplitude'])
    sb=np.full(3,initial['bias_sd']);sa=np.full(3,initial['amplitude_sd'])
    trace=[]
    for iteration in range(iterations):
        lb=-.5*((bgrid[None,:]-mb[types,None])/sb[types,None])**2
        la=-.5*((agrid[None,:]-ma[types,None])/sa[types,None])**2
        lb-=logsumexp(lb,axis=1)[:,None];la-=logsumexp(la,axis=1)[:,None]
        lp=likelihood+lb[:,None,None,:,None]+la[:,None,None,None,:]-np.log(j*2)
        normal=logsumexp(lp,axis=(1,2,3,4))
        post=np.exp(lp-normal[:,None,None,None,None]).sum(axis=(1,2))
        eb=(post*bgrid[None,:,None]).sum(axis=(1,2));eb2=(post*bgrid[None,:,None]**2).sum(axis=(1,2))
        ea=(post*agrid[None,None,:]).sum(axis=(1,2));ea2=(post*agrid[None,None,:]**2).sum(axis=(1,2))
        old=np.r_[mb,ma,sb,sa]
        for t in range(3):
            m=types==t
            mb[t]=eb[m].mean();ma[t]=ea[m].mean()
            sb[t]=np.sqrt(max(.1**2,eb2[m].mean()-mb[t]**2));sa[t]=np.sqrt(max(.15**2,ea2[m].mean()-ma[t]**2))
        trace.append(dict(iteration=iteration,log_likelihood=float(normal.sum()),bias=mb.tolist(),amplitude=ma.tolist(),bias_sd=sb.tolist(),amplitude_sd=sa.tolist()))
        if np.max(abs(np.r_[mb,ma,sb,sa]-old))<1e-5:break
    model=copy.deepcopy(initial)
    model.update(bias=mb.tolist(),amplitude_by_type=ma.tolist(),bias_sd_by_type=sb.tolist(),amplitude_sd_by_type=sa.tolist())
    return model,trace
