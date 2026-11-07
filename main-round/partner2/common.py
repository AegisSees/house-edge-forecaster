import csv, collections
PAY=(0,1,2,5,20)
def load(split, root='public'):
    D=f'{root}/{split}'
    q=list(csv.DictReader(open(f'{D}/queries.csv')))
    for r in q: r['cutoff']=int(r['cutoff']); r['signal']=int(r['signal'])
    H=collections.defaultdict(dict)
    for r in csv.DictReader(open(f'{D}/history.csv')):
        H[(r['episode_id'],r['machine_id'])][int(r['tick'])]=int(r['payout'])
    out={'q':q,'H':H}
    try: out['cond']={r['query_id']:r['condition'] for r in csv.DictReader(open(f'{D}/conditions.csv'))}
    except FileNotFoundError: pass
    try: out['lab']={r['query_id']:int(r['outcome']) for r in csv.DictReader(open(f'{D}/labels.csv'))}
    except FileNotFoundError: pass
    try: out['tgt']={r['query_id']:[float(r[k]) for k in ('p_0','p_1','p_2','p_5','p_20')] for r in csv.DictReader(open(f'{D}/targets.csv'))}
    except FileNotFoundError: pass
    return out
