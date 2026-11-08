"""Refit the friend's joint Bayesian filter (entry/partner) on a new kit's training split.
usage: python partner_refit.py <partner folder> <kit public folder> <out model.json> [ref model.json]
Keeps the friend's settings (configs, prior width); re-estimates win pairs (lags 1-8), population,
size lags, payout ratios and frequency priors from train data. Writes round/season from metadata."""
import csv, json, sys, time
from pathlib import Path
t0 = time.time()
pdir, pub, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
ref = json.loads(Path(sys.argv[4]).read_text()) if len(sys.argv) > 4 else json.loads((pdir / "model.json").read_text())
sys.path.insert(0, str(pdir))
import numpy as np
from adaptive_model import prepare, fit_priors
from state_model import fit_population, amount_priors
from population_em import fit_population_em
from joint_model import fit_size_relation
from slot_forecasting.io import load_inputs, read_labels
q, h = load_inputs(pub / "train")
items = prepare(q, h)
cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(pub / "train" / "conditions.csv"))}
pop0 = fit_population(items, cond)
pop, trace = fit_population_em(items, cond, pop0)
size = fit_size_relation(items, pop, cond)
ratios = amount_priors(items)
try:
    labels = read_labels(pub / "train" / "labels.csv", q)
except Exception as e:
    print("labels not used:", e); labels = None
priors = fit_priors(q, h, labels)
meta = json.loads((pub / "metadata.json").read_text())
comp = dict(ref["components"][0]); comp["population"] = pop; comp["weight"] = 1.0
model = dict(ref); model.update(components=[comp], size_model=size, amount_ratios=ratios, priors=priors,
                                round=meta["round"], season_id=meta["season_id"])
out.write_text(json.dumps(model))
pairs = [(lambda a, b: (a + 1, b + 1))(*__import__("adaptive_model").PAIRS[c - 9]) for c in pop["primary"]]
print(f"primary {pop['primary']} pairs {pairs} size lags {size['lags']} em iters {len(trace)} {time.time()-t0:.0f}s")
