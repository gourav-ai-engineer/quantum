from __future__ import annotations
import json
import sys
from itertools import combinations
from pathlib import Path
from statistics import mean,pstdev
def jac(a,b):
    A={(int(l),int(h)) for l,hs in a.items() for h in hs};B={(int(l),int(h)) for l,hs in b.items() for h in hs};U=A|B
    return 1.0 if not U else len(A&B)/len(U)
root=Path(sys.argv[1] if len(sys.argv)>1 else "v10_results");data=[json.loads(p.read_text()) for p in root.rglob("*.json") if p.name not in ("aggregate.json","run_meta.json")];out={"tasks":{}}
for task in sorted({x["task"] for x in data}):
    rs=sorted([x for x in data if x["task"]==task],key=lambda x:x["seed"]);td={"seeds":[x["seed"] for x in rs],"methods":{}}
    for m in rs[0]["methods"]:
        acc=[x["methods"][m]["accuracy"] for x in rs];loss=[x["methods"][m]["loss"] for x in rs];js=[jac(a["methods"][m]["selected_heads_zero_based"],b["methods"][m]["selected_heads_zero_based"]) for a,b in combinations(rs,2)]
        e={"accuracy_mean":mean(acc),"accuracy_std":pstdev(acc) if len(acc)>1 else 0.0,"loss_mean":mean(loss),"loss_std":pstdev(loss) if len(loss)>1 else 0.0,"selection_jaccard_mean":mean(js) if js else 1.0,"selection_jaccard_std":pstdev(js) if len(js)>1 else 0.0}
        if task=="mrpc":
            fs=[x["methods"][m]["f1"] for x in rs];e["f1_mean"]=mean(fs);e["f1_std"]=pstdev(fs) if len(fs)>1 else 0.0
        td["methods"][m]=e
    out["tasks"][task]=td
(root/"aggregate.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
