from __future__ import annotations
import argparse,json,os,torch
from datasets import load_dataset
from qfc.conditional import collect_per_sample_density_states,conditional_fidelity_kernels,conditional_greedy_select,conditional_weighted_greedy_select
from qfc.coverage import greedy_select,weighted_greedy_select
from qfc.fidelity import pairwise_fidelity
from qfc.hf_experiments import _move_batch,collect_mean_density_states,load_sequence_classifier,make_text_loader,michel_head_importance
from qfc.metrics import paired_bootstrap_delta

SPECS={
"sst2":{"model_id":"textattack/bert-base-uncased-SST-2","revision":"205ffbd1bc5c5b89802266f4948a601f53556b00","dataset":("stanfordnlp/sst2",None),"text_fields":("sentence",)},
"mrpc":{"model_id":"textattack/bert-base-uncased-MRPC","revision":"ddeddf4a04cd7b9415b00e40b00e78f0c61a7921","dataset":("glue","mrpc"),"text_fields":("sentence1","sentence2")}}

@torch.no_grad()
def evaluate(model,loader,device,head_mask=None):
    model.eval();losses=[];correct=[];labels=[];preds=[]
    for batch in loader:
        batch=_move_batch(batch,device); y=batch.pop("labels")
        out=model(**batch,labels=y,head_mask=head_mask,return_dict=True); z=out.logits.float(); p=z.argmax(-1).cpu(); yc=y.cpu()
        losses.append(torch.nn.functional.cross_entropy(z,y,reduction="none").cpu());correct.append((p==yc).float());labels.append(yc);preds.append(p)
    return torch.cat(losses),torch.cat(correct),torch.cat(labels),torch.cat(preds)

def f1(y,p):
    tp=((p==1)&(y==1)).sum().item();fp=((p==1)&(y==0)).sum().item();fn=((p==0)&(y==1)).sum().item();d=2*tp+fp+fn
    return 0.0 if d==0 else 2*tp/d

def mask(selection,L,H,device):
    m=torch.zeros((L,H),dtype=torch.float32,device=device)
    for l,hs in selection.items(): m[l,hs]=1.0
    return m

def main():
    p=argparse.ArgumentParser();p.add_argument("--task",choices=SPECS,required=True);p.add_argument("--seed",type=int,required=True)
    p.add_argument("--calibration-size",type=int,default=128);p.add_argument("--heads-to-keep",type=int,default=6);p.add_argument("--batch-size",type=int,default=16);p.add_argument("--max-length",type=int,default=128);p.add_argument("--bootstrap",type=int,default=1000);p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu");p.add_argument("--output-dir",default="results/step18_v10_confirmatory");a=p.parse_args()
    s=SPECS[a.task];dn,cfg=s["dataset"];ds=load_dataset(dn,cfg) if cfg else load_dataset(dn);train,valid=ds["train"],ds["validation"]
    cal=train.shuffle(seed=a.seed).select(range(a.calibration_size))
    model,tok=load_sequence_classifier(s["model_id"],a.device,revision=s["revision"])
    cl=make_text_loader(cal,tok,text_fields=s["text_fields"],batch_size=a.batch_size,max_length=a.max_length)
    vl=make_text_loader(valid,tok,text_fields=s["text_fields"],batch_size=a.batch_size,max_length=a.max_length)
    bl,bc,by,bp=evaluate(model,vl,a.device)
    ms=collect_mean_density_states(model,cl,a.device);ps=collect_per_sample_density_states(model,cl,a.device);ks=conditional_fidelity_kernels(ps,max_pairs_per_chunk=64)
    mic=michel_head_importance(model,cl,a.device).detach().cpu()
    from step13_mrpc_stability import shannon_scores,vn_scores
    sh=shannon_scores(model,cl,a.device);vn=vn_scores(ms);L=len(ms);H=int(ms[0].shape[0]);k=a.heads_to_keep
    sel={"MeanStateQFC":{},"ConditionalQFC":{},"MeanStateIWQFC":{},"ConditionalIWQFC":{},"VonNeumann":{},"Shannon":{},"MichelGate":{},"Random":{}}
    g=torch.Generator().manual_seed(2027+k);randperm=torch.randperm(H,generator=g)
    for l in range(L):
        sim=pairwise_fidelity(ms[l]);sel["MeanStateQFC"][l],_=greedy_select(sim,k);sel["ConditionalQFC"][l],_=conditional_greedy_select(ks[l],k)
        raw=mic[l].clamp_min(0);w=raw/raw.sum() if float(raw.sum())>0 else torch.full_like(raw,1.0/len(raw))
        sel["MeanStateIWQFC"][l],_=weighted_greedy_select(sim,w,k);sel["ConditionalIWQFC"][l],_=conditional_weighted_greedy_select(ks[l],w,k)
        sel["VonNeumann"][l]=torch.topk(vn[l],k).indices.tolist();sel["Shannon"][l]=torch.topk(sh[l],k).indices.tolist();sel["MichelGate"][l]=torch.topk(mic[l],k).indices.tolist();sel["Random"][l]=randperm.tolist()
    r={"task":a.task,"seed":a.seed,"model_id":s["model_id"],"model_revision":s["revision"],"calibration_size":a.calibration_size,"evaluation_size":len(valid),"heads_to_keep_per_layer":k,"baseline":{"accuracy":float(bc.mean()),"loss":float(bl.mean())},"methods":{}}
    if a.task=="mrpc":r["baseline"]["f1"]=f1(by,bp)
    for n,ss in sel.items():
        lo,co,y,pred=evaluate(model,vl,a.device,mask(ss,L,H,a.device));e={"accuracy":float(co.mean()),"loss":float(lo.mean()),"accuracy_delta":float(co.mean()-bc.mean()),"loss_delta":float(lo.mean()-bl.mean()),"accuracy_bootstrap_delta":paired_bootstrap_delta(co,bc,n_boot=a.bootstrap,seed=a.seed+30000),"selected_heads_zero_based":ss}
        if a.task=="mrpc":e["f1"]=f1(y,pred);e["f1_delta"]=e["f1"]-r["baseline"]["f1"]
        r["methods"][n]=e
    os.makedirs(a.output_dir,exist_ok=True);path=os.path.join(a.output_dir,f"{a.task}_seed{a.seed}.json");json.dump(r,open(path,"w",encoding="utf-8"),indent=2)
    print(f"{a.task} seed={a.seed} "+" | ".join(f"{n}={r['methods'][n]['accuracy']:.4f}" for n in sel))
if __name__=="__main__":main()
