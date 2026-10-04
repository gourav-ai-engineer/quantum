from __future__ import annotations
import argparse,json,os,torch
from datasets import load_dataset
from qfc.baselines import calibrated_selections,evaluate_random_distribution,heads_kept_per_layer,validate_all
from qfc.conditional import collect_per_sample_density_states,conditional_fidelity_kernels
from qfc.hf_experiments import _move_batch,collect_mean_density_states,head_mask_from_selection,load_sequence_classifier,make_text_loader,michel_head_importance
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

def main():
    p=argparse.ArgumentParser();p.add_argument("--task",choices=SPECS,required=True);p.add_argument("--seed",type=int,required=True)
    p.add_argument("--calibration-size",type=int,default=128);p.add_argument("--evaluation-size",type=int,default=-1,help="-1 = full validation split");p.add_argument("--heads-to-keep",type=int,default=6);p.add_argument("--random-masks",type=int,default=30);p.add_argument("--batch-size",type=int,default=16);p.add_argument("--max-length",type=int,default=128);p.add_argument("--bootstrap",type=int,default=1000);p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu");p.add_argument("--output-dir",default="results/step18_v10_confirmatory");a=p.parse_args()
    s=SPECS[a.task];dn,cfg=s["dataset"];ds=load_dataset(dn,cfg) if cfg else load_dataset(dn);train,valid=ds["train"],ds["validation"]
    if a.evaluation_size>=0:
        if a.evaluation_size>len(valid):raise ValueError("evaluation-size exceeds validation split")
        valid=valid.select(range(a.evaluation_size))
    cal=train.shuffle(seed=a.seed).select(range(a.calibration_size))
    model,tok=load_sequence_classifier(s["model_id"],a.device,revision=s["revision"])
    cl=make_text_loader(cal,tok,text_fields=s["text_fields"],batch_size=a.batch_size,max_length=a.max_length)
    vl=make_text_loader(valid,tok,text_fields=s["text_fields"],batch_size=a.batch_size,max_length=a.max_length)
    bl,bc,by,bp=evaluate(model,vl,a.device)
    ms=collect_mean_density_states(model,cl,a.device);ps=collect_per_sample_density_states(model,cl,a.device);ks=conditional_fidelity_kernels(ps,max_pairs_per_chunk=64)
    mic=michel_head_importance(model,cl,a.device).detach().cpu()
    from step13_mrpc_stability import shannon_scores,vn_scores
    sh=shannon_scores(model,cl,a.device);vn=vn_scores(ms);L=len(ms);H=int(ms[0].shape[0]);k=a.heads_to_keep
    sel=calibrated_selections(ms,ks,mic,vn,sh,k)
    validate_all(sel,L,H,k)
    r={"task":a.task,"seed":a.seed,"model_id":s["model_id"],"model_revision":s["revision"],"calibration_size":a.calibration_size,"evaluation_size":len(valid),"heads_to_keep_per_layer":k,"random_masks":a.random_masks,"baseline":{"accuracy":float(bc.mean()),"loss":float(bl.mean())},"methods":{},"research_note":"Entropy baselines are reported keep-high and keep-low. Random is the mean over random_masks independent masks (seed = 2027 + 1000*k + i); see methods.Random.distribution for mean/std/min/max."}
    if a.task=="mrpc":r["baseline"]["f1"]=f1(by,bp)
    for n,ss in sel.items():
        lo,co,y,pred=evaluate(model,vl,a.device,head_mask_from_selection(L,H,ss,a.device));e={"accuracy":float(co.mean()),"loss":float(lo.mean()),"accuracy_delta":float(co.mean()-bc.mean()),"loss_delta":float(lo.mean()-bl.mean()),"accuracy_bootstrap_delta":paired_bootstrap_delta(co,bc,n_boot=a.bootstrap,seed=a.seed+30000),"selected_heads_zero_based":ss}
        if a.task=="mrpc":e["f1"]=f1(y,pred);e["f1_delta"]=e["f1"]-r["baseline"]["f1"]
        r["methods"][n]=e
    r["methods"]["Random"]=evaluate_random_distribution(lambda m:evaluate(model,vl,a.device,m),L,H,k,a.random_masks,2027+1000*k,a.device,with_f1=a.task=="mrpc",baseline=r["baseline"])
    r["heads_kept_per_layer"]={n:heads_kept_per_layer(m["selected_heads_zero_based"]) for n,m in r["methods"].items()}
    for n,kept in r["heads_kept_per_layer"].items():
        if kept!=[k]*L:raise RuntimeError(f"{n} keeps {kept}, expected {k} per layer")
    os.makedirs(a.output_dir,exist_ok=True);path=os.path.join(a.output_dir,f"{a.task}_seed{a.seed}.json");json.dump(r,open(path,"w",encoding="utf-8"),indent=2)
    print(f"{a.task} seed={a.seed} eval_n={len(valid)} baseline_acc={r['baseline']['accuracy']:.4f} baseline_loss={r['baseline']['loss']:.4f}"+(f" baseline_f1={r['baseline']['f1']:.4f}" if a.task=="mrpc" else ""))
    for n,m in r["methods"].items():print(f"  {n:20s} acc={m['accuracy']:.4f} loss={m['loss']:.4f} heads_kept_per_layer={r['heads_kept_per_layer'][n]}"+(f" f1={m['f1']:.4f}" if a.task=="mrpc" else ""))
    rd=r["methods"]["Random"]["distribution"];print("  Random distribution (n=%d): "%a.random_masks+" ".join(f"{key}[mean={v['mean']:.4f} std={v['std']:.4f} min={v['min']:.4f} max={v['max']:.4f}]" for key,v in rd.items()))
if __name__=="__main__":main()
