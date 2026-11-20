############################ EXP2_RELATIVE_VS_ABSOLUTE ############################
# Relative vs absolute features: 34 features (23 + 11 within-swarm z-scores) vs the 23 absolute ones, d = 16.
# Outputs: comparison table (AUPRC, train/val/test Spearman), metric bars, true-vs-predicted rank plots for one test swarm.
# Everything is saved in the folder ./exp2_relative_vs_absolute/ (created automatically). Single cell: run the whole file at once.
############################ CONFIGURATION ############################
import os
DATA_PATH=os.environ.get("SWARM_CSV",r"/media/user/Annu_Pressu/Siddharth/swarm_criticality_dataset_100_scenarios.csv")
RANDOM_STATE=42                # case split (same split as every earlier run)
GRAPH_THRESHOLD=55.0           # sensing range: edge if distance <= 55
EPOCHS=2000
PATIENCE=500                   # epochs without a new best validation Spearman before stopping
VAL_EVERY=10                   # validate / checkpoint every 10 epochs
LR=1e-3
WEIGHT_DECAY=1e-5
DROPOUT=0.0
ATTN_HIDDEN=16
SOFT_RANK_TEMPERATURE=0.1
LOGIT_ANCHOR=1e-3              # keeps pre-sigmoid scores from drifting into saturation (needed when BCE_WEIGHT = 0)
TIME_STRIDE=3                  # every 3rd sampled timestep as model input (labels always use all steps)
LABEL_MODE="mean"              # per-agent criticality target: mean over alive steps
CRITICAL_LABEL_MODE="any"      # per-agent is_critical label: critical at any alive step
TOP_K=[1,5,10]
SHOW_PLOTS=True                # also display figures inline (they are always saved as PNG + PDF)
EXAMPLE_CASE_ID=None           # test swarm used in single-swarm plots; None = the swarm with median test Spearman
EXPERIMENT_DIR="exp2_relative_vs_absolute"
D=16
SEED=42
BCE_WEIGHT=0.5
############################ SHARED CORE (identical in all five experiment scripts) ############################
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr,kendalltau
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score,average_precision_score,roc_auc_score,accuracy_score,precision_score,recall_score,f1_score

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(EXPERIMENT_DIR,exist_ok=True)
print("Experiment folder:",os.path.abspath(EXPERIMENT_DIR),"| device:",DEVICE,"| CPU threads:",torch.get_num_threads())

############################ FEATURES ############################
DYNAMIC_FEATURES=["vx","vy","vz","speed","accel","heading_rate","angular_coverage","sin_heading","cos_heading","sin_fpa","cos_fpa"]
POSITION_FEATURES=["rel_x","rel_y","rel_z","dist_to_centroid"]
NEIGHBOUR_FEATURES=["num_neighbors","mean_neighbor_dist","min_neighbor_dist","weighted_density"]
CENTRALITY_FEATURES=["degree_centrality","closeness_centrality","betweenness_centrality","eigenvector_centrality"]
RELATIVE_BASE_FEATURES=NEIGHBOUR_FEATURES+CENTRALITY_FEATURES+["dist_to_centroid","angular_coverage","speed"]
RELATIVE_FEATURES=[f"{f}_swarm_z" for f in RELATIVE_BASE_FEATURES]
GRAPH_POSITION_COLUMNS=["rel_x","rel_y","rel_z"]      # raw positions, used only to build edges and edge distances
def feature_list(use_relative):
    return DYNAMIC_FEATURES+POSITION_FEATURES+NEIGHBOUR_FEATURES+CENTRALITY_FEATURES+(RELATIVE_FEATURES if use_relative else [])
def feature_group(name):
    if name.endswith("_swarm_z"): return "Within-swarm relative"
    if name in DYNAMIC_FEATURES: return "Dynamics"
    if name in POSITION_FEATURES: return "Position"
    if name in NEIGHBOUR_FEATURES: return "Neighbourhood"
    return "Centrality"

############################ DATA ############################
def add_derived_features(df):
    df["sin_heading"]=np.sin(df["heading"])
    df["cos_heading"]=np.cos(df["heading"])
    df["sin_fpa"]=np.sin(df["flight_path_angle"])
    df["cos_fpa"]=np.cos(df["flight_path_angle"])
    df[["mean_neighbor_dist","min_neighbor_dist"]]=df[["mean_neighbor_dist","min_neighbor_dist"]].fillna(GRAPH_THRESHOLD)   # no neighbour in range
    groups=df.groupby(["case_id","step"])[RELATIVE_BASE_FEATURES]
    mean=groups.transform("mean")
    std=groups.transform("std")
    for f in RELATIVE_BASE_FEATURES:                     # z-score of each agent among the agents alive at the same (case, step)
        s=std[f].where(std[f]>0,1.0).fillna(1.0)
        df[f"{f}_swarm_z"]=(df[f]-mean[f])/s
    return df
def load_dataset(path):
    df=pd.read_csv(path)
    df=df.sort_values(["case_id","step","agent_id"]).reset_index(drop=True)
    df["criticality_score"]=df["criticality_score"].astype(float)
    df["is_critical_label"]=df["is_critical"].astype(str).str.strip().str.lower().isin(["true","1","1.0"]).astype(float)
    return add_derived_features(df)
def split_cases(df):
    meta=df.groupby("case_id")[["num_agents","attrition_rate","obstacle_density","clutter_index"]].first().reset_index()
    meta["attrition_clean"]=meta["attrition_rate"].astype(str).str.lower().str.strip()
    holdout=(meta["num_agents"].isin([52,68,75]))|(meta["attrition_clean"].isin(["high","heavy","light"]))|(meta["obstacle_density"]<0.07)
    val,test=train_test_split(meta[holdout],test_size=0.50,random_state=RANDOM_STATE)
    return meta,meta[~holdout]["case_id"].tolist(),val["case_id"].tolist(),test["case_id"].tolist()
def normalization_stats(df,train_ids,features):
    train=df[df["case_id"].isin(train_ids)]
    median=train[features].median()
    values=train[features].fillna(median)
    return {"median":median,"mean":values.mean(),"std":values.std().replace(0,1).fillna(1)}
def prepare_case(g,features,stats):
    case_id=g["case_id"].iloc[0]
    by_agent=g.groupby("agent_id")
    score=by_agent["criticality_score"].mean() if LABEL_MODE=="mean" else (by_agent["criticality_score"].first() if LABEL_MODE=="first" else by_agent["criticality_score"].last())
    crit=by_agent["is_critical_label"].max() if CRITICAL_LABEL_MODE=="any" else (by_agent["is_critical_label"].first() if CRITICAL_LABEL_MODE=="first" else by_agent["is_critical_label"].last())
    steps=np.sort(g["step"].unique())[::TIME_STRIDE]
    g=g[g["step"].isin(steps)]
    agents=np.sort(g["agent_id"].unique())
    T,N=len(steps),len(agents)
    idx=pd.MultiIndex.from_product([steps,agents],names=["step","agent_id"])
    p=g.set_index(["step","agent_id"]).reindex(idx)
    alive=torch.tensor((~p["case_id"].isna()).to_numpy().reshape(T,N))
    x=((p[features].fillna(stats["median"])-stats["mean"])/stats["std"]).to_numpy(np.float32).reshape(T,N,len(features))
    pos=torch.tensor(np.nan_to_num(p[GRAPH_POSITION_COLUMNS].to_numpy(np.float32).reshape(T,N,3),nan=0.0))
    y=score.reindex(agents).to_numpy(np.float32)
    dist=torch.cdist(pos,pos)
    adj=(dist<=GRAPH_THRESHOLD)&alive.unsqueeze(2)&alive.unsqueeze(1)&~torch.eye(N,dtype=torch.bool).unsqueeze(0)
    flat=torch.full((T,N),-1,dtype=torch.long)
    flat[alive]=torch.arange(int(alive.sum()))
    et,es,ed=torch.nonzero(adj,as_tuple=True)
    step_obs=torch.arange(T).unsqueeze(1).expand(T,N)[alive]
    return {"case_id":case_id,"agent_ids":agents,"steps":steps,"x":torch.tensor(x)[alive],
            "edge_index":torch.stack([flat[et,es],flat[et,ed]]),"edge_attr":(dist[et,es,ed]/GRAPH_THRESHOLD).unsqueeze(1).float(),
            "agent_obs_index":torch.arange(N).unsqueeze(0).expand(T,N)[alive],"step_obs_index":step_obs,"t_norm":step_obs.float()/max(T-1,1),
            "obs_per_agent":alive.sum(0),"y":torch.tensor(y),"y_rank":torch.tensor(pd.Series(y).rank(method="average").to_numpy(np.float32)),
            "y_crit":torch.tensor(crit.reindex(agents).to_numpy(np.float32)),"num_agents":N,"num_steps":T}
def prepare_cases(df,case_ids,features,stats):
    sub=df[df["case_id"].isin(case_ids)]
    groups={cid:g for cid,g in sub.groupby("case_id")}
    return [prepare_case(groups[cid],features,stats) for cid in case_ids]

############################ MODEL ############################
def segment_mean(values,index,n):
    out=torch.zeros(n,values.size(1),device=values.device).index_add_(0,index,values)
    cnt=torch.zeros(n,1,device=values.device).index_add_(0,index,torch.ones(index.size(0),1,device=values.device))
    return out/cnt.clamp_min(1)
def segment_softmax(scores,index,n):
    seg_max=torch.full((n,),float("-inf"),device=scores.device).scatter_reduce(0,index,scores.detach(),reduce="amax",include_self=True)
    e=torch.exp(scores-seg_max[index])
    return e/torch.zeros(n,device=scores.device).index_add_(0,index,e)[index].clamp_min(1e-12)
class GINELayer(nn.Module):
    # h_i' = MLP((1+eps) h_i + sum_j ReLU(h_j + W_e e_ij)),  e_ij = distance / 55
    def __init__(self,in_dim,hidden_dim,out_dim,dropout):
        super().__init__()
        self.eps=nn.Parameter(torch.zeros(1))
        self.edge_proj=nn.Linear(1,in_dim)
        self.mlp=nn.Sequential(nn.Linear(in_dim,hidden_dim),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_dim,out_dim))
    def forward(self,x,edge_index,edge_attr):
        agg=torch.zeros_like(x)
        if edge_index.numel()>0:
            agg.index_add_(0,edge_index[1],F.relu(x[edge_index[0]]+self.edge_proj(edge_attr)))
        return self.mlp((1+self.eps)*x+agg)
class FeatureAttention(nn.Module):
    # beta = softmax(a): one weight per feature, shared by every agent, timestep and case; x~ = F * beta * x
    def __init__(self,num_features):
        super().__init__()
        self.logits=nn.Parameter(torch.zeros(num_features))
    def weights(self):
        return torch.softmax(self.logits,dim=0)
    def forward(self,x):
        return x*(self.weights()*x.size(1))
class TemporalAttentionPool(nn.Module):
    # e = w^T tanh(W [z ; t/(T-1)] + b); softmax over each agent's own alive steps; z_bar = sum alpha z
    def __init__(self,d,hidden_dim):
        super().__init__()
        self.score=nn.Sequential(nn.Linear(d+1,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,1))
    def forward(self,z,t_norm,agent_index,n_agents):
        alpha=segment_softmax(self.score(torch.cat([z,t_norm.unsqueeze(1)],1)).squeeze(1),agent_index,n_agents)
        return torch.zeros(n_agents,z.size(1),device=z.device).index_add_(0,agent_index,alpha.unsqueeze(1)*z),alpha
class CriticalGINE(nn.Module):
    def __init__(self,input_dim,d,attn_hidden,dropout):
        super().__init__()
        self.feature_attention=FeatureAttention(input_dim)
        self.gin1=GINELayer(input_dim,64,64,dropout)
        self.norm1=nn.LayerNorm(64)
        self.gin2=GINELayer(64,64,32,dropout)
        self.norm2=nn.LayerNorm(32)
        self.global_mix=nn.Linear(64,32)
        self.norm_global=nn.LayerNorm(32)
        self.gin3=GINELayer(32,32,d,dropout)
        self.temporal_pool=TemporalAttentionPool(d,attn_hidden)
        self.head=nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Dropout(dropout),nn.Linear(32,1),nn.Sigmoid())
    def forward(self,c):
        x=self.feature_attention(c["x"])
        h=F.relu(self.norm1(self.gin1(x,c["edge_index"],c["edge_attr"])))
        h=F.relu(self.norm2(self.gin2(h,c["edge_index"],c["edge_attr"])))
        swarm=segment_mean(h,c["step_obs_index"],c["num_steps"])
        h=F.relu(self.norm_global(self.global_mix(torch.cat([h,swarm[c["step_obs_index"]]],1))))
        z=self.gin3(h,c["edge_index"],c["edge_attr"])
        pooled,alpha=self.temporal_pool(z,c["t_norm"],c["agent_obs_index"],c["num_agents"])
        self.last_logit=self.head[:-1](pooled).squeeze(1)       # pre-sigmoid score (used by the ranking loss)
        return self.head[-1](self.last_logit),z,pooled,alpha
def move_case(c):
    return {k:(v.to(DEVICE) if torch.is_tensor(v) else v) for k,v in c.items()}
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

############################ LOSSES ############################
def soft_rank(x,temperature):
    x=(x-x.mean())/(x.std()+1e-8)
    return 0.5+torch.sigmoid((x.unsqueeze(1)-x.unsqueeze(0))/temperature).sum(1)
def pearson(a,b):
    a=a-a.mean()
    b=b-b.mean()
    return (a*b).sum()/(a.norm()*b.norm()+1e-8)
def soft_spearman_loss(pred,true_rank):
    return 1.0-pearson(soft_rank(pred,SOFT_RANK_TEMPERATURE),true_rank)
def ranking_loss(logit,true_rank):
    # Soft Spearman is invariant to the scale and offset of the scores, so on its own nothing stops them from drifting.
    # Computed on sigmoid outputs, that drift ends in saturation (p == 1.0 in float32), tied scores and zero gradients.
    # Fix: rank on the pre-sigmoid logit (sigmoid is monotonic, so the ranking is identical) + a tiny anchor on its size.
    return soft_spearman_loss(logit,true_rank)+LOGIT_ANCHOR*(logit**2).mean()
def weighted_bce(p,y,pos_weight):
    p=p.clamp(1e-6,1-1e-6)
    return -(pos_weight*y*torch.log(p)+(1-y)*torch.log(1-p)).mean()

############################ METRICS ############################
def ranking_metrics(y,pred):
    out={}
    ok=np.std(y)>0 and np.std(pred)>0
    out["spearman"]=spearmanr(y,pred)[0] if ok else np.nan
    out["kendall"]=kendalltau(y,pred)[0] if ok else np.nan
    true_order,pred_order=np.argsort(-y),np.argsort(-pred)
    for k in TOP_K:
        kk=min(k,len(y))
        out[f"precision@{k}"]=len(set(true_order[:kk])&set(pred_order[:kk]))/kk
        out[f"ndcg@{k}"]=ndcg_score(y[None,:],pred[None,:],k=kk) if len(y)>1 else np.nan
    return out
def per_swarm_table(per_case,meta):
    rows=[]
    for c in per_case:
        both=0<c["y_crit"].sum()<len(c["y_crit"])
        rows.append({"case_id":c["case_id"],"agents":len(c["y"]),"critical_agents":int(c["y_crit"].sum()),**ranking_metrics(c["y"],c["pred"]),
                     "auprc":average_precision_score(c["y_crit"],c["pred"]) if both else np.nan})
    return pd.DataFrame(rows).merge(meta[["case_id","num_agents","attrition_clean","obstacle_density","clutter_index"]],on="case_id",how="left")
def pooled(per_case,key):
    return np.concatenate([c[key] for c in per_case])
def classification_metrics(per_case,threshold):
    y,p=pooled(per_case,"y_crit"),pooled(per_case,"pred")
    yhat=(p>=threshold).astype(int)
    both=0<y.sum()<len(y)
    return {"threshold":threshold,"auprc":average_precision_score(y,p) if both else np.nan,"roc_auc":roc_auc_score(y,p) if both else np.nan,
            "prevalence":float(y.mean()),"accuracy":accuracy_score(y,yhat),"precision":precision_score(y,yhat,zero_division=0),
            "recall":recall_score(y,yhat,zero_division=0),"f1":f1_score(y,yhat,zero_division=0),
            "TP":int(((yhat==1)&(y==1)).sum()),"FP":int(((yhat==1)&(y==0)).sum()),"FN":int(((yhat==0)&(y==1)).sum()),"TN":int(((yhat==0)&(y==0)).sum())}
def best_f1_threshold(per_case):
    y,p=pooled(per_case,"y_crit"),pooled(per_case,"pred")
    grid=np.linspace(0.01,0.99,99)
    return float(grid[int(np.argmax([f1_score(y,(p>=t).astype(int),zero_division=0) for t in grid]))])
def split_summary(per_case,meta,threshold):
    table=per_swarm_table(per_case,meta)
    out={}
    for col in ["spearman","kendall"]+[f"precision@{k}" for k in TOP_K]+[f"ndcg@{k}" for k in TOP_K]:
        out[f"{col}_mean"]=table[col].mean()
        out[f"{col}_std"]=table[col].std()
    out["per_swarm_auprc_mean"]=table["auprc"].mean()
    out.update(classification_metrics(per_case,threshold))
    return out,table

############################ TRAINING ############################
DATASET_CACHE={}
def get_datasets(df,meta,train_ids,val_ids,test_ids,use_relative):
    if use_relative not in DATASET_CACHE:
        features=feature_list(use_relative)
        stats=normalization_stats(df,train_ids,features)
        print(f"Preparing graphs ({len(features)} features)...")
        DATASET_CACHE[use_relative]={"features":features,"stats":stats,"train":prepare_cases(df,train_ids,features,stats),
                                     "val":prepare_cases(df,val_ids,features,stats),"test":prepare_cases(df,test_ids,features,stats)}
    return DATASET_CACHE[use_relative]
def predict(model,data):
    model.eval()
    out=[]
    with torch.inference_mode():
        for raw in data:
            c=move_case(raw)
            scores,_,_,_=model(c)
            out.append({"case_id":raw["case_id"],"agent_ids":raw["agent_ids"],"y":raw["y"].numpy(),"y_crit":raw["y_crit"].numpy(),"pred":scores.cpu().numpy()})
    return out
def train_model(train_data,val_data,n_features,d,bce_weight,seed,label,checkpoint=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng=np.random.default_rng(seed)
    model=CriticalGINE(n_features,d,ATTN_HIDDEN,DROPOUT).to(DEVICE)
    if checkpoint is not None and os.path.exists(checkpoint):          # reuse a finished run instead of retraining
        model.load_state_dict(torch.load(checkpoint,map_location=DEVICE)["model_state_dict"])
        prev=pd.read_csv(os.path.join(os.path.dirname(checkpoint),"summary.csv")).iloc[0]
        print(f"\n[{label}] loaded finished run from {checkpoint} (best epoch {int(prev['best_epoch'])}) - not retrained")
        return model,{"best_epoch":int(prev["best_epoch"]),"best_val_spearman":float(prev["best_val_spearman"]),"training_minutes":float(prev["training_minutes"]),
                      "pos_weight":float(prev["pos_weight"]),"parameters":count_parameters(model)}
    optimizer=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    n_pos=sum(float(c["y_crit"].sum()) for c in train_data)
    pos_weight=sum(float((1-c["y_crit"]).sum()) for c in train_data)/max(n_pos,1.0)
    best,best_state,best_epoch,waited=-np.inf,None,0,0
    start=time.time()
    print(f"\n[{label}] training | d={d} | features={n_features} | BCE weight={bce_weight} | parameters={count_parameters(model)}")
    for epoch in range(1,EPOCHS+1):
        model.train()
        total,total_rank,total_bce,agents=0.0,0.0,0.0,0
        for k in rng.permutation(len(train_data)):
            c=move_case(train_data[k])
            optimizer.zero_grad()
            scores,_,_,_=model(c)
            rank_loss=ranking_loss(model.last_logit,c["y_rank"])
            bce_loss=weighted_bce(scores,c["y_crit"],pos_weight)       # always computed and printed; enters the loss only if bce_weight > 0
            loss=rank_loss+bce_weight*bce_loss
            loss.backward()
            optimizer.step()
            n=c["num_agents"]
            total+=loss.item()*n
            total_rank+=rank_loss.item()*n
            total_bce+=bce_loss.item()*n
            agents+=n
        if epoch%VAL_EVERY!=0 and epoch!=EPOCHS:
            continue
        val_pred=predict(model,val_data)
        val_rho=np.nanmean([ranking_metrics(c["y"],c["pred"])["spearman"] for c in val_pred])
        val_auprc=classification_metrics(val_pred,0.5)["auprc"]
        if np.isfinite(val_rho) and val_rho>best:
            best,best_epoch,waited=val_rho,epoch,0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            waited+=VAL_EVERY
        print(f"[{label}] epoch {epoch:4d} | train loss {total/agents:.4f} = rank {total_rank/agents:.4f} + {bce_weight}×BCE {total_bce/agents:.4f} | "
              f"val Spearman {val_rho:.4f} | val AUPRC {val_auprc:.4f} | best epoch {best_epoch} | {(time.time()-start)/60:.1f} min")
        if waited>=PATIENCE:
            break
    model.load_state_dict(best_state)
    return model,{"best_epoch":best_epoch,"best_val_spearman":best,"training_minutes":(time.time()-start)/60,"pos_weight":pos_weight,"parameters":count_parameters(model)}
def run_configuration(df,meta,split_ids,d,use_relative,bce_weight,seed,run_dir,label,reuse=False):
    # trains one configuration, evaluates train/val/test, saves its checkpoint, predictions and per-swarm metrics in run_dir
    os.makedirs(run_dir,exist_ok=True)
    data=get_datasets(df,meta,*split_ids,use_relative)
    model,info=train_model(data["train"],data["val"],len(data["features"]),d,bce_weight,seed,label,
                           checkpoint=os.path.join(run_dir,"model.pt") if reuse else None)
    preds={s:predict(model,data[s]) for s in ["train","val","test"]}
    threshold=best_f1_threshold(preds["val"])            # fitted on validation only
    summary={"label":label,"d":d,"use_relative":use_relative,"bce_weight":bce_weight,"seed":seed,"num_features":len(data["features"]),**info,"threshold":threshold}
    tables={}
    for s in ["train","val","test"]:
        out,tables[s]=split_summary(preds[s],meta,threshold)
        summary.update({f"{s}_{k}":v for k,v in out.items()})
        tables[s].to_csv(os.path.join(run_dir,f"{s}_per_swarm_metrics.csv"),index=False)
    rows=[]
    for c in preds["test"]:
        true_rank=pd.Series(c["y"]).rank(ascending=False,method="min").astype(int).to_numpy()
        pred_rank=pd.Series(c["pred"]).rank(ascending=False,method="min").astype(int).to_numpy()
        for i,a in enumerate(c["agent_ids"]):
            rows.append({"case_id":c["case_id"],"agent_id":a,"criticality_score":c["y"][i],"predicted_score":c["pred"][i],"true_rank":true_rank[i],
                         "predicted_rank":pred_rank[i],"is_critical":int(c["y_crit"][i]),"predicted_critical":int(c["pred"][i]>=threshold)})
    pd.DataFrame(rows).to_csv(os.path.join(run_dir,"test_predictions.csv"),index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(run_dir,"summary.csv"),index=False)
    torch.save({"model_state_dict":model.state_dict(),"features":data["features"],"d":d,"use_relative":use_relative,"bce_weight":bce_weight,
                "threshold":threshold,"normalization":{k:v.to_dict() for k,v in data["stats"].items()}},os.path.join(run_dir,"model.pt"))
    print(f"[{label}] test Spearman {summary['test_spearman_mean']:.4f} | test AUPRC {summary['test_auprc']:.4f} | test F1 {summary['test_f1']:.4f} | saved in {run_dir}")
    return {"model":model,"summary":summary,"preds":preds,"tables":tables,"data":data,"threshold":threshold}

############################ PLOT STYLE (slide-ready) ############################
BLUE,ORANGE,AQUA,YELLOW,MAGENTA="#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4"
INK,INK2,MUTED,GRID,SURFACE="#0b0b0b","#52514e","#8a8984","#e6e5e0","#ffffff"
GROUP_COLORS={"Dynamics":BLUE,"Position":ORANGE,"Neighbourhood":AQUA,"Centrality":YELLOW,"Within-swarm relative":MAGENTA}
SPLIT_COLORS={"train":BLUE,"test":ORANGE}
plt.rcParams.update({"figure.facecolor":SURFACE,"axes.facecolor":SURFACE,"savefig.facecolor":SURFACE,"font.size":12,"axes.titlesize":15,
                     "axes.titleweight":"semibold","axes.titlelocation":"left","axes.titlepad":30,"axes.labelsize":12,"axes.labelcolor":INK2,
                     "axes.edgecolor":"#c9c8c2","axes.linewidth":0.8,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
                     "grid.color":GRID,"grid.linewidth":0.8,"axes.axisbelow":True,"xtick.color":INK2,"ytick.color":INK2,"xtick.labelsize":11,
                     "ytick.labelsize":11,"legend.frameon":False,"legend.fontsize":11,"text.color":INK,"lines.linewidth":2.0,"lines.markersize":8})
def save_figure(fig,name):
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(os.path.join(EXPERIMENT_DIR,f"{name}.{ext}"),dpi=220,bbox_inches="tight")
    print("FIGURE SAVED:",os.path.join(EXPERIMENT_DIR,f"{name}.png"))
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
def subtitle(ax,text):
    ax.text(0,1.02,text,transform=ax.transAxes,fontsize=11,color=INK2,va="bottom")
def pick_example_case(run):
    # the test swarm whose Spearman is closest to the median: a representative, not a cherry-picked, example
    if EXAMPLE_CASE_ID is not None:
        return EXAMPLE_CASE_ID
    t=run["tables"]["test"].dropna(subset=["spearman"])
    return t.iloc[(t["spearman"]-t["spearman"].median()).abs().argsort().iloc[0]]["case_id"]
def plot_rank_agreement(ax,run,case_id,title):
    c=next(c for c in run["preds"]["test"] if c["case_id"]==case_id)
    true_rank=pd.Series(c["y"]).rank(ascending=False,method="average").to_numpy()
    pred_rank=pd.Series(c["pred"]).rank(ascending=False,method="average").to_numpy()
    n=len(true_rank)
    ax.plot([1,n],[1,n],color=MUTED,lw=1.2,ls=(0,(4,3)),zorder=1)
    crit=c["y_crit"]>0.5
    ax.scatter(true_rank[~crit],pred_rank[~crit],s=46,color=BLUE,edgecolor=SURFACE,linewidth=1.2,zorder=3,label="Decoy")
    ax.scatter(true_rank[crit],pred_rank[crit],s=70,color=ORANGE,edgecolor=SURFACE,linewidth=1.2,zorder=4,label="Critical asset")
    rho=spearmanr(c["y"],c["pred"])[0]
    ax.set_xlim(0,n+1)
    ax.set_ylim(0,n+1)
    ax.set_aspect("equal")
    ax.set_xlabel("True rank (1 = most critical)")
    ax.set_ylabel("Predicted rank")
    ax.set_title(title)
    subtitle(ax,f"{case_id}  ·  {n} agents  ·  Spearman ρ = {rho:.3f}")
    ax.text(n*0.97,n*0.06,"perfect ranking on the dashed line",ha="right",fontsize=10,color=MUTED)
    ax.legend(loc="upper left")

############################ TWO-MODEL COMPARISON HELPERS ############################
def comparison_table(runs):
    rows=[]
    for name,run in runs.items():
        s=run["summary"]
        rows.append({"model":name,"features":s["num_features"],"bce_weight":s["bce_weight"],"best_epoch":s["best_epoch"],
                     "train_spearman":s["train_spearman_mean"],"val_spearman":s["val_spearman_mean"],"test_spearman":s["test_spearman_mean"],
                     "test_spearman_sd_swarms":s["test_spearman_std"],"test_ndcg@10":s["test_ndcg@10_mean"],"test_precision@5":s["test_precision@5_mean"],
                     "test_auprc":s["test_auprc"],"test_f1":s["test_f1"],"test_precision":s["test_precision"],"test_recall":s["test_recall"]})
    return pd.DataFrame(rows)
def plot_comparison_bars(table,metrics,title,note,name):
    labels=[m[1] for m in metrics]
    x=np.arange(len(metrics))
    width=0.36
    fig,ax=plt.subplots(figsize=(9,5))
    for j,(model,color) in enumerate(zip(table["model"],[BLUE,ORANGE])):
        vals=table.loc[table["model"]==model,[m[0] for m in metrics]].to_numpy().ravel()
        bars=ax.bar(x+(j-0.5)*width,vals,width=width*0.94,color=color,label=model,zorder=3)
        for b,v in zip(bars,vals):
            ax.text(b.get_x()+b.get_width()/2,v+(0.012 if v>=0 else -0.012),f"{v:.3f}",ha="center",va="bottom" if v>=0 else "top",fontsize=10,color=INK2)
    all_vals=table[[m[0] for m in metrics]].to_numpy().ravel()
    ax.axhline(0,color="#c9c8c2",lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(min(0,np.nanmin(all_vals)-0.12),1.12)
    ax.set_ylabel("Score")
    ax.set_title(title)
    subtitle(ax,note)
    ax.legend(loc="upper right",ncol=2)
    ax.grid(axis="x",visible=False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x",length=0)
    save_figure(fig,name)
def plot_rank_pair(runs,case_id,title,name_prefix):
    names=list(runs)
    for name,slug in zip(names,["a","b"]):
        fig,ax=plt.subplots(figsize=(6.4,6.4))
        plot_rank_agreement(ax,runs[name],case_id,name)
        save_figure(fig,f"{name_prefix}_{slug}_{name.lower().replace(' ','_').replace('(','').replace(')','').replace('+','plus').replace('-','')}")
    fig,axes=plt.subplots(1,2,figsize=(13,6.6))
    for ax,name in zip(axes,names):
        plot_rank_agreement(ax,runs[name],case_id,name)
    save_figure(fig,f"{name_prefix}_side_by_side")

############################ EXPERIMENT 2: RELATIVE vs ABSOLUTE FEATURES ############################
df=load_dataset(DATA_PATH)
meta,train_ids,val_ids,test_ids=split_cases(df)
split_ids=(train_ids,val_ids,test_ids)
print(f"Cases: train {len(train_ids)} | validation {len(val_ids)} | test {len(test_ids)}")
runs={"With relative features":run_configuration(df,meta,split_ids,D,True,BCE_WEIGHT,SEED,os.path.join(EXPERIMENT_DIR,"with_relative_features"),"with relative"),
      "Without relative features":run_configuration(df,meta,split_ids,D,False,BCE_WEIGHT,SEED,os.path.join(EXPERIMENT_DIR,"without_relative_features"),"without relative")}
table=comparison_table(runs)
table.to_csv(os.path.join(EXPERIMENT_DIR,"relative_vs_absolute_table.csv"),index=False)
print("\nRELATIVE vs ABSOLUTE FEATURES (d =",D,"| BCE weight",BCE_WEIGHT,")")
print(table.to_string(index=False,float_format=lambda v:f"{v:.4f}"))
plot_comparison_bars(table,[("train_spearman","Train Spearman ρ"),("test_spearman","Test Spearman ρ"),("test_auprc","Test AUPRC"),("test_f1","Test F1")],
                     "Within-swarm relative features",f"34 features (23 absolute + 11 relative) vs 23 absolute · d = {D}","relative_vs_absolute_metrics")
case_id=pick_example_case(runs["With relative features"])
print("Example test swarm for the rank plots:",case_id)
plot_rank_pair(runs,case_id,"True vs predicted ranking in one test swarm","rank_agreement")
