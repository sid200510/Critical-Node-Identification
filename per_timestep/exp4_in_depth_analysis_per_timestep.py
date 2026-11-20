############################ EXP4_IN_DEPTH_ANALYSIS_PER_TIMESTEP ############################
# Per-timestep model (no temporal aggregation).
# In-depth analysis of the final model (relative features + BCE, d = 16):
# 1. feature attention weights  2. train/test performance vs swarm size, attrition, obstacle density, clutter index
# 3. ranking + classification metrics and confusion matrix  4. behaviour over time (performance across the mission,
# step-to-step consistency of the predicted ranking, true vs predicted criticality over time).
# Everything is saved in the folder ./exp4_in_depth_analysis_per_timestep/ (created automatically). Single cell: run the whole file at once.
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
SOFT_RANK_TEMPERATURE=0.1
LOGIT_ANCHOR=1e-3              # keeps pre-sigmoid scores from drifting into saturation (needed when BCE_WEIGHT = 0)
TIME_STRIDE=3                  # every 3rd sampled timestep is used for training and evaluation (each one is scored on its own)
LABEL_MODE="mean"              # only for the secondary agent-level metrics: true score = mean over alive steps
CRITICAL_LABEL_MODE="any"      # only for the secondary agent-level metrics: critical at any alive step
TOP_K=[1,5,10]
SHOW_PLOTS=True                # also display figures inline (they are always saved as PNG + PDF)
EXAMPLE_CASE_ID=None           # test swarm used in single-swarm plots; None = the swarm with median test Spearman
# The model is trained and evaluated PER TIMESTEP: targets are criticality_score and is_critical of each agent at each step.
EXPERIMENT_DIR="exp4_in_depth_analysis_per_timestep"
D=12
SEED=42
BCE_WEIGHT=0.5
HEATMAP_AGENTS=12              # agents shown in the criticality-over-time heatmap
############################ SHARED CORE (identical in all five experiment scripts) ############################
# PER-TIMESTEP VERSION: no temporal aggregation. The latent vector z_i(t) of every agent at every timestep goes straight
# into the head, giving a score s_i(t). The loss is the same as before, applied per timestep:
#   ranking:        1 - soft Spearman between s(t) and the true criticality ranking at t, over the agents alive at t,
#                   averaged over timesteps (plus the tiny logit anchor)
#   classification: class-weighted cross-entropy between s_i(t) and is_critical_i(t), over all agent-timesteps
# Ranking metrics are computed per timestep, averaged over the timesteps of a swarm, then over swarms.
# Classification metrics are pooled over all agent-timesteps.
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
    by_agent=g.groupby("agent_id")                       # agent-level labels: only for the secondary (time-averaged) metrics
    score=by_agent["criticality_score"].mean() if LABEL_MODE=="mean" else (by_agent["criticality_score"].first() if LABEL_MODE=="first" else by_agent["criticality_score"].last())
    crit=by_agent["is_critical_label"].max() if CRITICAL_LABEL_MODE=="any" else (by_agent["is_critical_label"].first() if CRITICAL_LABEL_MODE=="first" else by_agent["is_critical_label"].last())
    steps=np.sort(g["step"].unique())[::TIME_STRIDE]
    g=g[g["step"].isin(steps)]
    agents=np.sort(g["agent_id"].unique())
    T,N=len(steps),len(agents)
    idx=pd.MultiIndex.from_product([steps,agents],names=["step","agent_id"])
    p=g.set_index(["step","agent_id"]).reindex(idx)
    alive_np=(~p["case_id"].isna()).to_numpy().reshape(T,N)
    alive=torch.tensor(alive_np)
    x=((p[features].fillna(stats["median"])-stats["mean"])/stats["std"]).to_numpy(np.float32).reshape(T,N,len(features))
    pos=torch.tensor(np.nan_to_num(p[GRAPH_POSITION_COLUMNS].to_numpy(np.float32).reshape(T,N,3),nan=0.0))
    # per-timestep targets
    y_grid=p["criticality_score"].to_numpy(np.float32).reshape(T,N)
    crit_grid=p["is_critical_label"].to_numpy(np.float32).reshape(T,N)
    rank_grid=pd.DataFrame(np.where(alive_np,y_grid,np.nan)).rank(axis=1,method="average").to_numpy(np.float32)   # rank among agents alive at t
    n_alive=alive_np.sum(1)
    y_sd=np.nanstd(np.where(alive_np,y_grid,np.nan),axis=1)
    step_valid=(n_alive>=2)&(np.nan_to_num(y_sd)>0)        # a ranking exists at this step
    dist=torch.cdist(pos,pos)
    adj=(dist<=GRAPH_THRESHOLD)&alive.unsqueeze(2)&alive.unsqueeze(1)&~torch.eye(N,dtype=torch.bool).unsqueeze(0)
    flat=torch.full((T,N),-1,dtype=torch.long)
    flat[alive]=torch.arange(int(alive.sum()))
    et,es,ed=torch.nonzero(adj,as_tuple=True)
    step_obs=torch.arange(T).unsqueeze(1).expand(T,N)[alive]
    return {"case_id":case_id,"agent_ids":agents,"steps":steps,"x":torch.tensor(x)[alive],
            "edge_index":torch.stack([flat[et,es],flat[et,ed]]),"edge_attr":(dist[et,es,ed]/GRAPH_THRESHOLD).unsqueeze(1).float(),
            "agent_obs_index":torch.arange(N).unsqueeze(0).expand(T,N)[alive],"step_obs_index":step_obs,"t_norm":step_obs.float()/max(T-1,1),
            "alive":alive,"y_obs":torch.tensor(y_grid[alive_np]),"y_crit_obs":torch.tensor(crit_grid[alive_np]),
            "y_rank_grid":torch.tensor(np.nan_to_num(rank_grid)),"step_valid":torch.tensor(step_valid),
            "y":torch.tensor(score.reindex(agents).to_numpy(np.float32)),"y_crit":torch.tensor(crit.reindex(agents).to_numpy(np.float32)),
            "num_agents":N,"num_steps":T,"num_obs":int(alive.sum())}
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
class CriticalGINE(nn.Module):
    def __init__(self,input_dim,d,dropout):
        super().__init__()
        self.feature_attention=FeatureAttention(input_dim)
        self.gin1=GINELayer(input_dim,64,64,dropout)
        self.norm1=nn.LayerNorm(64)
        self.gin2=GINELayer(64,64,32,dropout)
        self.norm2=nn.LayerNorm(32)
        self.global_mix=nn.Linear(64,32)
        self.norm_global=nn.LayerNorm(32)
        self.gin3=GINELayer(32,32,d,dropout)
        self.head=nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Dropout(dropout),nn.Linear(32,1),nn.Sigmoid())
    def forward(self,c):
        x=self.feature_attention(c["x"])
        h=F.relu(self.norm1(self.gin1(x,c["edge_index"],c["edge_attr"])))
        h=F.relu(self.norm2(self.gin2(h,c["edge_index"],c["edge_attr"])))
        swarm=segment_mean(h,c["step_obs_index"],c["num_steps"])
        h=F.relu(self.norm_global(self.global_mix(torch.cat([h,swarm[c["step_obs_index"]]],1))))
        z=self.gin3(h,c["edge_index"],c["edge_attr"])
        self.last_logit=self.head[:-1](z).squeeze(1)            # one pre-sigmoid score per agent-timestep (no temporal pooling)
        return self.head[-1](self.last_logit),z
def move_case(c):
    return {k:(v.to(DEVICE) if torch.is_tensor(v) else v) for k,v in c.items()}
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

############################ LOSSES ############################
def per_step_soft_spearman_loss(logit,c):
    # 1 - soft Spearman at every timestep, over the agents alive at that step, averaged over the steps that have a ranking.
    # soft rank_i(t) = 0.5 + sum_j sigmoid((x_i - x_j)/tau) over alive j, with x the logits standardised within the step.
    alive=c["alive"].float()
    T,N=alive.shape
    grid=torch.zeros(T,N,device=logit.device)
    grid[c["alive"]]=logit
    n=alive.sum(1,keepdim=True).clamp_min(1)
    mean=(grid*alive).sum(1,keepdim=True)/n
    std=torch.sqrt((((grid-mean)*alive)**2).sum(1,keepdim=True)/(n-1).clamp_min(1))
    x=(grid-mean)/(std+1e-8)*alive
    soft=0.5+(torch.sigmoid((x.unsqueeze(2)-x.unsqueeze(1))/SOFT_RANK_TEMPERATURE)*alive.unsqueeze(1)).sum(2)
    def centred(v):
        return (v-(v*alive).sum(1,keepdim=True)/n)*alive
    a,b=centred(soft),centred(c["y_rank_grid"])
    corr=(a*b).sum(1)/(a.norm(dim=1)*b.norm(dim=1)+1e-8)
    valid=c["step_valid"]
    return (1.0-corr[valid]).mean() if bool(valid.any()) else logit.sum()*0.0
def ranking_loss(logit,c):
    # Soft Spearman is invariant to the scale and offset of the scores; ranking on the pre-sigmoid logit plus a tiny anchor
    # on its size stops the scores drifting into sigmoid saturation (tied scores, zero gradients).
    return per_step_soft_spearman_loss(logit,c)+LOGIT_ANCHOR*(logit**2).mean()
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
def step_groups(c):
    order=np.argsort(c["step_idx"],kind="stable")
    bounds=np.flatnonzero(np.diff(c["step_idx"][order]))+1
    return np.split(order,bounds)
def fast_step_spearman(c):
    # mean over timesteps of the Spearman correlation among the agents alive at that step (vectorised, used for validation)
    d=pd.DataFrame({"s":c["step_idx"],"y":c["y_obs"],"p":c["pred"]})
    g=d.groupby("s")
    d["ry"]=g["y"].rank(method="average")
    d["rp"]=g["p"].rank(method="average")
    d["ry"]-=d.groupby("s")["ry"].transform("mean")
    d["rp"]-=d.groupby("s")["rp"].transform("mean")
    agg=pd.DataFrame({"num":d["ry"]*d["rp"],"a":d["ry"]**2,"b":d["rp"]**2,"s":d["s"]}).groupby("s").sum()
    ok=(agg["a"]>0)&(agg["b"]>0)
    return float((agg.loc[ok,"num"]/np.sqrt(agg.loc[ok,"a"]*agg.loc[ok,"b"])).mean()) if ok.any() else np.nan
def case_ranking_metrics(c):
    # ranking metrics at every timestep (agents alive at that step), averaged over the timesteps of the swarm
    rows=[ranking_metrics(c["y_obs"][ix],c["pred"][ix]) for ix in step_groups(c) if len(ix)>=2 and np.std(c["y_obs"][ix])>0]
    return pd.DataFrame(rows).mean(skipna=True).to_dict() if rows else {}
def per_swarm_table(per_case,meta):
    rows=[]
    for c in per_case:
        both=0<c["y_crit_obs"].sum()<len(c["y_crit_obs"])
        both_agent=0<c["y_crit"].sum()<len(c["y_crit"])
        agent_ok=np.std(c["y"])>0 and np.std(c["pred_agent"])>0
        rows.append({"case_id":c["case_id"],"agents":len(c["y"]),"timesteps":len(c["steps"]),"critical_agents":int(c["y_crit"].sum()),
                     **case_ranking_metrics(c),"auprc":average_precision_score(c["y_crit_obs"],c["pred"]) if both else np.nan,
                     "agent_spearman":spearmanr(c["y"],c["pred_agent"])[0] if agent_ok else np.nan,
                     "agent_auprc":average_precision_score(c["y_crit"],c["pred_agent"]) if both_agent else np.nan})
    return pd.DataFrame(rows).merge(meta[["case_id","num_agents","attrition_clean","obstacle_density","clutter_index"]],on="case_id",how="left")
def pooled(per_case,key):
    return np.concatenate([c[key] for c in per_case])
def classification_metrics(per_case,threshold,label_key="y_crit_obs",pred_key="pred"):
    y,p=pooled(per_case,label_key),pooled(per_case,pred_key)
    yhat=(p>=threshold).astype(int)
    both=0<y.sum()<len(y)
    return {"threshold":threshold,"auprc":average_precision_score(y,p) if both else np.nan,"roc_auc":roc_auc_score(y,p) if both else np.nan,
            "prevalence":float(y.mean()),"accuracy":accuracy_score(y,yhat),"precision":precision_score(y,yhat,zero_division=0),
            "recall":recall_score(y,yhat,zero_division=0),"f1":f1_score(y,yhat,zero_division=0),
            "TP":int(((yhat==1)&(y==1)).sum()),"FP":int(((yhat==1)&(y==0)).sum()),"FN":int(((yhat==0)&(y==1)).sum()),"TN":int(((yhat==0)&(y==0)).sum())}
def best_f1_threshold(per_case):
    y,p=pooled(per_case,"y_crit_obs"),pooled(per_case,"pred")
    grid=np.linspace(0.01,0.99,99)
    return float(grid[int(np.argmax([f1_score(y,(p>=t).astype(int),zero_division=0) for t in grid]))])
def split_summary(per_case,meta,threshold):
    table=per_swarm_table(per_case,meta)
    out={}
    for col in ["spearman","kendall"]+[f"precision@{k}" for k in TOP_K]+[f"ndcg@{k}" for k in TOP_K]+["agent_spearman"]:
        out[f"{col}_mean"]=table[col].mean()
        out[f"{col}_std"]=table[col].std()
    out["per_swarm_auprc_mean"]=table["auprc"].mean()
    out.update(classification_metrics(per_case,threshold))                       # per agent-timestep
    agent=classification_metrics(per_case,threshold,"y_crit","pred_agent")       # secondary: agent level, scores averaged over time
    out["agent_auprc"]=agent["auprc"]
    out["agent_roc_auc"]=agent["roc_auc"]
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
            scores,_=model(c)
            pred=scores.cpu().numpy()
            agent_idx=raw["agent_obs_index"].numpy()
            pred_agent=np.bincount(agent_idx,weights=pred,minlength=raw["num_agents"])/np.bincount(agent_idx,minlength=raw["num_agents"]).clip(min=1)
            out.append({"case_id":raw["case_id"],"agent_ids":raw["agent_ids"],"steps":raw["steps"],"step_idx":raw["step_obs_index"].numpy(),
                        "agent_idx":agent_idx,"y_obs":raw["y_obs"].numpy(),"y_crit_obs":raw["y_crit_obs"].numpy(),"pred":pred,
                        "y":raw["y"].numpy(),"y_crit":raw["y_crit"].numpy(),"pred_agent":pred_agent})
    return out
def train_model(train_data,val_data,n_features,d,bce_weight,seed,label,checkpoint=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng=np.random.default_rng(seed)
    model=CriticalGINE(n_features,d,DROPOUT).to(DEVICE)
    saved=torch.load(checkpoint,map_location=DEVICE) if checkpoint is not None and os.path.exists(checkpoint) else None
    if saved is not None and (saved.get("d")!=d or len(saved.get("features",[]))!=n_features or saved.get("bce_weight")!=bce_weight):
        print(f"\n[{label}] checkpoint {checkpoint} was trained with d={saved.get('d')}, {len(saved.get('features',[]))} features, "
              f"BCE weight={saved.get('bce_weight')}; current run is d={d}, {n_features} features, BCE weight={bce_weight} - retraining")
        saved=None
    if saved is not None:                                               # reuse a finished run instead of retraining
        model.load_state_dict(saved["model_state_dict"])
        prev=pd.read_csv(os.path.join(os.path.dirname(checkpoint),"summary.csv")).iloc[0]
        print(f"\n[{label}] loaded finished run from {checkpoint} (best epoch {int(prev['best_epoch'])}) - not retrained")
        return model,{"best_epoch":int(prev["best_epoch"]),"best_val_spearman":float(prev["best_val_spearman"]),"training_minutes":float(prev["training_minutes"]),
                      "pos_weight":float(prev["pos_weight"]),"parameters":count_parameters(model)}
    optimizer=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    n_pos=sum(float(c["y_crit_obs"].sum()) for c in train_data)              # counted over training agent-timesteps
    pos_weight=sum(float((1-c["y_crit_obs"]).sum()) for c in train_data)/max(n_pos,1.0)
    best,best_state,best_epoch,waited=-np.inf,None,0,0
    start=time.time()
    print(f"\n[{label}] training (per timestep) | d={d} | features={n_features} | BCE weight={bce_weight} | pos_weight={pos_weight:.3f} | parameters={count_parameters(model)}")
    for epoch in range(1,EPOCHS+1):
        model.train()
        total,total_rank,total_bce,obs=0.0,0.0,0.0,0
        for k in rng.permutation(len(train_data)):
            c=move_case(train_data[k])
            optimizer.zero_grad()
            scores,_=model(c)
            rank_loss=ranking_loss(model.last_logit,c)
            bce_loss=weighted_bce(scores,c["y_crit_obs"],pos_weight)   # always computed and printed; enters the loss only if bce_weight > 0
            loss=rank_loss+bce_weight*bce_loss
            loss.backward()
            optimizer.step()
            n=c["num_obs"]
            total+=loss.item()*n
            total_rank+=rank_loss.item()*n
            total_bce+=bce_loss.item()*n
            obs+=n
        if epoch%VAL_EVERY!=0 and epoch!=EPOCHS:
            continue
        val_pred=predict(model,val_data)
        val_rho=np.nanmean([fast_step_spearman(c) for c in val_pred])
        val_auprc=classification_metrics(val_pred,0.5)["auprc"]
        if np.isfinite(val_rho) and val_rho>best:
            best,best_epoch,waited=val_rho,epoch,0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            waited+=VAL_EVERY
        print(f"[{label}] epoch {epoch:4d} | train loss {total/obs:.4f} = rank {total_rank/obs:.4f} + {bce_weight}×BCE {total_bce/obs:.4f} | "
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
    frames=[]
    for c in preds["test"]:                                  # one row per agent-timestep; ranks are within the timestep
        f=pd.DataFrame({"case_id":c["case_id"],"step":c["steps"][c["step_idx"]],"agent_id":c["agent_ids"][c["agent_idx"]],
                        "criticality_score":c["y_obs"],"predicted_score":c["pred"],"is_critical":c["y_crit_obs"].astype(int),
                        "predicted_critical":(c["pred"]>=threshold).astype(int)})
        f["true_rank"]=f.groupby("step")["criticality_score"].rank(ascending=False,method="min").astype(int)
        f["predicted_rank"]=f.groupby("step")["predicted_score"].rank(ascending=False,method="min").astype(int)
        frames.append(f)
    pd.concat(frames,ignore_index=True).to_csv(os.path.join(run_dir,"test_predictions.csv"),index=False)
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
def example_step(c):
    # the middle sampled timestep of the swarm (same step for every model, so comparisons are like for like)
    return int(c["steps"][len(c["steps"])//2])
def plot_rank_agreement(ax,run,case_id,title,step=None):
    c=next(c for c in run["preds"]["test"] if c["case_id"]==case_id)
    step=example_step(c) if step is None else step
    ix=c["step_idx"]==int(np.flatnonzero(c["steps"]==step)[0])
    y,pred,crit=c["y_obs"][ix],c["pred"][ix],c["y_crit_obs"][ix]>0.5
    true_rank=pd.Series(y).rank(ascending=False,method="average").to_numpy()
    pred_rank=pd.Series(pred).rank(ascending=False,method="average").to_numpy()
    n=len(true_rank)
    ax.plot([1,n],[1,n],color=MUTED,lw=1.2,ls=(0,(4,3)),zorder=1)
    ax.scatter(true_rank[~crit],pred_rank[~crit],s=46,color=BLUE,edgecolor=SURFACE,linewidth=1.2,zorder=3,label="Decoy")
    ax.scatter(true_rank[crit],pred_rank[crit],s=70,color=ORANGE,edgecolor=SURFACE,linewidth=1.2,zorder=4,label="Critical asset")
    rho=spearmanr(y,pred)[0] if np.std(y)>0 and np.std(pred)>0 else np.nan
    ax.set_xlim(0,n+1)
    ax.set_ylim(0,n+1)
    ax.set_aspect("equal")
    ax.set_xlabel("True rank at this timestep (1 = most critical)")
    ax.set_ylabel("Predicted rank")
    ax.set_title(title)
    subtitle(ax,f"{case_id}  ·  step {step}  ·  {n} agents alive  ·  Spearman ρ = {rho:.3f}")
    ax.text(n*0.97,n*0.06,"perfect ranking on the dashed line",ha="right",fontsize=10,color=MUTED)
    ax.legend(loc="upper left")

############################ EXPERIMENT 4: IN-DEPTH ANALYSIS (per-timestep model, relative features + BCE) ############################
df=load_dataset(DATA_PATH)
meta,train_ids,val_ids,test_ids=split_cases(df)
split_ids=(train_ids,val_ids,test_ids)
print(f"Cases: train {len(train_ids)} | validation {len(val_ids)} | test {len(test_ids)}")
run=run_configuration(df,meta,split_ids,D,True,BCE_WEIGHT,SEED,os.path.join(EXPERIMENT_DIR,"model"),"in-depth")
model,features=run["model"],run["data"]["features"]

############################ 4.1 FEATURE ATTENTION WEIGHTS ############################
with torch.inference_mode():
    beta=model.feature_attention.weights().cpu().numpy()
fw=pd.DataFrame({"feature":features,"group":[feature_group(f) for f in features],"attention_weight":beta,"relative_to_uniform":beta*len(features)})
fw=fw.sort_values("attention_weight",ascending=False).reset_index(drop=True)
fw.insert(0,"rank",np.arange(1,len(fw)+1))
fw.to_csv(os.path.join(EXPERIMENT_DIR,"feature_attention_weights.csv"),index=False)
print("\nFEATURE ATTENTION WEIGHTS (uniform = 1/F =",f"{1/len(features):.4f}; relative_to_uniform > 1 = up-weighted)")
print(fw.to_string(index=False,float_format=lambda v:f"{v:.4f}"))
fig,ax=plt.subplots(figsize=(9,0.32*len(fw)+1.8))
plot_df=fw.iloc[::-1]
ax.barh(plot_df["feature"],plot_df["relative_to_uniform"]-1,left=1,color=[GROUP_COLORS[g] for g in plot_df["group"]],height=0.72,zorder=3)
ax.axvline(1,color=INK2,lw=1.2,zorder=4)
ax.text(1,len(fw)-0.4,"  uniform weight",color=INK2,fontsize=10,va="bottom")
span=max(abs(fw["relative_to_uniform"]-1).max(),0.005)
for i,v in enumerate(plot_df["relative_to_uniform"]):
    ax.text(v+(span*0.03 if v>=1 else -span*0.03),i,f"{v:.3f}×",va="center",ha="left" if v>=1 else "right",fontsize=9,color=INK2)
ax.set_xlim(1-span*1.35,1+span*1.35)
ax.set_xlabel("Learned feature weight relative to uniform (×)")
ax.grid(axis="y",visible=False)
ax.set_title("Which inputs the model relies on")
subtitle(ax,"Global feature attention β (shared by all agents, timesteps and swarms) · bars grow from the uniform weight")
ax.legend(handles=[Patch(color=c,label=g) for g,c in GROUP_COLORS.items()],loc="upper center",bbox_to_anchor=(0.5,-0.06),ncol=3)
save_figure(fig,"feature_attention_weights")

############################ 4.2 PERFORMANCE vs SCENARIO CONDITIONS (train and test) ############################
swarms=pd.concat([run["tables"]["train"].assign(split="train"),run["tables"]["test"].assign(split="test")],ignore_index=True)
swarms.to_csv(os.path.join(EXPERIMENT_DIR,"per_swarm_metrics_train_test.csv"),index=False)
ATTRITION_ORDER=["none","light","medium","heavy","high"]
def categorical_condition(column,label,order=None):
    cats=[c for c in (order or sorted(swarms[column].unique())) if c in set(swarms[column])]
    tab=swarms.groupby(["split",column]).agg(swarms=("case_id","count"),spearman_mean=("spearman","mean"),spearman_std=("spearman","std"),
                                             auprc_mean=("auprc","mean"),auprc_std=("auprc","std")).reset_index()
    tab.to_csv(os.path.join(EXPERIMENT_DIR,f"condition_{column}.csv"),index=False)
    print(f"\nPERFORMANCE BY {label.upper()}")
    print(tab.to_string(index=False,float_format=lambda v:f"{v:.4f}"))
    fig,axes=plt.subplots(1,2,figsize=(13,5.2))
    x=np.arange(len(cats))
    for ax,metric,ylabel in zip(axes,["spearman","auprc"],["Spearman ρ (mean over swarms)","AUPRC (mean over swarms)"]):
        for xi,c in zip(x,cats):
            present=[sp for sp in ["train","test"] if ((tab["split"]==sp)&(tab[column]==c)).any()]
            for j,split in enumerate(present):
                row=tab[(tab["split"]==split)&(tab[column]==c)].iloc[0]
                offset=0.0 if len(present)==1 else (j-0.5)*0.38
                ax.bar(xi+offset,row[f"{metric}_mean"],width=0.36,yerr=np.nan_to_num(row[f"{metric}_std"]),color=SPLIT_COLORS[split],zorder=3,
                       error_kw=dict(ecolor=INK2,elinewidth=1,capsize=3))
                ax.text(xi+offset,row[f"{metric}_mean"]+np.nan_to_num(row[f"{metric}_std"])+0.02,f"{row[f'{metric}_mean']:.2f}",ha="center",fontsize=10,color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in cats])
        ax.set_xlabel(label)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0,1.1)
        ax.grid(axis="x",visible=False)
    axes[0].set_title(f"Ranking quality by {label.lower()}")
    axes[1].set_title(f"Critical-asset detection by {label.lower()}")
    subtitle(axes[0],"bars = mean over swarms, whiskers = ± 1 sd")
    subtitle(axes[1],"conditions held out of training appear in the test split only (OOD split)")
    axes[1].legend(handles=[Patch(color=SPLIT_COLORS[s],label=s.capitalize()) for s in ["train","test"]],loc="upper right")
    save_figure(fig,f"performance_vs_{column}")
def continuous_condition(column,label):
    bins=pd.qcut(swarms[column],q=min(4,swarms[column].nunique()),duplicates="drop")
    tab=swarms.assign(bin=bins.astype(str)).groupby(["split","bin"]).agg(swarms=("case_id","count"),spearman_mean=("spearman","mean"),
                                                                        auprc_mean=("auprc","mean")).reset_index()
    tab.to_csv(os.path.join(EXPERIMENT_DIR,f"condition_{column}.csv"),index=False)
    print(f"\nPERFORMANCE BY {label.upper()} (quartile bins over all train and test swarms)")
    print(tab.to_string(index=False,float_format=lambda v:f"{v:.4f}"))
    fig,axes=plt.subplots(1,2,figsize=(13,5.2))
    for ax,metric,ylabel in zip(axes,["spearman","auprc"],["Spearman ρ (per swarm)","AUPRC (per swarm)"]):
        for split in ["train","test"]:
            sub=swarms[swarms["split"]==split].dropna(subset=[metric])
            ax.scatter(sub[column],sub[metric],s=60,color=SPLIT_COLORS[split],edgecolor=SURFACE,linewidth=1.2,label=split.capitalize(),zorder=3,alpha=0.9)
            if len(sub)>=3 and sub[column].nunique()>1:
                coef=np.polyfit(sub[column],sub[metric],1)
                xs=np.linspace(sub[column].min(),sub[column].max(),50)
                ax.plot(xs,np.polyval(coef,xs),color=SPLIT_COLORS[split],lw=1.6,ls=(0,(5,3)),zorder=2)
        ax.set_xlabel(label)
        ax.set_ylabel(ylabel)
        lo=np.nanmin(swarms[metric])
        ax.set_ylim(max(0,lo-0.08),1.03)
    axes[0].set_title(f"Ranking quality vs {label.lower()}")
    axes[1].set_title(f"Critical-asset detection vs {label.lower()}")
    subtitle(axes[0],"each dot is one swarm · dashed = linear trend per split")
    subtitle(axes[1],"each dot is one swarm · dashed = linear trend per split")
    axes[1].legend(loc="lower right")
    save_figure(fig,f"performance_vs_{column}")
categorical_condition("num_agents","Swarm size (agents)")
categorical_condition("attrition_clean","Attrition level",ATTRITION_ORDER)
continuous_condition("obstacle_density","Obstacle density")
continuous_condition("clutter_index","Clutter index")

############################ 4.3 RANKING + CLASSIFICATION METRICS AND CONFUSION MATRIX ############################
metric_rows=[("Spearman ρ (per timestep, mean over swarms)","spearman_mean"),("Kendall τ (per timestep, mean over swarms)","kendall_mean")]
metric_rows+=[(f"Precision@{k} (per timestep, mean over swarms)",f"precision@{k}_mean") for k in TOP_K]+[(f"NDCG@{k} (per timestep, mean over swarms)",f"ndcg@{k}_mean") for k in TOP_K]
metric_rows+=[("AUPRC (pooled agent-timesteps)","auprc"),("AUPRC (mean over swarms)","per_swarm_auprc_mean"),("ROC-AUC (pooled agent-timesteps)","roc_auc"),("Accuracy","accuracy"),
              ("Precision","precision"),("Recall","recall"),("F1-score","f1"),("Share of critical agent-timesteps (random AUPRC)","prevalence"),
              ("Agent-level Spearman ρ (scores averaged over time)","agent_spearman_mean"),("Agent-level AUPRC (scores averaged over time)","agent_auprc")]
s=run["summary"]
metrics_table=pd.DataFrame({"metric":[m[0] for m in metric_rows],**{split:[s[f"{split}_{m[1]}"] for m in metric_rows] for split in ["train","val","test"]}})
metrics_table.to_csv(os.path.join(EXPERIMENT_DIR,"metrics_table.csv"),index=False)
print(f"\nRANKING AND CLASSIFICATION METRICS (threshold θ = {run['threshold']:.2f}, fitted on validation by best F1)")
print(metrics_table.to_string(index=False,float_format=lambda v:f"{v:.4f}"))
conf=pd.DataFrame([[s["test_TP"],s["test_FN"]],[s["test_FP"],s["test_TN"]]],index=["Truly critical","Truly decoy"],columns=["Predicted critical","Predicted decoy"])
conf.to_csv(os.path.join(EXPERIMENT_DIR,"confusion_matrix_test.csv"))
print("\nCONFUSION MATRIX (test, counts are agent-timesteps)")
print(conf.to_string())
fig,ax=plt.subplots(figsize=(6.6,5.6))
vals=conf.to_numpy().astype(float)
row_share=vals/vals.sum(1,keepdims=True).clip(min=1)
ax.imshow(row_share,cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blue",["#f4f8fd","#86b6ef","#1c5cab"]),vmin=0,vmax=1)
for i in range(2):
    for j in range(2):
        ax.text(j,i,f"{int(vals[i,j])}\n{row_share[i,j]*100:.1f}%",ha="center",va="center",fontsize=15,fontweight="semibold",
                color=SURFACE if row_share[i,j]>0.55 else INK)
ax.set_xticks([0,1])
ax.set_xticklabels(conf.columns)
ax.set_yticks([0,1])
ax.set_yticklabels(conf.index)
ax.grid(False)
ax.axhline(0.5,color=SURFACE,lw=4)
ax.axvline(0.5,color=SURFACE,lw=4)
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_title("Confusion matrix · test set")
subtitle(ax,f"agent-timesteps · threshold θ = {run['threshold']:.2f} (best F1 on validation) · % of each true class")
save_figure(fig,"confusion_matrix_test")

############################ 4.4 BEHAVIOUR OVER TIME (no temporal aggregation) ############################
# (a) ranking and detection quality across the mission, (b) step-to-step consistency of the predicted ranking
# compared with the true ranking, (c) true vs predicted criticality over time for one test swarm.
def step_frame(per_case,split):
    frames=[]
    for c in per_case:
        T=len(c["steps"])
        f=pd.DataFrame({"case_id":c["case_id"],"split":split,"step_idx":c["step_idx"],"agent_idx":c["agent_idx"],"agent_id":c["agent_ids"][c["agent_idx"]],
                        "y":c["y_obs"],"crit":c["y_crit_obs"],"pred":c["pred"]})
        f["mission_time"]=f["step_idx"]/max(T-1,1)
        frames.append(f)
    return pd.concat(frames,ignore_index=True)
obs=pd.concat([step_frame(run["preds"]["train"],"train"),step_frame(run["preds"]["test"],"test")],ignore_index=True)
obs["time_bin"]=np.minimum((obs["mission_time"]*10).astype(int),9)
step_rows=[]
for (cid,split,si),g in obs.groupby(["case_id","split","step_idx"]):
    ok=len(g)>=2 and g["y"].std()>0 and g["pred"].std()>0
    both=0<g["crit"].sum()<len(g)
    step_rows.append({"case_id":cid,"split":split,"step_idx":si,"time_bin":int(g["time_bin"].iloc[0]),"agents_alive":len(g),
                      "spearman":spearmanr(g["y"],g["pred"])[0] if ok else np.nan,"auprc":average_precision_score(g["crit"],g["pred"]) if both else np.nan})
steps_df=pd.DataFrame(step_rows)
steps_df.to_csv(os.path.join(EXPERIMENT_DIR,"per_timestep_metrics_train_test.csv"),index=False)
over_time=steps_df.groupby(["split","time_bin"]).agg(spearman_mean=("spearman","mean"),spearman_sem=("spearman","sem"),
                                                     auprc_mean=("auprc","mean"),auprc_sem=("auprc","sem"),timesteps=("step_idx","count")).reset_index()
over_time.to_csv(os.path.join(EXPERIMENT_DIR,"performance_over_mission_time.csv"),index=False)
print("\nPERFORMANCE OVER MISSION TIME (tenths of the mission; mean over timesteps of all swarms in the split)")
print(over_time.to_string(index=False,float_format=lambda v:f"{v:.4f}"))
fig,axes=plt.subplots(1,2,figsize=(13,5.2))
for ax,metric,ylabel in zip(axes,["spearman","auprc"],["Spearman ρ at each timestep","AUPRC at each timestep"]):
    for split in ["train","test"]:
        prof=over_time[over_time["split"]==split].set_index("time_bin").reindex(range(10)).dropna(subset=[f"{metric}_mean"])
        centers=(prof.index.to_numpy()+0.5)/10
        ax.fill_between(centers,prof[f"{metric}_mean"]-prof[f"{metric}_sem"],prof[f"{metric}_mean"]+prof[f"{metric}_sem"],color=SPLIT_COLORS[split],alpha=0.15,lw=0)
        ax.plot(centers,prof[f"{metric}_mean"],color=SPLIT_COLORS[split],marker="o",markeredgecolor=SURFACE,markeredgewidth=1.2,label=split.capitalize(),zorder=3)
    ax.set_xlim(0,1)
    ax.set_xlabel("Mission time t / (T − 1)")
    ax.set_ylabel(ylabel)
axes[0].set_title("Ranking quality across the mission")
axes[1].set_title("Critical-asset detection across the mission")
subtitle(axes[0],"mean ± standard error per tenth of the mission")
subtitle(axes[1],"mean ± standard error per tenth of the mission")
axes[1].legend(loc="lower right")
save_figure(fig,"performance_over_mission_time")

# (b) temporal consistency: Spearman between the ranking at consecutive sampled timesteps (agents alive at both)
cons_rows=[]
for (cid,split),g in obs.groupby(["case_id","split"]):
    grid_y=g.pivot_table(index="step_idx",columns="agent_idx",values="y")
    grid_p=g.pivot_table(index="step_idx",columns="agent_idx",values="pred")
    for a,b in zip(grid_y.index[:-1],grid_y.index[1:]):
        m=grid_y.loc[a].notna()&grid_y.loc[b].notna()
        if m.sum()<3:
            continue
        ya,yb,pa,pb=grid_y.loc[a][m],grid_y.loc[b][m],grid_p.loc[a][m],grid_p.loc[b][m]
        cons_rows.append({"case_id":cid,"split":split,"step_idx":a,
                          "true_rank_consistency":spearmanr(ya,yb)[0] if ya.std()>0 and yb.std()>0 else np.nan,
                          "predicted_rank_consistency":spearmanr(pa,pb)[0] if pa.std()>0 and pb.std()>0 else np.nan,
                          "mean_abs_change_true":float(np.abs(yb-ya).mean()),"mean_abs_change_pred":float(np.abs(pb-pa).mean())})
cons=pd.DataFrame(cons_rows)
cons.to_csv(os.path.join(EXPERIMENT_DIR,"temporal_consistency_steps.csv"),index=False)
cons_summary=cons.groupby("split").agg(step_pairs=("step_idx","count"),true_rank_consistency=("true_rank_consistency","mean"),
                                       predicted_rank_consistency=("predicted_rank_consistency","mean"),
                                       mean_abs_change_true=("mean_abs_change_true","mean"),mean_abs_change_pred=("mean_abs_change_pred","mean")).reset_index()
cons_summary.to_csv(os.path.join(EXPERIMENT_DIR,"temporal_consistency_summary.csv"),index=False)
print("\nTEMPORAL CONSISTENCY (Spearman between rankings at consecutive sampled timesteps; |change| of the score per step)")
print(cons_summary.to_string(index=False,float_format=lambda v:f"{v:.4f}"))
test_cons=cons[cons["split"]=="test"]
fig,ax=plt.subplots(figsize=(8.5,5))
lo=np.nanmin(test_cons[["true_rank_consistency","predicted_rank_consistency"]].to_numpy())
bins=np.linspace(max(-1.0,np.floor((lo-0.02)*20)/20),1.0,41)
ax.hist(test_cons["true_rank_consistency"].dropna(),bins=bins,color=MUTED,alpha=0.55,label="True ranking",zorder=2)
ax.hist(test_cons["predicted_rank_consistency"].dropna(),bins=bins,color=ORANGE,alpha=0.75,label="Predicted ranking",zorder=3)
for col,color in [("true_rank_consistency",INK2),("predicted_rank_consistency",ORANGE)]:
    ax.axvline(test_cons[col].mean(),color=color,lw=1.6,ls=(0,(4,3)))
ax.set_xlabel("Spearman ρ between the rankings at consecutive timesteps")
ax.set_ylabel("Number of step pairs (test)")
ax.set_title("How stable is the ranking from one timestep to the next?")
subtitle(ax,f"test set · dashed = mean (true {test_cons['true_rank_consistency'].mean():.3f}, predicted {test_cons['predicted_rank_consistency'].mean():.3f})")
ax.legend(loc="upper left")
save_figure(fig,"temporal_consistency_test")

# (c) true vs predicted criticality over time for the agents of one test swarm
case_id=pick_example_case(run)
g=obs[(obs["case_id"]==case_id)&(obs["split"]=="test")]
order=g.groupby("agent_id")["pred"].mean().sort_values(ascending=False).index[:HEATMAP_AGENTS]
crit_ids=set(g.loc[g["crit"]>0.5,"agent_id"])
c_ex=next(c for c in run["preds"]["test"] if c["case_id"]==case_id)
step_labels=c_ex["steps"]
fig,axes=plt.subplots(1,2,figsize=(14,0.42*len(order)+2.4),sharey=True)
cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blue",["#f4f8fd","#86b6ef","#2a78d6","#104281"]).copy()
cmap.set_bad("#efeeea")
for ax,col,title in zip(axes,["y","pred"],["True criticality score","Predicted score"]):
    grid=g.pivot_table(index="agent_id",columns="step_idx",values=col).reindex(order).reindex(columns=range(len(step_labels)))
    im=ax.imshow(np.ma.masked_invalid(grid.to_numpy()),aspect="auto",cmap=cmap,interpolation="nearest",
                 vmin=np.nanmin(g[col]),vmax=np.nanmax(g[col]))
    ticks=np.linspace(0,len(step_labels)-1,min(6,len(step_labels))).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(step_labels[t]) for t in ticks])
    ax.set_xlabel("Simulation step  (grey = agent no longer alive)")
    ax.grid(False)
    ax.set_title(title)
    cb=fig.colorbar(im,ax=ax,fraction=0.04,pad=0.02)
    cb.outline.set_visible(False)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels([f"agent {a}{'  ★' if a in crit_ids else ''}" for a in order])
subtitle(axes[0],f"test swarm {case_id} · top-{HEATMAP_AGENTS} agents by mean predicted score · ★ = critical asset")
save_figure(fig,"criticality_over_time_example")
