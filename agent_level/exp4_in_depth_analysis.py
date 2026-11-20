############################ EXP4_IN_DEPTH_ANALYSIS ############################
# In-depth analysis of the final model (relative features + BCE, d = 16):
# 1. feature attention weights  2. train/test performance vs swarm size, attrition, obstacle density, clutter index
# 3. ranking + classification metrics and confusion matrix  4. temporal attention analysis.
# Everything is saved in the folder ./exp4_in_depth_analysis/ (created automatically). Single cell: run the whole file at once.
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
EXPERIMENT_DIR="exp4_in_depth_analysis"
D=16
SEED=42
BCE_WEIGHT=0.5
HEATMAP_AGENTS=12              # agents shown in the temporal-attention heatmap
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

############################ EXPERIMENT 4: IN-DEPTH ANALYSIS (relative features + BCE) ############################
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
metric_rows=[("Spearman ρ (mean over swarms)","spearman_mean"),("Kendall τ (mean over swarms)","kendall_mean")]
metric_rows+=[(f"Precision@{k} (mean over swarms)",f"precision@{k}_mean") for k in TOP_K]+[(f"NDCG@{k} (mean over swarms)",f"ndcg@{k}_mean") for k in TOP_K]
metric_rows+=[("AUPRC (pooled)","auprc"),("AUPRC (mean over swarms)","per_swarm_auprc_mean"),("ROC-AUC (pooled)","roc_auc"),("Accuracy","accuracy"),
              ("Precision","precision"),("Recall","recall"),("F1-score","f1"),("Share of critical agents (random AUPRC)","prevalence")]
s=run["summary"]
metrics_table=pd.DataFrame({"metric":[m[0] for m in metric_rows],**{split:[s[f"{split}_{m[1]}"] for m in metric_rows] for split in ["train","val","test"]}})
metrics_table.to_csv(os.path.join(EXPERIMENT_DIR,"metrics_table.csv"),index=False)
print(f"\nRANKING AND CLASSIFICATION METRICS (threshold θ = {run['threshold']:.2f}, fitted on validation by best F1)")
print(metrics_table.to_string(index=False,float_format=lambda v:f"{v:.4f}"))
conf=pd.DataFrame([[s["test_TP"],s["test_FN"]],[s["test_FP"],s["test_TN"]]],index=["Truly critical","Truly decoy"],columns=["Predicted critical","Predicted decoy"])
conf.to_csv(os.path.join(EXPERIMENT_DIR,"confusion_matrix_test.csv"))
print("\nCONFUSION MATRIX (test)")
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
subtitle(ax,f"threshold θ = {run['threshold']:.2f} (best F1 on validation) · % of each true class")
save_figure(fig,"confusion_matrix_test")

############################ 4.4 TEMPORAL ATTENTION ANALYSIS ############################
model.eval()
att_rows=[]
with torch.inference_mode():
    for raw in run["data"]["test"]:
        c=move_case(raw)
        scores,_,_,alpha=model(c)
        alpha=alpha.cpu().numpy()
        agent_idx=raw["agent_obs_index"].numpy()
        alive_steps=raw["obs_per_agent"].numpy()
        pred=scores.cpu().numpy()
        pred_rank=pd.Series(pred).rank(ascending=False,method="min").to_numpy()
        for o in range(len(alpha)):
            i=agent_idx[o]
            att_rows.append({"case_id":raw["case_id"],"agent_id":raw["agent_ids"][i],"step":raw["steps"][raw["step_obs_index"][o]],
                             "mission_time":float(raw["t_norm"][o]),"alpha":alpha[o],"alive_steps":int(alive_steps[i]),
                             "relative_attention":alpha[o]*alive_steps[i],"attrited":bool(alive_steps[i]<raw["num_steps"]),
                             "critical":bool(raw["y_crit"][i]>0.5),"predicted_rank":int(pred_rank[i])})
att=pd.DataFrame(att_rows)
att.to_csv(os.path.join(EXPERIMENT_DIR,"temporal_attention_test.csv"),index=False)
ent=att.groupby(["case_id","agent_id"]).agg(alive_steps=("alive_steps","first"),attrited=("attrited","first"),critical=("critical","first"),
                                           entropy=("alpha",lambda a:-(a*np.log(a.clip(1e-12))).sum()),peak=("relative_attention","max")).reset_index()
ent["normalized_entropy"]=np.where(ent["alive_steps"]>1,ent["entropy"]/np.log(ent["alive_steps"].clip(lower=2)),1.0)
summary_att=pd.concat([ent.assign(group="all agents"),ent[ent["critical"]].assign(group="critical"),ent[~ent["critical"]].assign(group="decoy"),
                       ent[ent["attrited"]].assign(group="attrited"),ent[~ent["attrited"]].assign(group="survived")]).groupby("group").agg(
    agents=("agent_id","count"),normalized_entropy=("normalized_entropy","mean"),peak_relative_attention=("peak","mean")).reset_index()
summary_att.to_csv(os.path.join(EXPERIMENT_DIR,"temporal_attention_summary.csv"),index=False)
print("\nTEMPORAL ATTENTION SUMMARY (normalized entropy 1 = uniform over alive steps, lower = more focused; peak = max α·alive_steps)")
print(summary_att.to_string(index=False,float_format=lambda v:f"{v:.4f}"))
att["time_bin"]=np.minimum((att["mission_time"]*10).astype(int),9)
def attention_profile(column,labels,colors,title,name):
    fig,ax=plt.subplots(figsize=(9,5))
    centers=(np.arange(10)+0.5)/10
    for value,label,color in zip([True,False],labels,colors):
        prof=att[att[column]==value].groupby("time_bin")["relative_attention"].agg(["mean","sem"]).reindex(range(10))
        if prof["mean"].notna().sum()==0:
            continue
        ax.fill_between(centers,prof["mean"]-prof["sem"],prof["mean"]+prof["sem"],color=color,alpha=0.15,lw=0)
        ax.plot(centers,prof["mean"],color=color,marker="o",markeredgecolor=SURFACE,markeredgewidth=1.2,label=label,zorder=3)
    ax.axhline(1,color=INK2,lw=1.1,ls=(0,(4,3)))
    ax.text(0.995,1,"uniform (mean pooling)",ha="right",va="bottom",fontsize=10,color=INK2,transform=ax.get_yaxis_transform())
    ax.set_xlim(0,1)
    ax.set_xlabel("Mission time t / (T − 1)")
    ax.set_ylabel("Attention relative to uniform  (α · alive steps)")
    ax.set_title(title)
    subtitle(ax,"test set · mean ± standard error per tenth of the mission · above 1 = emphasised")
    ax.legend(loc="best")
    save_figure(fig,name)
attention_profile("attrited",["Attrited agents","Agents that survived"],[ORANGE,BLUE],"When does the model look? Attrited vs surviving agents","temporal_attention_attrited_vs_survived")
attention_profile("critical",["Critical assets","Decoys"],[ORANGE,BLUE],"When does the model look? Critical assets vs decoys","temporal_attention_critical_vs_decoy")
case_id=pick_example_case(run)
top=att[(att["case_id"]==case_id)&(att["predicted_rank"]<=HEATMAP_AGENTS)]
grid=top.pivot_table(index=["predicted_rank","agent_id"],columns="step",values="alpha").sort_index()
crit_ids=set(top.loc[top["critical"],"agent_id"])
fig,ax=plt.subplots(figsize=(12,0.42*len(grid)+2.2))
cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blue",["#f4f8fd","#86b6ef","#2a78d6","#104281"]).copy()
cmap.set_bad("#efeeea")
im=ax.imshow(np.ma.masked_invalid(grid.to_numpy()),aspect="auto",cmap=cmap,interpolation="nearest")
ax.set_yticks(range(len(grid)))
ax.set_yticklabels([f"#{r}  agent {a}{'  ★ critical' if a in crit_ids else ''}" for r,a in grid.index])
steps=grid.columns.to_numpy()
ticks=np.linspace(0,len(steps)-1,min(8,len(steps))).astype(int)
ax.set_xticks(ticks)
ax.set_xticklabels([str(steps[t]) for t in ticks])
ax.set_xlabel("Simulation step  (grey = agent no longer alive)")
ax.grid(False)
cb=fig.colorbar(im,ax=ax,fraction=0.025,pad=0.02)
cb.set_label("attention weight α",color=INK2)
cb.outline.set_visible(False)
ax.set_title(f"Temporal attention of the top-{HEATMAP_AGENTS} predicted agents")
subtitle(ax,f"test swarm {case_id} · rows ordered by predicted rank · ★ = true critical asset")
save_figure(fig,"temporal_attention_heatmap")
