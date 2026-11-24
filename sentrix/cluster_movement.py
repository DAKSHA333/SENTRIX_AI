"""Optional unsupervised exploration of anonymous movement features."""
import argparse
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURES=["x_norm","y_norm","box_w_norm","box_h_norm","speed_px_s","heading_rad","dwell_seconds","nearest_bag_distance_norm","crowd_density_per_mpix","local_flow_x","local_flow_y"]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("features_csv");p.add_argument("--clusters",type=int,default=5);p.add_argument("--out",default="reports/movement_clusters.csv");p.add_argument("--split",default="development");a=p.parse_args()
    data=pd.read_csv(a.features_csv);subset=data[data["split"].astype(str).str.lower()==a.split.lower()].copy()
    missing=[x for x in FEATURES if x not in subset.columns]
    if missing:raise SystemExit("Missing feature columns: {}".format(missing))
    if len(subset)<max(10,a.clusters):raise SystemExit("Not enough development rows for this clustering run.")
    values=subset[FEATURES].apply(pd.to_numeric,errors="coerce").fillna(0.0)
    subset["movement_cluster"]=KMeans(n_clusters=a.clusters,random_state=42,n_init=10).fit_predict(StandardScaler().fit_transform(values))
    subset["cluster_use_note"]="Exploratory motion grouping only; cluster IDs do not indicate risk or intent."
    os.makedirs(os.path.dirname(a.out) or ".",exist_ok=True);subset.to_csv(a.out,index=False)
    print("Saved {} exploratory track samples to {}".format(len(subset),a.out))
if __name__=="__main__":main()
