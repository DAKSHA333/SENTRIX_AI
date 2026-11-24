"""Optional GRU experiment over reviewed trajectory features; requires tensorflow."""
import argparse
import json
import os
import pandas as pd

FEATURES=["x_norm","y_norm","box_w_norm","box_h_norm","speed_px_s","heading_rad","dwell_seconds","nearest_bag_distance_norm","pose_visible_keypoints","pose_arms_raised"]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("features_csv");parser.add_argument("--out",default="reports/gru_metrics.json");parser.add_argument("--window",type=int,default=16);parser.add_argument("--stride",type=int,default=8);parser.add_argument("--epochs",type=int,default=30)
    args=parser.parse_args()
    try:import tensorflow as tf
    except ImportError as exc:raise SystemExit("Install a compatible TensorFlow build to run this optional experiment.") from exc
    data=pd.read_csv(args.features_csv)
    required={"clip_id","scene_id","session_id","track_id","timestamp_seconds","label","split"}
    missing=required-set(data.columns)
    if missing:raise SystemExit("Feature CSV is missing required reviewed-label columns: {}".format(sorted(missing)))
    for col in FEATURES:
        if col not in data:data[col]=0.0
    data[FEATURES]=data[FEATURES].apply(pd.to_numeric,errors="coerce").fillna(0.0)
    classes=sorted(data.label.dropna().astype(str).unique())
    if len(classes)<2:raise SystemExit("Need at least two reviewed event labels for a sequence classifier.")
    label_to_id={name:i for i,name in enumerate(classes)}
    xs,ys,splits=[],[],[]
    for (clip,track),group in data.sort_values("timestamp_seconds").groupby(["clip_id","track_id"]):
        labels=group.label.dropna().astype(str).unique()
        if len(labels)!=1 or labels[0] not in label_to_id:continue
        arr=group[FEATURES].to_numpy(dtype="float32")
        split=str(group.split.iloc[0]).lower()
        for start in range(0,max(0,len(arr)-args.window+1),args.stride):
            xs.append(arr[start:start+args.window]);ys.append(label_to_id[labels[0]]);splits.append(split)
    import numpy as np
    X=np.asarray(xs,dtype="float32");y=np.asarray(ys,dtype="int32");splits=np.asarray(splits)
    train=splits=="train";dev=np.isin(splits,["development","dev"]);test=splits=="test"
    if train.sum()<20 or test.sum()<1:raise SystemExit("Insufficient independent labeled sequences; need >=20 train windows and >=1 held-out test window. Do not train on frame splits.")
    mean=X[train].mean(axis=(0,1),keepdims=True);std=X[train].std(axis=(0,1),keepdims=True)+1e-6;X=(X-mean)/std
    model=tf.keras.Sequential([tf.keras.layers.Input(shape=(args.window,len(FEATURES))),tf.keras.layers.GRU(32),tf.keras.layers.Dropout(.2),tf.keras.layers.Dense(len(classes),activation="softmax")])
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss" if dev.any() else "loss",patience=5,restore_best_weights=True)]
    model.fit(X[train],y[train],validation_data=(X[dev],y[dev]) if dev.any() else None,epochs=args.epochs,batch_size=32,callbacks=callbacks,verbose=2)
    loss,accuracy=model.evaluate(X[test],y[test],verbose=0)
    probs=model.predict(X[test],verbose=0);pred=probs.argmax(axis=1)
    per_class={}
    for ident,name in enumerate(classes):
        tp=int(((y[test]==ident)&(pred==ident)).sum());fp=int(((y[test]!=ident)&(pred==ident)).sum());fn=int(((y[test]==ident)&(pred!=ident)).sum())
        precision=tp/(tp+fp) if tp+fp else 0.0;recall=tp/(tp+fn) if tp+fn else 0.0
        per_class[name]={"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0.0}
    report={"classes":classes,"train_windows":int(train.sum()),"dev_windows":int(dev.sum()),"test_windows":int(test.sum()),"test_loss":float(loss),"test_accuracy":float(accuracy),"test_per_class":per_class,"feature_normalization":"train split only","grouping":"clip/session-derived windows; no cross-split frame overlap","caution":"Optional ablation only; this does not replace interpretable rules or establish deployment performance."}
    os.makedirs(os.path.dirname(args.out) or ".",exist_ok=True)
    model.save(os.path.splitext(args.out)[0]+".keras")
    with open(args.out,"w",encoding="utf-8") as handle:json.dump(report,handle,indent=2)
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
