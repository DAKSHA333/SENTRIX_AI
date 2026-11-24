"""Append scene-local derived trajectory features for analysis/optional GRU."""
import csv
import os
from math import atan2, hypot
import cv2
import numpy as np


class FeatureLogger:
    FIELDS=["clip_id","scene_id","session_id","split","frame_index","timestamp_seconds","track_id","class_name","x_norm","y_norm","box_w_norm","box_h_norm","speed_px_s","heading_rad","dwell_seconds","nearest_bag_distance_norm","zone_names","crowd_density_per_mpix","local_flow_x","local_flow_y","pose_visible_keypoints","pose_arms_raised","label"]
    def __init__(self,path,clip_id,fps,scene_id="",session_id="",split=""):
        self.path,self.clip_id,self.fps=path,clip_id,max(float(fps),1.0)
        self.scene_id,self.session_id,self.split=scene_id,session_id,split
        os.makedirs(os.path.dirname(path) or ".",mode=0o700,exist_ok=True)
        self.handle=open(path,"a",newline="",encoding="utf-8")
        try:os.chmod(path,0o600);os.chmod(os.path.dirname(path) or ".",0o700)
        except OSError:pass
        self.writer=csv.DictWriter(self.handle,fieldnames=self.FIELDS)
        if os.path.getsize(path)==0:self.writer.writeheader()
        self.previous={};self.first_seen={}
    def update(self,tracks,frame_index,shape,bag_classes,pose_features=None,zones=(),current_time=None):
        h,w=shape[:2];now=current_time if current_time is not None else frame_index/self.fps
        bags=[t for t in tracks if t.label in bag_classes]
        people=[t for t in tracks if t.label=="person"]
        velocities=[]
        for person in people:
            prior=self.previous.get(person.id)
            if prior:
                dx=(person.box[0]+person.box[2])/2-prior[1];dy=(person.box[1]+person.box[3])/2-prior[2]
                norm=hypot(dx,dy)
                if norm>0:velocities.append((dx/norm,dy/norm))
        flow_x=sum(v[0] for v in velocities)/len(velocities) if velocities else 0.0
        flow_y=sum(v[1] for v in velocities)/len(velocities) if velocities else 0.0
        for track in tracks:
            if track.label!="person":continue
            x1,y1,x2,y2=track.box;cx=(x1+x2)/2;cy=(y1+y2)/2
            prior=self.previous.get(track.id);speed=0.0;heading=0.0
            if prior:
                dt=max(1e-3,now-prior[0]);dx=cx-prior[1];dy=cy-prior[2]
                speed=hypot(dx,dy)/dt;heading=atan2(dy,dx)
            self.previous[track.id]=(now,cx,cy);self.first_seen.setdefault(track.id,now)
            nearest=min((hypot(cx-(b.box[0]+b.box[2])/2,cy-(b.box[1]+b.box[3])/2) for b in bags),default=None)
            in_zones=[]
            for zone in zones:
                if len(zone.polygon)>=3 and cv2.pointPolygonTest(np.array(zone.polygon,dtype="int32"),(int(cx),int(cy)),False)>=0:in_zones.append(zone.name)
            self.writer.writerow({"clip_id":self.clip_id,"scene_id":self.scene_id,"session_id":self.session_id,"split":self.split,"frame_index":frame_index,"timestamp_seconds":round(now,4),"track_id":track.id,"class_name":"person",
                "x_norm":round(cx/w,6),"y_norm":round(cy/h,6),"box_w_norm":round((x2-x1)/w,6),"box_h_norm":round((y2-y1)/h,6),
                "speed_px_s":round(speed,3),"heading_rad":round(heading,4),"dwell_seconds":round(now-self.first_seen[track.id],3),
                "nearest_bag_distance_norm":round(nearest/hypot(w,h),6) if nearest is not None else "",
                "zone_names":";".join(in_zones),"crowd_density_per_mpix":round(len(people)/(w*h/1e6),4),"local_flow_x":round(flow_x,4),"local_flow_y":round(flow_y,4),
                "pose_visible_keypoints":(pose_features or {}).get(track.id,{}).get("visible_keypoints",""),
                "pose_arms_raised":(pose_features or {}).get(track.id,{}).get("arms_raised",""),"label":""})
        self.handle.flush()
    def close(self):self.handle.close()
