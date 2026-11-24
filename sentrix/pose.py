"""Optional scene-local pose cues; features describe posture only, never intent."""
from .tracking import iou


def extract_pose_features(model, frame, people, confidence=0.35):
    if not people:
        return {}
    result=model.predict(frame,conf=confidence,verbose=False)[0]
    output={}
    if result.boxes is None or result.keypoints is None:
        return output
    for person in people:
        best,best_score=None,0.0
        for index,box in enumerate(result.boxes):
            candidate=tuple(int(v) for v in box.xyxy[0].tolist())
            score=iou(person.box,candidate)
            if score>best_score:best,best_score=index,score
        if best is None or best_score<0.1:continue
        points=result.keypoints.xy[best].cpu().tolist()
        confs=result.keypoints.conf[best].cpu().tolist() if result.keypoints.conf is not None else [1.0]*len(points)
        visible=sum(1 for value in confs if value>=0.35);raised=0
        # COCO-17 shoulder indices 5/6 and wrist indices 9/10.
        for wrist,shoulder in ((9,5),(10,6)):
            if len(points)>max(wrist,shoulder) and confs[wrist]>=0.35 and confs[shoulder]>=0.35 and points[wrist][1]<points[shoulder][1]:raised+=1
        output[person.id]={"visible_keypoints":visible,"arms_raised":raised}
    return output
