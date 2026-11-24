"""Simple frame-difference baseline for the proposal's motion-only comparison."""
import cv2


class MotionBaseline:
    def __init__(self, settings, fps):
        self.settings, self.fps = settings, max(1.0,float(fps))
        self.previous = None
        self.last_alert = -1e9

    def update(self, frame, frame_index, current_time=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(9,9),0)
        if self.previous is None:
            self.previous=gray
            return []
        delta=cv2.absdiff(self.previous,gray);self.previous=gray
        _,mask=cv2.threshold(delta,self.settings.motion_pixel_threshold,255,cv2.THRESH_BINARY)
        mask=cv2.dilate(mask,None,iterations=2)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        area=sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c)>100)
        fraction=area/(frame.shape[0]*frame.shape[1])
        now=current_time if current_time is not None else frame_index/self.fps
        if fraction>=self.settings.motion_area_fraction and now-self.last_alert>=self.settings.motion_cooldown_seconds:
            self.last_alert=now
            return [{"event_type":"motion_alarm","timestamp_seconds":round(now,2),"zone":None,"track_id":None,
                "evidence_quality":"SUFFICIENT","reason_code":"FRAME_DIFFERENCE_AREA_THRESHOLD",
                "reason":"Frame-to-frame pixel change exceeded the baseline threshold; event meaning is unknown.",
                "measurements":{"changed_area_fraction":round(fraction,5)}}]
        return []
