import unittest
from datetime import datetime
from zoneinfo import ZoneInfo
from sentrix.config import Settings, Zone
from sentrix.events import EventEngine
from sentrix.tracking import Track


class EventTests(unittest.TestCase):
    def test_restricted_entry_is_explained(self):
        settings=Settings(restricted_zones=[Zone("door",[(0,0),(100,0),(100,100),(0,100)])],low_quality_min_person_height=20,zone_entry_confirm_samples=1)
        engine=EventEngine(settings,25)
        track=Track(1,(20,20,60,90),"person",.9,1)
        engine.update([Track(1,(120,20,160,90),"person",.9,0)],0,(100,100,3))
        alerts=engine.update([track],1,(100,100,3))
        self.assertEqual(alerts[0]["event_type"],"restricted_zone_entry")
        self.assertEqual(alerts[0]["reason_code"],"PERSON_TRACK_CROSSED_CONFIGURED_BOUNDARY")

    def test_object_class_alert_does_not_reference_unset_quality(self):
        engine=EventEngine(Settings(),25)
        track=Track(4,(20,20,50,60),"scissors",.8,1)
        alerts=engine.update([track],1,(100,100,3))
        self.assertEqual(alerts[0]["event_type"],"potentially_hazardous_object_detection")
        self.assertEqual(alerts[0]["measurements"]["object_label"],"scissors")

    def test_low_quality_candidate_is_withheld(self):
        settings=Settings(restricted_zones=[Zone("door",[(0,0),(100,0),(100,100),(0,100)])],low_quality_min_person_height=200,zone_entry_confirm_samples=1)
        engine=EventEngine(settings,25)
        engine.update([Track(1,(120,20,160,90),"person",.9,0)],0,(100,100,3))
        alerts=engine.update([Track(1,(20,20,60,90),"person",.9,1)],1,(100,100,3))
        self.assertEqual(alerts[0]["event_type"],"insufficient_evidence")

    def test_configured_schedule_requires_timestamp_for_recorded_video(self):
        zone=Zone("scheduled",[(0,0),(100,0),(100,100),(0,100)],active_hours=["09:00-17:00"])
        settings=Settings(restricted_zones=[zone],low_quality_min_person_height=20,zone_entry_confirm_samples=1)
        track=Track(1,(20,20,60,90),"person",.9,1)
        scheduled_engine=EventEngine(settings,25)
        scheduled_engine.update([Track(1,(120,20,160,90),"person",.9,0)],0,(100,100,3))
        withheld=scheduled_engine.update([track],1,(100,100,3))
        self.assertEqual(withheld[0]["event_type"],"insufficient_evidence")
        active_engine=EventEngine(settings,25)
        active_engine.update([Track(1,(120,20,160,90),"person",.9,0)],0,(100,100,3))
        inside=active_engine.update([track],1,(100,100,3),wall_clock_time=datetime(2026,11,23,10,0,tzinfo=ZoneInfo("Asia/Kolkata")))
        self.assertEqual(inside[0]["event_type"],"restricted_zone_entry")

    def test_existing_occupant_is_not_reported_as_crossing(self):
        settings=Settings(restricted_zones=[Zone("door",[(0,0),(100,0),(100,100),(0,100)])])
        track=Track(1,(20,20,60,90),"person",.9,1)
        alerts=EventEngine(settings,25).update([track],0,(100,100,3))
        self.assertEqual(alerts,[])

    def test_optional_dwell_pacing_counterflow_and_dispersion(self):
        zone=Zone("waiting",[(0,0),(100,0),(100,100),(0,100)],alert_on_entry=False,enable_dwell=True)
        settings=Settings(restricted_zones=[zone],enable_dwell_pacing=True,dwell_seconds=1,low_quality_min_person_height=10)
        engine=EventEngine(settings,1);track=Track(1,(20,20,40,80),"person",.9,1)
        engine.update([track],0,(100,100,3));events=engine.update([track],2,(100,100,3))
        self.assertIn("prolonged_zone_dwell",[e["event_type"] for e in events])

        pacing=EventEngine(Settings(enable_dwell_pacing=True,pacing_reversals=2,pacing_window_seconds=30,low_quality_min_person_height=10),1)
        ids=Track(2,(0,20,20,80),"person",.9,1)
        found=[]
        for frame,x in enumerate([0,40,0,40,0]):
            ids.box=(x,20,x+20,80);found+=pacing.update([ids],frame,(100,100,3))
        self.assertIn("repeated_path_reversals",[e["event_type"] for e in found])

        settings=Settings(enable_counterflow=True,counterflow_min_crowd_tracks=3,counterflow_min_speed_px_s=1,counterflow_sustain_seconds=0,low_quality_min_person_height=10)
        flow=EventEngine(settings,1);crowd=[Track(i,(x,20,x+20,80),"person",.9,0) for i,x in enumerate([10,40,70],1)]
        flow.update(crowd,0,(100,100,3))
        for track,dx in zip(crowd,[5,5,-5]):track.box=(track.box[0]+dx,20,track.box[2]+dx,80)
        found=flow.update(crowd,1,(100,100,3))
        self.assertIn("sustained_counter_flow",[e["event_type"] for e in found])

        settings=Settings(enable_dispersion=True,dispersion_min_people=4,dispersion_speed_px_s=1,dispersion_sustain_seconds=0)
        spread=EventEngine(settings,1);crowd=[Track(i,(x,20,x+10,40),"person",.9,0) for i,x in enumerate([20,40,60,80],1)]
        spread.update(crowd,0,(100,100,3))
        for track,dx in zip(crowd,[-3,-1,1,3]):track.box=(track.box[0]+dx,20,track.box[2]+dx,40)
        found=spread.update(crowd,1,(100,100,3))
        self.assertIn("sudden_group_dispersion",[e["event_type"] for e in found])


if __name__=="__main__":unittest.main()
