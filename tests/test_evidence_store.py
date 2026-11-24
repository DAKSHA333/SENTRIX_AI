import os
import tempfile
import unittest
import cv2
import numpy as np
from sentrix.evidence import EvidenceRecorder
from sentrix.store import AlertStore


class EvidenceStoreTests(unittest.TestCase):
    def test_evidence_clip_is_written_and_person_region_is_blurred(self):
        with tempfile.TemporaryDirectory() as root:
            recorder=EvidenceRecorder(os.path.join(root,"clips"),10,0.2,0.2,True)
            frame=np.zeros((120,120,3),dtype=np.uint8)
            for y in range(10,50,4):
                for x in range(20,80,4):frame[y:y+2,x:x+2]=255
            blurred=recorder._blur(frame.copy(),[(10,5,90,110)])
            self.assertFalse(np.array_equal(frame,blurred))
            for _ in range(2):recorder.add_frame(frame,[(10,5,90,110)])
            recorder.trigger(12)
            results=[]
            for _ in range(3):results.extend(recorder.add_frame(frame,[(10,5,90,110)]))
            self.assertEqual(results[0][0],12)
            self.assertTrue(os.path.isfile(results[0][1]))
            recorder.close()

    def test_reviewer_action_is_audited(self):
        with tempfile.TemporaryDirectory() as root:
            store=AlertStore(os.path.join(root,"alerts.db"))
            ident=store.add("clip1",{"event_type":"restricted_zone_entry","timestamp_seconds":1})
            store.decide(ident,"confirmed",actor="reviewer",note="staged event")
            store.rate_explanation(ident,4,actor="reviewer")
            self.assertEqual(store.list()[0]["decision"],"confirmed")
            self.assertEqual(store.list()[0]["explanation_rating"],4)
            self.assertEqual(len(store.audit_rows()),2)
            store.close()


if __name__=="__main__":unittest.main()
