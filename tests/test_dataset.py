import unittest
from sentrix.dataset import split_groups, validate


class DatasetTests(unittest.TestCase):
    def test_split_keeps_scene_and_session_together(self):
        rows=[{"clip_id":"a","scene_id":"x","session_id":"s1"},{"clip_id":"b","scene_id":"x","session_id":"s2"},
              {"clip_id":"c","scene_id":"y","session_id":"s3"},{"clip_id":"d","scene_id":"z","session_id":"s4"},
              {"clip_id":"e","scene_id":"q","session_id":"s5"}]
        split=split_groups(rows,seed=2)
        self.assertEqual(split["a"],split["b"])
        self.assertEqual(set(split.values()),{"train","development","test"})

    def test_validate_requires_consent_and_split_integrity(self):
        manifest=[{"clip_id":"a","scene_id":"s","session_id":"r","consent_verified":"true","split":"train","duration_seconds":"10"},
                  {"clip_id":"b","scene_id":"s","session_id":"q","consent_verified":"true","split":"test","duration_seconds":"10"}]
        errors=validate(manifest,[])
        self.assertTrue(any("leaks across splits" in e for e in errors))


if __name__=="__main__":unittest.main()
