# series_recorder.py
import json, time

class SeriesRecorder:
    """
    SeriesRecorder
    --------------
    Utility class for recording joint motion sequences into a JSON Lines (.jsonl) file.
    Each line corresponds to one frame, containing:
        - timestamp (t)
        - joints: dict {joint_name: [x, y, z]}
        - style: optional style embedding / vector
        - conf:  optional confidence values for each joint
    This format is convenient for training ML models (streamable, line-by-line).
    """

    def __init__(self, path="series_001.jsonl"):
        """
        Initialize a recorder.
        Args:
            path (str): output file path (default: series_001.jsonl)
        """
        self.path = path
        self.f = None

    def start(self):
        """
        Open the file for writing (overwrite mode).
        """
        self.f = open(self.path, "w", encoding="utf-8")

    def push(self, joints, style=None, conf=None):
        """
        Append one frame of data.
        Args:
            joints (dict): {joint_name: [x, y, z]}
            style (optional): style embedding or label (list/array-like)
            conf (optional): confidence dict {joint_name: float}
        """
        if not self.f:
            return
        rec = {"t": time.time(), "joints": joints}
        if style is not None:
            rec["style"] = list(style)
        if conf is not None:
            rec["conf"] = conf
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def stop(self):
        """
        Close the file handle if it is open.
        """
        if self.f:
            self.f.close()
            self.f = None
