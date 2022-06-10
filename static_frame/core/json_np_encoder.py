import json
import numpy as np


class NpDtypeMapper(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return obj.tolist()
        raise NotImplementedError(f"{type(obj)} mappings are not yet supported.")
