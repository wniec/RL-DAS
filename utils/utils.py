import numpy as np


def mean_info(info, key):
    # Extract values from the info list
    values = [x['info'].get()[key] for x in info]

    if isinstance(values[0], (list, np.ndarray)):
        max_len = max(len(v) for v in values)

        padded_values = []
        for v in values:
            pad_width = max_len - len(v)

            # mode='edge' repeats the last valid value for the duration of the padding
            if pad_width > 0:
                padded_v = np.pad(v, (0, pad_width), mode='edge')
                padded_values.append(padded_v)
            else:
                padded_values.append(v)

        return np.mean(padded_values, axis=0)
    else:
        # Handle scalar values (like 'FEs' or 'best_cost') normally
        return np.mean(values)