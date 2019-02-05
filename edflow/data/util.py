import numpy as np


def is_image(v):
    if isinstance(v, np.ndarray) \
            and len(v.shape) == 3 \
            and (v.shape[-1] == 3 or v.shape[-1] == 4):
        return True
    return False


def is_points(v):
    if isinstance(v, np.ndarray) \
            and v.shape[-1] == 2:
        return True
    elif isinstance(v, list):
        if np.array(v).dtype != object and np.array(v).shape[-1] == 2:
            return True
    return False


def is_seq(v):
    if isinstance(v, list):
        if np.array(v).dtype != object:
            return True
    elif isinstance(v, np.ndarray) \
            and len(v.shape) > 2:
        return True
    return False


def plot_datum(datum, title='example datum', savepath='datum.pdf'):
    '''Makes a plot of the data contained in one datum.
    
    Args:
        datum (dict): Containing all the data.
    '''

    n_content = len(datum)

    plotable_content = []
    for k, v in datum.items():
        kind = 'im' if is_image(v) else 'seq-' if is_seq(v) else \
                'points' if is_points(v) else 'other'

        if kind == 'seq':
            s1 = v[0]
            kind += 'im' if is_image(s1) else 'points' is_points(s1) \
                    else 'other'

        plotable_content += [k, kind, v]

    plotable_content = sorted(plotable_content, key=lambda n, k, v: k, n)

    for name, kind, value in plotable_content:
        pass
