import pickle
import numpy as np
import os


# return score and config of best result
def get_best_values_from_res_grid(res_grid, metric="sq_avg_error"):
    try:
        best_result = res_grid.get_best_result(scope="avg")
    except Exception:
        print("Could not get best_result from res_grid!")
        return np.inf, None
    best_config = best_result.config
    config_arr = np.array([best_config[str(i)] for i in range(len(best_config))])
    score = best_result.metrics_dataframe[metric].mean()
    return score, config_arr



def compare_and_update_config(score, config, checkpoint_path, overwrite_regardless=False):
    if not os.path.exists(checkpoint_path):
        prev_score = None
    else:
        with open(checkpoint_path, "rb") as f:
            prev_score, prev_config = pickle.load(f)
        
    if (score is not None) and (overwrite_regardless or prev_score is None or score < prev_score):
        with open(checkpoint_path, "wb") as f:
            pickle.dump((score, config), f)
        print("Updated parameters. Using new ones.")
    elif score is None and prev_score is None:
        raise RuntimeError("Current and previous scores are None.")
    else:
        score, config = prev_score, prev_config
        print("Keeping old parameters.")
    
    return score, config

# taken from https://jwalton.info/Embed-Publication-Matplotlib-Latex/
def set_size(width, height=None, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if height is not None:
        fig_height_in = height
    else:
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)



def print_header(path):
    with open(path, "r") as f:
        print('\n'.join([f"{i}: {name}" for (i, name) in enumerate(f.readline().split())]))