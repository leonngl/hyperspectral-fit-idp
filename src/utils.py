import pickle
import numpy as np
import os


# return score and config of best result
def get_best_values_from_res_grid(res_grid, metric="sq_avg_error"):
    best_result = res_grid.get_best_result(scope="avg")
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