from collections import defaultdict
import pandas as pd

def summarize_model_metadata(models_metadata, model_sizes):
    # 2. create a df
    tmp = pd.DataFrame.from_dict(models_metadata, orient='index')
    tmp = tmp.reset_index().rename(columns={'index': 'model'})[['model','size']]
    tmp.loc[:,'model'] = tmp.model.apply(lambda x:x.replace('-q4_K_M',''))

    # Group models by size
    grouped = defaultdict(list)

    for size in model_sizes:
        grouped[size] = tmp.loc[tmp["size"] == size, "model"].tolist()

    # Pad columns to equal length
    max_len = max(len(grouped[s]) for s in model_sizes)

    for s in model_sizes:
        grouped[s] += [""] * (max_len - len(grouped[s]))

    # Create reshaped dataframe
    latex_df = pd.DataFrame({s: grouped[s] for s in model_sizes})

    # Escape underscores for LaTeX
    latex_df = latex_df.map(
        lambda x: x.replace("_", r"\_") if isinstance(x, str) else x
    )

    return latex_df