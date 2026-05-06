import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.stats.proportion import proportion_confint

from tqdm.auto import tqdm
tqdm.pandas()

from libs import constants

import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.stats.proportion import proportion_confint


def aggregate_per_attempt(df, group_cols, metric_agg):

    def pct(x):
        return x["n_counts"] / x["n_total"]
    
    def pct_inverse(x):
        return 1 - x["n_counts"] / x["n_total"]
    
    if type(metric_agg) == tuple:
        per_attempt = (
            df
            .groupby(group_cols, as_index=False)
            .agg(
                metric = metric_agg
            )
            .reset_index(drop=True)
        )
    else:
        k = [k for k in metric_agg.keys() if k != 'ntotal']

        if len(k) == 1:
            k = k[0]
            per_attempt = (
                df
                .groupby(group_cols, as_index=False)
                .agg(
                    n_total = metric_agg['ntotal'],
                    n_counts = metric_agg[k]
                )
                .assign(metric=lambda x: pct(x) if k in ['npositive'] else pct_inverse(x))
                .reset_index(drop=True)
            )
        else:
            raise ValueError(f"Metric {metric_agg} not supported")

    return per_attempt

def aggregate_per_group(per_attempt, main_col_group, alpha=0.05, metric_value_col='metric_value', metric_name_col='metric_name'):

    per_group = (
        per_attempt
        .groupby(main_col_group)
        .agg(
            n      =(metric_value_col, "count"),
            mean   =(metric_value_col, "mean"),
            std    =(metric_value_col, "std"),
            median =(metric_value_col, "median"),
            sum    =(metric_value_col, "sum"),
        )
    )

    # Initialize
    per_group["ci"] = np.nan
    per_group["ci_low"] = np.nan
    per_group["ci_high"] = np.nan

    per_group.reset_index(inplace=True)
    for metric_name, df in per_group.groupby(metric_name_col):
        is_bernoulli = metric_name in constants.BENCHMARK_BINARY_METRICS
        index = df.index

        if is_bernoulli:
            # For bernoulli distributions, use the Wilson score interval (i.e., refusal and validity)
            per_group.loc[index, "ci_low"], per_group.loc[index, "ci_high"] = proportion_confint(count=per_group.loc[index, "sum"],
                                                                                                 nobs=per_group.loc[index, "n"],
                                                                                                 method="wilson")
            per_group.loc[index, "ci"] = (per_group.loc[index, "ci_high"] - per_group.loc[index, "ci_low"]) / 2.0 # half-width of the confidence interval (just for visualization purposes)
                                                                                    
        else:
            # calculate confidence interval (means margin of error, t-distribution for non-bernoulli distributions)
            per_group.loc[index, "ci"] = (t.ppf(1 - alpha/2, per_group.loc[index, "n"] - 1) * per_group.loc[index, "std"] / np.sqrt(per_group.loc[index, "n"]))
            per_group.loc[index, "ci_low"] = per_group.loc[index, "mean"] - per_group.loc[index, "ci"]
            per_group.loc[index, "ci_high"] = per_group.loc[index, "mean"] + per_group.loc[index, "ci"]

    return per_group





def aggregate_refusal(df_summary):
    metric_agg  = {'ntotal':("is_refusal", "size"), 'npositive':("is_refusal", lambda s: (s == constants.REFUSAL_TRUE).sum())}
    per_attempt = aggregate_per_attempt(df_summary, constants.BENCHMARK_PER_ATTEMPT_COLS, metric_agg)
    return per_attempt


def aggregate_validity(df_summary):              
    metric_agg  = ("valid_attempt", "any")
    per_attempt = aggregate_per_attempt(df_summary, constants.BENCHMARK_PER_REQUEST_COLS, metric_agg)
    per_attempt.metric = per_attempt.metric.astype(float)
    return per_attempt

def aggregate_duplicates(df_factuality_author):
    metric_agg  = {'ntotal':("clean_name", "size"), 'nunique':("clean_name", "nunique")}
    per_attempt = aggregate_per_attempt(df_factuality_author, constants.BENCHMARK_PER_ATTEMPT_COLS, metric_agg)
    return per_attempt

def aggregate_consistency(df_factuality_author):
    def jaccard(a: set, b: set) -> float:
        u = a | b
        return (len(a & b) / len(u)) if len(u) else 1.0

    def set_transition_stability(name_sets: pd.Series) -> float:
        sets = list(name_sets)
        if len(sets) <= 1:
            return np.nan
        vals = [jaccard(sets[i], sets[i-1]) for i in range(1, len(sets))]
        return float(np.mean(vals))
        
    # remove duplicates per attempt
    df_factuality_author_clean = df_factuality_author.drop_duplicates(subset=['model_access','model_size','model_class','model','grounded','temperature','date','time','task_name','task_param','task_attempt','clean_name']).copy()
    df_factuality_author_clean = df_factuality_author_clean.query("@pd.notna(clean_name) and @pd.notnull(clean_name) and clean_name != ''")

    # sort by date and time
    group_cols = ['model_access','model_size','model_class',"model", 'grounded','temperature', "task_name", "task_param", "date", "time"]
    df_factuality_author_clean = df_factuality_author_clean.sort_values(group_cols)

    # name_set per timestamp within each group
    per_time = (
        df_factuality_author_clean.groupby(group_cols, sort=True)["clean_name"]
        .agg(lambda s: set(s.dropna().astype(str)))
        .reset_index(name="name_set")
    )

    # final
    group_cols = ['model_access','model_size','model_class',"model",'grounded', 'temperature', "task_name", "task_param"]
    metric_agg  = ("name_set",set_transition_stability)
    per_attempt = aggregate_per_attempt(per_time, group_cols, metric_agg)
    per_attempt.metric = per_attempt.metric.astype(float)
    return per_attempt











def aggregate_factuality_author(df_factuality_author):
    df_factuality_author_clean = df_factuality_author.drop_duplicates(subset=['model','grounded','temperature','date','time','task_name','task_param','task_attempt','clean_name']).copy()
    df_factuality_author_clean.author_exists = df_factuality_author_clean.author_exists.astype(int)
    metric_agg  = ("author_exists", "mean") # how many authors are found in the ground truth
    per_attempt = aggregate_per_attempt(df_factuality_author_clean, constants.BENCHMARK_PER_ATTEMPT_COLS, metric_agg)
    return per_attempt

def aggregate_factuality_task(df_factuality_task, metric):
    fact_column = constants.BENCHMARK_FACTUALITY_FIELD_METRICS_MAP[metric]
    df_factuality_task_clean = df_factuality_task.drop_duplicates(subset=['model','grounded','temperature','date','time','task_name','task_param','task_attempt','clean_name']).copy()
    df_factuality_task_clean.dropna(subset=[fact_column], inplace=True) # eg. in field, some authors do not have a list of topics associated with them so we cannot check fact.
    # print(df_factuality_task_clean[fact_column].value_counts())
    df_factuality_task_clean[fact_column] = df_factuality_task_clean[fact_column].astype(int)
    metric_agg  = (fact_column, "mean") # how many factual field-author / epoch-author / seniority-author checks are true
    per_attempt = aggregate_per_attempt(df_factuality_task_clean, constants.BENCHMARK_PER_ATTEMPT_COLS, metric_agg)
    return per_attempt








def get_number_of_categories(attribute):
    if attribute not in constants.BENCHMARK_DEMOGRAPHIC_ATTRIBUTES:
        raise ValueError(f"attribute must be one of {constants.BENCHMARK_DEMOGRAPHIC_ATTRIBUTES}")

    if attribute == 'gender':
        return len(constants.GENDER_LIST) - 1 # -1 for unknown
    elif attribute == 'ethnicity':
        return len(constants.ETHNICITY_LIST) - 1 # -1 for unknown
    
    return len(constants.PROMINENCE_CATEGORIES)
    

def aggregate_diversity(df_factuality_author, attribute=None):
    
    if attribute not in constants.BENCHMARK_DEMOGRAPHIC_ATTRIBUTES:
        raise ValueError(f"attribute must be one of {constants.BENCHMARK_DEMOGRAPHIC_ATTRIBUTES}")


    def normalized_shannon_entropy(counts):
        """
        Compute normalized Shannon entropy.

        Parameters
        ----------
        counts : array-like
            Counts per label for *known* labels only (zeros allowed).
        K : int
            Total number of possible labels in the taxonomy
            (including labels not observed in this response).

        Returns
        -------
        float
            Normalized Shannon entropy in [0, 1].
            Returns np.nan if no known items are present.
        """

        labels_without_unknown = counts.drop(constants.UNKNOWN_STR, errors='ignore')
        labels_without_unknown = labels_without_unknown[~labels_without_unknown.index.isna()]
        value_counts = labels_without_unknown.value_counts(normalize=False)
        
        K = get_number_of_categories(attribute)

        counts_without_unknown = value_counts.to_numpy(dtype=float) #np.asarray(counts_without_unknown, dtype=float)
        N = counts_without_unknown.sum()

        if N == 0:
            return np.nan

        p = counts_without_unknown / N
        p = p[p > 0]          # drop zero-probability labels

        H = -np.sum(p * np.log(p))
        ne = H / np.log(K)

        return ne

    df_factuality_author_clean = df_factuality_author.drop_duplicates(subset=['model','grounded','temperature','date','time','task_name','task_param','task_attempt','clean_name']).copy()
    df_factuality_author_clean = df_factuality_author_clean.query("@pd.notna(id_author_oa) and @pd.notnull(id_author_oa) and @pd.notna(@attribute) and @pd.notnull(@attribute)")
    
    per_attempt = (
        df_factuality_author_clean
        .groupby(constants.BENCHMARK_PER_ATTEMPT_COLS)[attribute]
        .apply(normalized_shannon_entropy)
        .reset_index(name="metric")
    )

    return per_attempt

def aggregate_diversity_gender(df_factuality_author):
    return aggregate_diversity(df_factuality_author, attribute='gender')

def aggregate_diversity_ethnicity(df_factuality_author):
    return aggregate_diversity(df_factuality_author, attribute='ethnicity')

def aggregate_diversity_prominence_pub(df_factuality_author):
    return aggregate_diversity(df_factuality_author, attribute='prominence_pub')

def aggregate_diversity_prominence_cit(df_factuality_author):
    return aggregate_diversity(df_factuality_author, attribute='prominence_cit')






def aggregate_parity(df_factuality_author, attribute=None, **kwargs):
    
    if attribute not in constants.BENCHMARK_DEMOGRAPHIC_ATTRIBUTES:
        raise ValueError(f"attribute must be one of {constants.BENCHMARK_DEMOGRAPHIC_ATTRIBUTES}")

    df_gt = kwargs.get('gt', None)
    if df_gt is None:
        raise ValueError("gt must be provided")

    def parity(series):
        # compares the fraction of each group from the results against the fraction of each group from the ground truth
        # this is 1 - TV (the total variation distance)
        p = series.value_counts(normalize=True)
        p_gt = df_gt[attribute].value_counts(normalize=True)

        # add missing categories to the results
        for cat in p_gt.index:
            if cat not in p.index:
                p[cat] = 0.0
                
        tv = (1/2 * sum(abs(p - p_gt)))
        parity = 1 - tv

        return parity
    
    # remove duplicates per attempt
    df_factuality_author_clean = df_factuality_author.drop_duplicates(subset=['model_access','model_size','model_class','model','grounded','temperature','date','time','task_name','task_param','task_attempt','clean_name']).copy()
    metric_agg  = (attribute, parity)
    per_attempt = aggregate_per_attempt(df_factuality_author_clean, constants.BENCHMARK_PER_ATTEMPT_COLS, metric_agg)

    return per_attempt

def aggregate_parity_gender(df_factuality_author, **kwargs):
    return aggregate_parity(df_factuality_author, attribute='gender', **kwargs)

def aggregate_parity_ethnicity(df_factuality_author, **kwargs):
    return aggregate_parity(df_factuality_author, attribute='ethnicity', **kwargs)

def aggregate_parity_prominence_pub(df_factuality_author, **kwargs):
    return aggregate_parity(df_factuality_author, attribute='prominence_pub', **kwargs)

def aggregate_parity_prominence_cit(df_factuality_author, **kwargs):
    return aggregate_parity(df_factuality_author, attribute='prominence_cit', **kwargs) 



def aggregate_similarity(df_factuality_author, **kwargs):
    df_similarity = kwargs.get('df_similarity', None)
    metric_similarity = kwargs.get('metric_similarity', None)

    if df_similarity is None or metric_similarity is None:
        raise ValueError("df_similarity and metric_similarity must be provided")

    group_cols = constants.BENCHMARK_PER_ATTEMPT_COLS + [metric_similarity]
    df = df_similarity[group_cols].copy()
    df.rename(columns={metric_similarity: 'metric'}, inplace=True)
    return df
