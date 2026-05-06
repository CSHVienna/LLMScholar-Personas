
from matplotlib import pyplot as plt
from libs.visuals.grid import PanelSpec
from libs.constants import *

# Define your panel order and labels
PANELS_METRICS_GROUPS = [
    PanelSpec("refusal_pct", r"Refusals", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("validity_pct", r"Validity$\uparrow$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("duplicates", r"Duplicates$\downarrow$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("consistency", r"Consistency", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("factuality_author", r"Factuality$\uparrow$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("connectedness", r"Connectedness", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("similarity_pca", r"Similarity", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("diversity_gender", r"Diversity", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("parity_gender", r"Parity$\uparrow$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
]


# Define your panel order and labels
PANELS_FACTUALITY_GROUPS = [
    PanelSpec("factuality_author", r"Factuality$_{author}\uparrow$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("factuality_field", r"Factuality$_{field}\uparrow$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("factuality_epoch", r"Factuality$_{epoch}\uparrow$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("factuality_seniority", r"Factuality$_{seniority}\uparrow$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
]

PANELS_DIVERSITY_GROUPS = [
    PanelSpec("diversity_gender", r"Diversity$_{gender}$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("diversity_ethnicity", r"Diversity$_{ethnicity}$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("diversity_prominence_pub", r"Diversity$_{pub}$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("diversity_prominence_cit", r"Diversity$_{cit}$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
]

PANELS_PARITY_GROUPS = [
    PanelSpec("parity_gender", r"Parity$_{gender}\uparrow$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("parity_ethnicity", r"Parity$_{ethnicity}\uparrow$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("parity_prominence_pub", r"Parity$_{pub}\uparrow$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("parity_prominence_cit", r"Parity$_{cit}\uparrow$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
]


PANELS_BY_SOCIAL_GROUP = {
    "factuality": PANELS_FACTUALITY_GROUPS,
    "diversity": PANELS_DIVERSITY_GROUPS,
    "parity": PANELS_PARITY_GROUPS,
}

ylim_t = (0, 1.05)
xlim_t = (0, 2.0)
xticks_t = [0, 0.5, 1.0, 1.5, 2.0]
PANELS_METRICS = [
    PanelSpec("refusal_pct", r"Refusals", ylim=ylim_t, xlim=xlim_t, xticks=xticks_t, draw_ci=True),
    PanelSpec("validity_pct", r"Validity$\uparrow$", ylim=ylim_t, xlim=xlim_t, xticks=xticks_t, draw_ci=True),
    PanelSpec("duplicates", r"Duplicates$\downarrow$", ylim=ylim_t, xlim=xlim_t, xticks=xticks_t, draw_ci=True),
    PanelSpec("consistency", r"Consistency", ylim=ylim_t, xlim=xlim_t, xticks=xticks_t, draw_ci=True),
    PanelSpec("factuality_author", r"Factuality$\uparrow$", ylim=ylim_t, xlim=xlim_t, xticks=xticks_t, draw_ci=True),
    PanelSpec("connectedness", r"Connectedness", ylim=ylim_t, xlim=xlim_t, xticks=xticks_t, draw_ci=True),
    PanelSpec("similarity_pca", r"Similarity", ylim=ylim_t, xlim=xlim_t, xticks=xticks_t, draw_ci=True),
    PanelSpec("diversity_gender", r"Diversity", ylim=(0, 1), xlim=(0, 2.0), xticks=[0, 0.5, 1.0, 1.5, 2.0], draw_ci=True),
    PanelSpec("parity_gender", r"Parity$\uparrow$", ylim=ylim_t, xlim=xlim_t, xticks=xticks_t, draw_ci=True),
]


ylim=(-0.1, 1.1)
yticks=[0, .25, .5, .75, 1.0]
PANELS_METRICS_BEFORE_AFTER = [
    PanelSpec("refusal_pct", r"Refusals", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("validity_pct", r"Validity$\uparrow$", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("duplicates", r"Duplicates$\downarrow$", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("consistency", r"Consistency", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("factuality_author", r"Factuality$\uparrow$", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("connectedness", r"Connectedness", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("similarity_pca", r"Similarity", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("diversity_gender", r"Diversity", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("parity_gender", r"Parity$\uparrow$", ylim=ylim, yticks=yticks, draw_ci=True),
]

PANELS_METRICS_BEFORE_AFTER_FULLER = [
    PanelSpec("refusal_pct", r"Refusals", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("validity_pct", r"Validity $\uparrow$", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("duplicates", r"Duplicates$\downarrow$", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("consistency", r"Consistency", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("factuality_author", r"Factuality$\uparrow$", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("connectedness", r"Connectedness", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("similarity_pca", r"Similarity", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("diversity_gender", r"Diversity$_{gen.}$", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("diversity_ethnicity", r"Diversity$_{eth.}$", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("parity_gender", r"Parity$_{gen.}\uparrow$", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("parity_ethnicity", r"Parity$_{eth.}\uparrow$", ylim=ylim, yticks=yticks, draw_ci=True),
]

PANELS_METRICS_BEFORE_AFTER_DIVERSITY = [
    PanelSpec("factuality_author", r"Factuality$\uparrow$", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("connectedness", r"Connectedness", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("similarity_pca", r"Similarity", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("diversity_gender", r"Diversity$_{gen.}$", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("diversity_ethnicity", r"Diversity$_{eth.}$", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("diversity_prominence_pub", r"Diversity$_{pub.}$", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("diversity_prominence_cit", r"Diversity$_{cit.}$", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("parity_gender", r"Parity$_{gen.}\uparrow$", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("parity_ethnicity", r"Parity$_{eth.}\uparrow$", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("parity_prominence_pub", r"Parity$_{pub.}\uparrow$", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("parity_prominence_cit", r"Parity$_{cit.}\uparrow$", ylim=ylim, yticks=yticks, draw_ci=True),
]


# Color palettes per group (your scheme)
tab20 = plt.get_cmap("tab20")
tab20c = plt.get_cmap("tab20c")
GROUP_COLORS = {
    "model_access": [tab20(0), tab20(1)][::-1],
    "model_size": [tab20c(i) for i in range(8, 12)][::-1],
    "model_class": [tab20(2), tab20(3)][::-1],
}

BIASED_PROMPT_GENDER_COLORS = {
    "top_100_bias_gender_equal": 'tab:grey',
    "top_100_bias_gender_female": GENDER_COLOR_DICT[GENDER_FEMALE],
    "top_100_bias_gender_male": GENDER_COLOR_DICT[GENDER_MALE],
    "top_100_bias_gender_neutral": GENDER_COLOR_DICT[GENDER_UNISEX],
}

BIASED_PROMPT_ETHNICITY_COLORS = {
    "top_100_bias_ethnicity_equal": 'tab:grey',
    "top_100_bias_ethnicity_asian": ETHNICITY_COLOR_DICT[ETHNICITY_ASIAN],
    "top_100_bias_ethnicity_black": ETHNICITY_COLOR_DICT[ETHNICITY_BLACK],
    "top_100_bias_ethnicity_latino": ETHNICITY_COLOR_DICT[ETHNICITY_LATINO],
    "top_100_bias_ethnicity_white": ETHNICITY_COLOR_DICT[ETHNICITY_WHITE],
}   

BIASED_PROMPT_CITATIONS_COLORS = {
    "top_100_bias_citations_high": "tab:blue",
    "top_100_bias_citations_low": "tab:red",
}

BIASED_PROMPT_DIVERSE_COLORS = {
    "top_100_bias_diverse": "tab:orange",
}

tab10 = plt.get_cmap("tab10")
TASK_COLORS = {c: tab10(i) for i, c in enumerate(EXPERIMENT_TASKS)}
TASK_COLORS_2TWINS = {c: tab10(i + (1 if i==len(EXPERIMENT_TASKS_2TWINS)-1 else 0)) for i, c in enumerate(EXPERIMENT_TASKS_2TWINS)}
TASK_COLORS_2TWIN_BIAS_TOP_K = TASK_COLORS_2TWINS | {EXPERIMENT_TASK_BIASED_TOP_K: tab10(len(EXPERIMENT_TASKS_2TWINS)+1)}

tab20 = plt.get_cmap("tab20")
TASK_PARAM_COLORS = {(task_name, task_param):tab20((i*2) + j) for i, (task_name, task_params) in enumerate(TASK_PARAMS_BY_TASK.items()) for j, task_param in enumerate(task_params)}

TITLE_FONT_SIZE = 15
TICK_FONT_SIZE = 8
TICK_FONT_COLOR = '#828282'
LABEL_PANEL_FONT_SIZE = 11
SPINE_LW = 0.3
BEFORE_AFTER_X_LABELS = ("B", "A") #(r"$\times$", r"$\checkmark$")
BEFORE_AFTER_TICK_FONT_SIZE = 10
YLABEL_FONT_SIZE = 13