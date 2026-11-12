"""
Perceived demographics inferred from names only (for GT authors and recommended people alike):
    gender.py     nomquamgender on the first name (female / male / unknown, with p(female))
    race.py       raceBERT race model on the full name (asian / black / hispanic / white / other; p_race_<class>)
    ethnicity.py  raceBERT ethnicity model on the full name (13 Wikipedia name classes; p_eth_<class>)
    stages.py     step A.3: every distinct name once per model, merged per author bucket
Probabilities are always kept, so thresholds and corrections (D.4) can be swept later.
"""
