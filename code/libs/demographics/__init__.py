"""
Perceived demographics inferred from names only (for GT authors and recommended people alike):
    gender.py     nomquamgender on the first name (female / male / unknown, with p(female))
    ethnicity.py  raceBERT on the full name (asian / black / hispanic / white / other, with probabilities)
Probabilities are always kept, so thresholds and corrections (D.4) can be swept later.
"""
