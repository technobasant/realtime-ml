"""The calendar every experiment shares, in days from the start of the generated data.

Days 0-13 only build feature history. A model is trained on day 63 with the labels that
have arrived by then, using transactions old enough (14 days) for most of their
chargebacks to have landed. Days 63-76 are its future; days 77-83 are the shadow period
in which a challenger is scored next to the champion.
"""

WARMUP_END = 14
TRAIN = (14, 42)
VALID = (43, 49)
TRAINED_AT = 63
MATURITY_DAYS = 14
TEST = (63, 76)
SHADOW = (77, 83)
DAYS = 90
