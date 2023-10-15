import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from utility import plot_settings

sys.path.append("../")

pd.set_option("display.max_rows", 50)
pd.set_option("display.max_columns", 50)
pd.options.display.float_format = "{:,.2f}".format
