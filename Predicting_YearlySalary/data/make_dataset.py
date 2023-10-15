import pandas as pd
import numpy as np

pd.set_option("display.max_rows", 50)
pd.set_option("display.max_columns", 50)
pd.options.display.float_format = "{:,.2f}".format

# Import the df created in the Notebook 1
df = pd.read_pickle("../../data/interim/interest_SO_survey_2023.pkl")

df.head()
df.info()

""" I will predict the yearly salary of a developer based on the survey data,
    so, the data_cleaning process is gonna be focused on the target variable (Yearly Salary).
    It is going to be a regression model."""

df2 = df.dropna(subset=["YearlySalary"]).copy()

df2.head()
df2.isna().sum()
# I prefer to fill this with 'Not_Responded' instead of dropping the column or the Nan values
df2["Industry"] = df2["Industry"].fillna("Not_Responded")
df2["Industry"] = df2["Industry"].apply(
    lambda x: "Technology"
    if x == "Information Services, IT, Software Development, or other Technology"
    else x
)
df2["Industry"] = df2["Industry"].apply(
    lambda x: "Manufacturing"
    if x == "Manufacturing, Transportation, or Supply Chain"
    else x
)
df2["Industry"] = df2["Industry"].apply(
    lambda x: "Retail" if x == "Retail and Consumer Services" else x
)
top_industry = df2["Industry"].value_counts().nlargest(8).index
df2["Industry"] = df2["Industry"].apply(lambda x: x if x in top_industry else "Other")
df2["Industry"].value_counts()

""" I will drop some features that will no be useful for my model, like the target things,
    or those ones with too many missing values."""

df["BuyNewTool"].value_counts()

columns_to_drop = [
    "BuyNewTool",
    "CoursesCert",
    "ProfessionalTech",
    "TargetAIDeveloper",
    "TargetAISearch",
    "TargetCollabTools",
    "TargetDatabase",
    "TargetFramework",
    "TargetLanguage",
    "TargetLibraries",
    "TargetPlatform",
    "TargetTools",
    "WorkedAIDeveloper",
    "WorkedAISearch",
    "z_BenefitsAI",
    "z_FavorableAI",
    "z_TrustAI",
]
df3 = df2.drop(columns_to_drop, axis="columns").copy()
df3.isna().sum()
df3["Employment"].value_counts()
df3["MainBranch"].value_counts()
df3["Developer"] = df3["MainBranch"].apply(
    lambda x: 1 if x == "I am a developer by profession" else 0
)
df3[["WorkingYears", "YearsCodePro"]].corr()

""" Almost all the people are employed. I've already dealed with 'MainBranch'
    and WorkingYears have too many missing values, and it is highly correlated 
    with YearsCodePro, so I will drop it."""

df3 = df3.drop(["Employment", "WorkingYears", "MainBranch"], axis="columns").copy()

df3["WorkedPlatform"].value_counts()
df3["WorkedFramework"].value_counts()
df3["WorkedLibraries"].value_counts()

""" I would like to keep these 3 columns, but they have too many missing values
    and are too messy. I think the algortithm will infer a little bit of this information
    from the 'CurrentJob' column, so I will drop them."""
df4 = df3.drop(
    ["WorkedPlatform", "WorkedFramework", "WorkedLibraries"], axis="columns"
).copy()

df4.info()

""" To predict a worker's salary, these seems to be the most important features.
    """

df4["z_UsingAI"].value_counts()
df4["z_UsingAI"] = df4["z_UsingAI"].apply(lambda x: 1 if x == "Yes" else 0)

len(df4["EdLevel"].unique())
df4["EdLevel"].value_counts()
