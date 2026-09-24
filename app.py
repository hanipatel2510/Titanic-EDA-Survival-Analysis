import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Titanic Data Analysis",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Titanic Data Analysis Dashboard")

data1 = pd.read_csv("train.csv")
df = pd.DataFrame(data1)

st.subheader("First 5 Rows")
st.dataframe(df.head())

st.subheader("Last 5 Rows")
st.dataframe(df.tail())

st.write("Shape:", df.shape)

st.write("Columns:")
st.write(df.columns.tolist())

st.subheader("Dataset Information")
info_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Non-Null Count": df.notnull().sum().values,
    "Null Count": df.isnull().sum().values
})

st.dataframe(info_df, use_container_width=True)

st.subheader("Descriptive Statistics")
st.dataframe(df.describe(), use_container_width=True)

st.subheader("Missing Values")
st.dataframe(
    df.isnull().sum().sort_values(ascending=False).to_frame("Missing Values"),
    use_container_width=True
)

df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Age"] = df["Age"].astype(int)

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df["Cabin"] = df["Cabin"].fillna("Unknown")

st.subheader("Data After Cleaning")
st.dataframe(df, use_container_width=True)

st.subheader("Missing Values After Cleaning")
st.dataframe(
    df.isnull().sum().to_frame("Missing Values"),
    use_container_width=True
)

st.write("Duplicates before:", df.duplicated().sum())

df = df.drop_duplicates().reset_index(drop=True)

st.write("Duplicates after:", df.duplicated().sum())

st.subheader("Data Types")
st.dataframe(
    df.dtypes.astype(str).to_frame("Data Type"),
    use_container_width=True
)

st.subheader("Survival Count")
st.dataframe(
    df["Survived"].value_counts().to_frame("Passengers"),
    use_container_width=True
)

survival_rate = df["Survived"].mean() * 100

st.metric(
    "Survival Rate",
    f"{round(survival_rate, 2)}%"
)

st.subheader("Survival Count")

fig, ax = plt.subplots()

sns.countplot(
    x="Survived",
    data=df,
    ax=ax
)

ax.set_title(
    "Survival Count",
    fontweight="bold",
    fontsize=20
)

ax.set_xlabel("Survived")
ax.set_ylabel("Number of Passengers")

st.pyplot(fig)
plt.close(fig)

st.subheader("Gender Count")

st.dataframe(
    df["Sex"].value_counts().to_frame("Passengers"),
    use_container_width=True
)

fig, ax = plt.subplots()

sns.countplot(
    x="Sex",
    hue="Survived",
    data=df,
    palette=["purple", "brown"],
    ax=ax
)

ax.set_title(
    "Survival by Gender",
    fontweight="bold",
    fontsize=20
)

ax.set_xlabel("Gender")
ax.set_ylabel("Number of Passengers")

st.pyplot(fig)
plt.close(fig)

class_survival_rate = (
    df.groupby("Pclass")["Survived"].mean() * 100
)

st.subheader("Survival Rate by Passenger Class")

st.dataframe(
    class_survival_rate.round(2).to_frame("Survival Rate (%)"),
    use_container_width=True
)

st.subheader("Age Statistics")

age_statistics = df["Age"].agg(
    ["count", "mean", "median", "min", "max"]
).to_frame("Age")

st.dataframe(
    age_statistics,
    use_container_width=True
)

fig, ax = plt.subplots()

sns.histplot(
    df["Age"],
    bins=25,
    kde=True,
    color="#4B0082",
    ax=ax
)

ax.set_title(
    "Age Distribution of Passengers",
    fontweight="bold",
    fontsize=15
)

ax.set_xlabel("Age")
ax.set_ylabel("Number of Passengers")

st.pyplot(fig)
plt.close(fig)

bins = [0, 12, 19, 59, 100]
labels = ["Child", "Teen", "Adult", "Senior"]

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=bins,
    labels=labels
)

age_survival = (
    df.groupby("AgeGroup", observed=False)["Survived"].mean() * 100
)

st.subheader("Survival Rate by Age Group")

st.dataframe(
    age_survival.round(2).to_frame("Survival Rate (%)"),
    use_container_width=True
)

fig, ax = plt.subplots()

sns.countplot(
    x="AgeGroup",
    hue="Survived",
    data=df,
    palette=["purple", "red"],
    ax=ax
)

ax.set_title(
    "Survival by Age Group",
    fontweight="bold",
    fontsize=20
)

ax.set_xlabel("Age Group")
ax.set_ylabel("Number of Passengers")

st.pyplot(fig)
plt.close(fig)

st.subheader("Fare Statistics")

fare_summary = df["Fare"].agg(
    ["count", "mean", "median", "min", "max"]
).to_frame("Fare")

st.dataframe(
    fare_summary,
    use_container_width=True
)

fare_by_survival = df.groupby(
    "Survived"
)["Fare"].agg(
    ["mean", "median", "min", "max"]
)

fare_by_survival.index = [
    "Did Not Survive",
    "Survived"
]

st.subheader("Fare by Survival")

st.dataframe(
    fare_by_survival,
    use_container_width=True
)

fig, ax = plt.subplots(figsize=(7, 5))

sns.boxplot(
    x="Survived",
    y="Fare",
    data=df,
    ax=ax
)

ax.set_title(
    "Fare vs Survival",
    fontweight="bold",
    fontsize=20
)

ax.set_xlabel("Survived (0 = No, 1 = Yes)")
ax.set_ylabel("Fare")

st.pyplot(fig)
plt.close(fig)

df["FamilySize"] = (
    df["SibSp"] +
    df["Parch"] +
    1
)

family_analysis = df.groupby(
    "FamilySize"
)["Survived"].agg(
    Total_Passengers="count",
    Survivors="sum",
    Survival_Rate="mean"
)

family_analysis["Survival_Rate"] *= 100

st.subheader("Family Analysis")

st.dataframe(
    family_analysis,
    use_container_width=True
)

fig, ax = plt.subplots(figsize=(8, 5))

sns.countplot(
    x="FamilySize",
    hue="Survived",
    data=df,
    palette="Set2",
    ax=ax
)

ax.set_title("Survival by Family Size")
ax.set_xlabel("Family Size (including self)")
ax.set_ylabel("Number of Passengers")

ax.legend(
    title="Survived",
    labels=["No", "Yes"]
)

st.pyplot(fig)
plt.close(fig)

numeric_df = df.select_dtypes(
    include=np.number
)

correlation_matrix = numeric_df.corr()

st.subheader("Correlation Matrix")

st.dataframe(
    correlation_matrix,
    use_container_width=True
)

fig, ax = plt.subplots()

sns.countplot(
    x="Embarked",
    hue="Survived",
    data=df,
    palette="dark",
    ax=ax
)

ax.set_title("Survival by Port of Embarkation")

ax.set_xlabel(
    "Port (C = Cherbourg, Q = Queenstown, S = Southampton)"
)

ax.set_ylabel("Number of Passengers")

ax.legend(
    title="Survived",
    labels=["No", "Yes"]
)

st.pyplot(fig)
plt.close(fig)

st.subheader("Multi-Factor Analysis")

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)

sns.countplot(
    x="Pclass",
    hue="Sex",
    data=df,
    ax=axes[0, 0],
    palette="Set2"
)

axes[0, 0].set_title(
    "Passenger Count by Class and Gender"
)

axes[0, 0].set_xlabel("Passenger Class")
axes[0, 0].set_ylabel("Count")

sns.kdeplot(
    data=df,
    x="Age",
    hue="Survived",
    fill=True,
    common_norm=False,
    ax=axes[0, 1],
    palette=["red", "green"]
)

axes[0, 1].set_title(
    "Age Distribution by Survival"
)

axes[0, 1].set_xlabel("Age")

sns.violinplot(
    x="Pclass",
    y="Fare",
    hue="Survived",
    data=df,
    split=True,
    ax=axes[1, 0],
    palette="Set1"
)

axes[1, 0].set_title(
    "Fare Distribution by Class and Survival"
)

axes[1, 0].set_xlabel("Passenger Class")
axes[1, 0].set_ylabel("Fare")

survival_by_embarked = (
    df.groupby("Embarked", observed=False)["Survived"]
    .mean()
    .reset_index()
)

sns.barplot(
    x="Embarked",
    y="Survived",
    data=survival_by_embarked,
    ax=axes[1, 1],
    errorbar=None
)

axes[1, 1].set_title(
    "Survival Rate by Embarkation Port"
)

axes[1, 1].set_xlabel("Embarked")
axes[1, 1].set_ylabel("Survival Rate")

plt.suptitle(
    "Titanic Dataset - Multi-Factor Dashboard",
    fontsize=16,
    y=1.02
)

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)


st.subheader(
    "Survival Rate by Age Group, Gender, and Class"
)

age_gender_class = (
    df.groupby(
        ["AgeGroup", "Sex", "Pclass"],
        observed=False
    )["Survived"]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(
    figsize=(12, 6)
)

sns.barplot(
    data=age_gender_class,
    x="AgeGroup",
    y="Survived",
    hue="Sex",
    ax=ax,
    errorbar=None
)

ax.set_title(
    "Survival Rate by Age Group, Gender, and Class"
)

ax.set_xlabel("Age Group")
ax.set_ylabel("Survival Rate")

st.pyplot(fig)
plt.close(fig)

st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(
    figsize=(10, 7)
)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    ax=ax
)

ax.set_title(
    "Correlation Heatmap",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.success(
    "Titanic Data Analysis completed successfully."
)