# Pandas — Understanding Document

Pandas is the standard Python library for structured data analysis. It provides two core data structures — `Series` and `DataFrame` — built on top of NumPy, with a rich API for loading, cleaning, transforming, aggregating and exporting tabular data.

---

## Table of Contents

1. [What Pandas Provides](#1-what-pandas-provides)
2. [Core Data Structures](#2-core-data-structures)
3. [Creating DataFrames](#3-creating-dataframes)
4. [Data Types](#4-data-types)
5. [Inspecting Data](#5-inspecting-data)
6. [Indexing and Selection](#6-indexing-and-selection)
7. [Filtering](#7-filtering)
8. [Adding and Removing Columns](#8-adding-and-removing-columns)
9. [Missing Data](#9-missing-data)
10. [Data Cleaning](#10-data-cleaning)
11. [Transformation and Apply](#11-transformation-and-apply)
12. [Sorting and Ranking](#12-sorting-and-ranking)
13. [GroupBy and Aggregation](#13-groupby-and-aggregation)
14. [Merging, Joining and Concatenation](#14-merging-joining-and-concatenation)
15. [Pivot Tables and Cross-tabulation](#15-pivot-tables-and-cross-tabulation)
16. [Window Functions](#16-window-functions)
17. [Time Series](#17-time-series)
18. [String Operations](#18-string-operations)
19. [Categorical Data](#19-categorical-data)
20. [Reading and Writing Files](#20-reading-and-writing-files)
21. [Performance and Memory](#21-performance-and-memory)
22. [Apache Arrow Backend](#22-apache-arrow-backend)
23. [Quick Reference](#23-quick-reference)

---

## 1. What Pandas Provides

| Capability | Description |
|---|---|
| **Labelled axes** | Rows and columns have names, not just integer positions |
| **Mixed types** | Each column can hold a different data type |
| **Alignment** | Operations align on index labels automatically |
| **Missing data** | First-class `NaN` / `NA` support across all dtypes |
| **GroupBy** | Split-apply-combine aggregation |
| **Reshaping** | Pivot, melt, stack, unstack |
| **Time series** | Date ranges, resampling, rolling windows, timezone handling |
| **I/O** | CSV, Parquet, JSON, Excel, SQL, HDF5, Feather, ORC |
| **Integration** | NumPy, Matplotlib, scikit-learn, Arrow, Spark |

---

## 2. Core Data Structures

### Series — 1-D labelled array

```python
import pandas as pd
import numpy as np

# A Series is a 1-D array with an index (labels)
s = pd.Series([10, 20, 30, 40], index=["a", "b", "c", "d"])
print(s)
# a    10
# b    20
# c    30
# d    40
# dtype: int64

print(s["b"])        # 20   — label access
print(s[1])          # 20   — positional access
print(s.index)       # Index(['a', 'b', 'c', 'd'], dtype='object')
print(s.values)      # array([10, 20, 30, 40])
print(s.dtype)       # int64
print(s.name)        # None  (optional name attribute)

# Arithmetic aligns on index
s2 = pd.Series([1, 2, 3, 4], index=["b", "c", "d", "e"])
print(s + s2)
# a    NaN   (a not in s2)
# b     21
# c     32
# d     43
# e    NaN   (e not in s)
```

### DataFrame — 2-D labelled table

```python
df = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol", "Dave"],
    "age":    [30, 25, 35, 28],
    "salary": [75000.0, 62000.0, 91000.0, 55000.0],
    "city":   ["Mumbai", "Delhi", "Bangalore", "Mumbai"],
})

print(df)
print(df.index)           # RangeIndex(start=0, stop=4, step=1)
print(df.columns)         # Index(['name', 'age', 'salary', 'city'])
print(df.shape)           # (4, 4)
print(df.dtypes)
# name      object
# age        int64
# salary   float64
# city      object

# A DataFrame is a dict of Series sharing the same index
print(type(df["age"]))    # <class 'pandas.core.series.Series'>
```

### Index — the axis labels

```python
# Integer index (default)
df = pd.DataFrame({"x": [1, 2, 3]})
print(df.index)   # RangeIndex(start=0, stop=3, step=1)

# String index
df2 = pd.DataFrame({"x": [1, 2, 3]}, index=["a", "b", "c"])

# DateTime index
dti = pd.date_range("2024-01-01", periods=5, freq="D")
df3 = pd.DataFrame({"v": range(5)}, index=dti)

# MultiIndex (hierarchical)
idx = pd.MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 1)],
                                 names=["group", "id"])
df4 = pd.DataFrame({"val": [10, 20, 30]}, index=idx)
```

---

## 3. Creating DataFrames

```python
import pandas as pd
import numpy as np

# From a dict of lists
df1 = pd.DataFrame({
    "a": [1, 2, 3],
    "b": [4.0, 5.0, 6.0],
})

# From a list of dicts (one dict per row)
df2 = pd.DataFrame([
    {"name": "Alice", "age": 30},
    {"name": "Bob",   "age": 25},
])

# From a 2-D NumPy array
arr = np.random.randn(4, 3)
df3 = pd.DataFrame(arr, columns=["x", "y", "z"])

# From a CSV/file (see Section 20)
df4 = pd.read_csv("data.csv")

# Empty DataFrame with a known schema
df5 = pd.DataFrame(columns=["id", "name", "score"])

# Constant value broadcast
df6 = pd.DataFrame({"status": "active"}, index=range(5))

# From a range
df7 = pd.DataFrame({"n": range(100)})

# Copy
df_copy = df1.copy()     # independent copy
df_view = df1            # NOT a copy — same object
```

---

## 4. Data Types

### Default dtype mapping

| Python / NumPy type | Pandas dtype | Notes |
|---|---|---|
| `int` | `int64` | No null support — null → float upcast |
| `float` | `float64` | NaN is the null sentinel |
| `bool` | `bool` | No null support |
| `str` | `object` | Heterogeneous; slow |
| `datetime64` | `datetime64[ns]` | Nanosecond precision |
| `timedelta` | `timedelta64[ns]` | |
| `category` | `CategoricalDtype` | Dictionary-encoded |

### Nullable integer and boolean types (Pandas 1.0+)

```python
# These support NA without upcasting to float
s = pd.Series([1, 2, None], dtype="Int64")   # capital I
print(s.dtype)   # Int64
print(s[2])      # <NA>

# All nullable extension types
pd.array([1, None], dtype="Int8")
pd.array([1, None], dtype="Int16")
pd.array([1, None], dtype="Int32")
pd.array([1, None], dtype="Int64")
pd.array([1, None], dtype="UInt8")
pd.array([True, None], dtype="boolean")
pd.array(["a", None], dtype="string")
pd.array([1.0, None], dtype="Float64")
```

### Checking and casting

```python
df = pd.DataFrame({"a": [1,2,3], "b": ["x","y","z"], "c": [1.0,2.0,3.0]})

# Inspect types
print(df.dtypes)
print(df.select_dtypes(include="number"))        # only numeric columns
print(df.select_dtypes(include=["object"]))      # only string columns
print(df.select_dtypes(exclude=["float64"]))

# Cast a column
df["a"] = df["a"].astype("float32")
df["b"] = df["b"].astype("category")

# Cast the whole DataFrame
df_f32 = df.select_dtypes("float64").astype("float32")

# Infer better types automatically
df_better = df.convert_dtypes()   # upcasts object→string, int→Int64, etc.
```

---

## 5. Inspecting Data

```python
df = pd.read_csv("data.csv")

# Shape and size
df.shape          # (rows, cols)
len(df)           # number of rows
df.size           # total cells
df.ndim           # 2

# Quick look
df.head(5)        # first 5 rows (default)
df.tail(3)        # last 3 rows
df.sample(10)     # 10 random rows
df.sample(frac=0.1) # 10% random sample

# Schema and types
df.dtypes
df.info()         # dtypes + non-null counts + memory
df.columns
df.index

# Statistics
df.describe()                      # count, mean, std, min, quartiles, max
df.describe(include="all")         # includes categorical columns
df.describe(include=["object"])    # string columns only

# Value counts
df["city"].value_counts()
df["city"].value_counts(normalize=True)   # proportions
df["city"].value_counts(dropna=False)     # include NaN

# Unique values
df["city"].unique()       # array of unique values
df["city"].nunique()      # count of unique values

# Missing data summary
df.isnull().sum()                          # null count per column
df.isnull().mean() * 100                   # null % per column
df.isnull().sum().sum()                    # total nulls
df.notnull().all()                         # are all values present?

# Memory usage
df.memory_usage(deep=True)                 # bytes per column
df.memory_usage(deep=True).sum() / 1e6    # total MB
```

---

## 6. Indexing and Selection

### Column selection

```python
# Single column → Series
s = df["name"]

# Multiple columns → DataFrame
sub = df[["name", "salary"]]

# By dtype
nums = df.select_dtypes(include="number")
```

### Row selection with `.loc` (label-based)

```python
# Single row by label
df.loc[0]              # row with index label 0

# Range of labels (inclusive on both ends)
df.loc[0:2]            # rows 0, 1, 2

# Row + column
df.loc[0, "name"]                  # scalar
df.loc[0:2, "name":"salary"]       # sub-table
df.loc[[0, 2], ["name", "age"]]    # specific rows and columns

# Boolean mask
mask = df["salary"] > 70000
df.loc[mask, ["name", "salary"]]

# Setting values
df.loc[0, "salary"] = 80000.0
df.loc[mask, "salary"] *= 1.1
```

### Row selection with `.iloc` (integer position-based)

```python
df.iloc[0]                    # first row
df.iloc[-1]                   # last row
df.iloc[0:3]                  # rows 0, 1, 2
df.iloc[[0, 2, 4]]            # rows 0, 2, 4
df.iloc[0:3, 1:3]             # rows 0-2, columns 1-2
df.iloc[:, -1]                # last column, all rows
```

### `at` and `iat` — fast scalar access

```python
df.at[2, "name"]      # fast single-cell label access
df.iat[2, 1]          # fast single-cell positional access
df.at[2, "name"] = "Carol Updated"
```

---

## 7. Filtering

```python
df = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol", "Dave"],
    "age":    [30, 25, 35, 28],
    "salary": [75000.0, 62000.0, 91000.0, 55000.0],
    "city":   ["Mumbai", "Delhi", "Bangalore", "Mumbai"],
})

# Single condition
df[df["age"] > 28]
df[df["city"] == "Mumbai"]

# Multiple conditions — use & and | with parentheses
df[(df["age"] > 25) & (df["salary"] > 60000)]
df[(df["city"] == "Mumbai") | (df["city"] == "Delhi")]

# Negation
df[~(df["city"] == "Mumbai")]

# .isin() — membership check
df[df["city"].isin(["Mumbai", "Bangalore"])]

# .between() — inclusive range
df[df["age"].between(25, 30)]

# .str methods for string filtering
df[df["name"].str.startswith("A")]
df[df["name"].str.contains("o", case=False)]

# .query() — SQL-like string expression
df.query("age > 28 and salary > 60000")
df.query("city in ['Mumbai', 'Delhi']")
df.query("age.between(25, 30)")   # via method access

# Filter using a boolean Series
mask = (df["salary"] > df["salary"].mean())
df[mask]

# Filter and reset index
df[df["age"] > 28].reset_index(drop=True)
```

---

## 8. Adding and Removing Columns

```python
df = pd.DataFrame({"a": [1,2,3], "b": [4.0,5.0,6.0]})

# Add a new column
df["c"]         = df["a"] + df["b"]
df["d"]         = [10, 20, 30]
df["constant"]  = 0
df["flag"]      = df["a"] > 1   # boolean column

# assign() — non-mutating, returns new DataFrame
df2 = df.assign(
    e=df["a"] * 2,
    f=lambda x: x["a"] + x["b"],   # can reference prior assigns
)

# insert at a specific position
df.insert(loc=1, column="new_col", value=[7, 8, 9])

# Remove columns
df.drop(columns=["constant", "flag"], inplace=True)

# Remove rows by index label
df.drop(index=[0, 2], inplace=True)

# pop() — remove and return a column
series = df.pop("d")

# Rename columns
df.rename(columns={"a": "alpha", "b": "beta"}, inplace=True)
df.columns = ["x", "y", "z"]   # rename all at once
```

---

## 9. Missing Data

```python
df = pd.DataFrame({
    "a": [1.0, np.nan, 3.0, np.nan, 5.0],
    "b": ["x", "y", None, "w", None],
    "c": [10, 20, 30, 40, 50],
})

# ── Detect ─────────────────────────────────────────────────────────────────────

df.isnull()               # boolean mask: True where NaN/None
df.notnull()              # inverse
df.isnull().sum()         # count per column
df.isnull().any(axis=1)   # True for rows with any null
df.isnull().all(axis=1)   # True for rows where all values are null

# ── Drop ───────────────────────────────────────────────────────────────────────

df.dropna()                          # drop any row with at least one null
df.dropna(how="all")                 # drop row only if ALL values are null
df.dropna(subset=["a"])              # drop only if null in column "a"
df.dropna(thresh=2)                  # keep rows with at least 2 non-null values
df.dropna(axis=1)                    # drop columns that contain any null

# ── Fill ───────────────────────────────────────────────────────────────────────

df.fillna(0)                         # fill all nulls with 0
df.fillna({"a": df["a"].median(), "b": "unknown"})  # per-column fill
df.fillna(method="ffill")            # forward-fill (propagate last valid)
df.fillna(method="bfill")            # backward-fill
df["a"].fillna(df["a"].mean())       # mean imputation

# Interpolate numeric columns
df["a"].interpolate(method="linear")
df["a"].interpolate(method="time")   # for time-indexed data

# ── Replace ────────────────────────────────────────────────────────────────────

df.replace(np.nan, 0)
df.replace({"b": {None: "missing"}})
df["a"].replace(to_replace=np.nan, value=df["a"].median())
```

---

## 10. Data Cleaning

```python
df = pd.DataFrame({
    "name":   [" Alice ", "BOB", "carol", "Dave", "Dave"],
    "age":    [30, 25, 35, 28, 28],
    "salary": ["$75,000", "$62,000", "$91,000", "$55,000", "$55,000"],
    "email":  ["alice@x.com", "bob@x.com", "carol@x.com", "dave@x.com", "dave@x.com"],
})

# ── Duplicates ─────────────────────────────────────────────────────────────────

df.duplicated()                         # boolean mask of duplicate rows
df.duplicated(subset=["name", "age"])   # based on specific columns
df.duplicated(keep="first")             # mark all but first occurrence
df.duplicated(keep="last")
df.duplicated(keep=False)              # mark ALL duplicates

df.drop_duplicates()                       # remove duplicate rows
df.drop_duplicates(subset=["name"])        # based on name column only
df.drop_duplicates(keep="last")

# ── String cleaning ────────────────────────────────────────────────────────────

df["name"] = df["name"].str.strip()          # remove whitespace
df["name"] = df["name"].str.title()          # title case
df["name"] = df["name"].str.lower()          # lowercase
df["name"] = df["name"].str.upper()          # uppercase
df["name"] = df["name"].str.replace(" ", "_")

# ── Numeric cleaning ───────────────────────────────────────────────────────────

# Remove currency symbols and commas, then convert
df["salary"] = (df["salary"]
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .astype(float))

# ── Type conversion ────────────────────────────────────────────────────────────

df["age"] = pd.to_numeric(df["age"], errors="coerce")  # coerce bad → NaN
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["name"] = df["name"].astype("category")

# ── Outlier clipping ───────────────────────────────────────────────────────────

q_low  = df["salary"].quantile(0.01)
q_high = df["salary"].quantile(0.99)
df["salary"] = df["salary"].clip(lower=q_low, upper=q_high)
```

---

## 11. Transformation and Apply

```python
df = pd.DataFrame({
    "a": [1, 2, 3, 4],
    "b": [10.0, 20.0, 30.0, 40.0],
    "city": ["Mumbai", "Delhi", "Bangalore", "Mumbai"],
})

# ── Vectorised operations (preferred — fast) ───────────────────────────────────

df["c"] = df["a"] * 2 + df["b"]
df["log_b"] = np.log(df["b"])
df["pct"] = df["b"] / df["b"].sum()

# ── map() — element-wise on a Series ──────────────────────────────────────────

df["city_code"] = df["city"].map({"Mumbai": 1, "Delhi": 2, "Bangalore": 3})

# Lambda via map
df["a_sq"] = df["a"].map(lambda x: x ** 2)

# ── apply() — row-wise or column-wise ─────────────────────────────────────────

# Apply a function to each column (axis=0, default)
df.apply(np.sum)                          # sum of each column

# Apply a function to each row (axis=1)
df["row_max"] = df[["a", "b"]].apply(max, axis=1)

# Apply with custom function
def normalise(col):
    return (col - col.min()) / (col.max() - col.min())

df[["a", "b"]] = df[["a", "b"]].apply(normalise)

# Apply returning multiple columns
def extract(row):
    return pd.Series({"first": row["city"][0], "length": len(row["city"])})

df[["first", "length"]] = df.apply(extract, axis=1)

# ── applymap / map (element-wise on DataFrame) ─────────────────────────────────

# Pandas 2.1+: use .map() instead of deprecated .applymap()
df[["a", "b"]].map(lambda x: round(x, 2))

# ── transform() — aggregation that keeps the original shape ───────────────────

# Useful inside groupby, but also standalone
df["a_zscore"] = (df["a"] - df["a"].mean()) / df["a"].std()

# Or with transform
df["b_norm"] = df["b"].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

# ── pipe() — method chaining ───────────────────────────────────────────────────

def add_ratio(df_, num_col, den_col, new_col):
    df_[new_col] = df_[num_col] / df_[den_col]
    return df_

result = (df
    .pipe(add_ratio, "a", "b", "ratio")
    .query("ratio > 0.05")
    .reset_index(drop=True))
```

---

## 12. Sorting and Ranking

```python
df = pd.DataFrame({
    "name":   ["Carol", "Alice", "Dave", "Bob"],
    "age":    [35, 30, 28, 25],
    "salary": [91000, 75000, 55000, 62000],
})

# Sort by one column
df.sort_values("age")
df.sort_values("age", ascending=False)

# Sort by multiple columns
df.sort_values(["city", "salary"], ascending=[True, False])

# Sort by index
df.sort_index()
df.sort_index(ascending=False)

# Rank
df["salary_rank"] = df["salary"].rank(ascending=False, method="dense")
# method: 'average', 'min', 'max', 'first', 'dense'

df["pct_rank"] = df["salary"].rank(pct=True)

# nlargest / nsmallest (more efficient than sort + head)
df.nlargest(2, "salary")
df.nsmallest(3, "age")
```

---

## 13. GroupBy and Aggregation

```python
df = pd.DataFrame({
    "region":  ["N","S","N","E","S","N","E"],
    "product": ["A","A","B","A","B","A","B"],
    "revenue": [100,200,150,300,250,180,220],
    "cost":    [60, 120,90, 180,150,100,130],
})

# ── Basic aggregation ─────────────────────────────────────────────────────────

# Single column, single agg
df.groupby("region")["revenue"].sum()
df.groupby("region")["revenue"].mean()
df.groupby("region")["revenue"].agg(["sum","mean","count","std"])

# Multi-column groupby
df.groupby(["region","product"])["revenue"].sum()

# ── Named aggregations (agg with dict) ────────────────────────────────────────

summary = df.groupby("region").agg(
    total_revenue=("revenue", "sum"),
    avg_revenue  =("revenue", "mean"),
    min_cost     =("cost",    "min"),
    n_orders     =("revenue", "count"),
)
print(summary)

# ── Custom aggregation functions ──────────────────────────────────────────────

def range_val(x):
    return x.max() - x.min()

df.groupby("region")["revenue"].agg(range_val)
df.groupby("region").agg({"revenue": range_val, "cost": "mean"})

# ── transform() — keeps original shape ────────────────────────────────────────

# Add group-level statistic back to each row
df["region_total"]  = df.groupby("region")["revenue"].transform("sum")
df["region_avg"]    = df.groupby("region")["revenue"].transform("mean")
df["revenue_zscore"] = df.groupby("region")["revenue"].transform(
    lambda x: (x - x.mean()) / x.std()
)

# ── filter() — remove groups not meeting a condition ──────────────────────────

# Keep only groups where total revenue > 400
df_filtered = df.groupby("region").filter(lambda x: x["revenue"].sum() > 400)

# ── apply() — arbitrary group-level function ──────────────────────────────────

def top_product(grp):
    return grp.nlargest(1, "revenue")

df.groupby("region").apply(top_product).reset_index(drop=True)

# ── Aggregation shortcuts ─────────────────────────────────────────────────────

df.groupby("region").sum()
df.groupby("region").mean(numeric_only=True)
df.groupby("region").size()          # count rows per group
df.groupby("region").count()         # non-null count per column per group
df.groupby("region").first()
df.groupby("region").last()
df.groupby("region").nth(0)          # first row of each group

# ── Multiple keys, include group in output ────────────────────────────────────

df.groupby("region", as_index=False)["revenue"].sum()  # region as column
```

---

## 14. Merging, Joining and Concatenation

```python
left = pd.DataFrame({
    "id":   [1, 2, 3, 4],
    "name": ["Alice","Bob","Carol","Dave"],
    "dept": ["Eng","Mkt","Eng","HR"],
})

right = pd.DataFrame({
    "id":     [1, 2, 3, 5],
    "salary": [75000, 62000, 91000, 50000],
    "city":   ["Mumbai","Delhi","Bangalore","Chennai"],
})

# ── merge() — SQL-style join ───────────────────────────────────────────────────

# Inner join (default)
pd.merge(left, right, on="id")

# Left join
pd.merge(left, right, on="id", how="left")

# Right join
pd.merge(left, right, on="id", how="right")

# Outer join
pd.merge(left, right, on="id", how="outer")

# Different key names
pd.merge(left, right, left_on="id", right_on="emp_id")

# Multiple keys
pd.merge(left, right, on=["dept", "id"])

# Suffix for overlapping column names
pd.merge(left, right, on="id", suffixes=("_left", "_right"))

# ── join() — join on index ─────────────────────────────────────────────────────

left_idx  = left.set_index("id")
right_idx = right.set_index("id")
left_idx.join(right_idx, how="left")

# ── concat() — stack DataFrames ───────────────────────────────────────────────

df1 = pd.DataFrame({"a": [1,2], "b": [3,4]})
df2 = pd.DataFrame({"a": [5,6], "b": [7,8]})

# Stack rows (axis=0, default)
pd.concat([df1, df2], ignore_index=True)

# Stack columns (axis=1)
pd.concat([df1, df2], axis=1)

# Concat with keys to create MultiIndex
pd.concat([df1, df2], keys=["batch1","batch2"])

# ── combine_first() — fill missing values from another DataFrame ───────────────

df_primary   = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 5, 6]})
df_secondary = pd.DataFrame({"a": [10, 20, 30],   "b": [40, 50, 60]})
df_primary.combine_first(df_secondary)
# a: [1, 20, 3]   b: [40, 5, 6]  — primary values win; NaNs filled from secondary
```

---

## 15. Pivot Tables and Cross-tabulation

```python
df = pd.DataFrame({
    "region":  ["N","S","N","E","S","N","E"],
    "product": ["A","A","B","A","B","A","B"],
    "revenue": [100,200,150,300,250,180,220],
    "quarter": ["Q1","Q1","Q1","Q2","Q2","Q2","Q2"],
})

# ── pivot_table ────────────────────────────────────────────────────────────────

pt = pd.pivot_table(
    df,
    values="revenue",
    index="region",
    columns="product",
    aggfunc="sum",
    fill_value=0,
    margins=True,            # add row/column totals
    margins_name="Total",
)
print(pt)

# Multiple values and aggregations
pd.pivot_table(df,
    values=["revenue"],
    index=["region","quarter"],
    columns="product",
    aggfunc={"revenue": ["sum","mean"]},
)

# ── pivot() — no aggregation, unique index required ────────────────────────────

wide = df.pivot(index="region", columns="product", values="revenue")

# ── melt() — wide to long (unpivot) ───────────────────────────────────────────

wide_df = pd.DataFrame({
    "name": ["Alice","Bob"],
    "math": [90, 80],
    "english": [85, 75],
})

long_df = wide_df.melt(
    id_vars="name",
    value_vars=["math","english"],
    var_name="subject",
    value_name="score",
)
#     name  subject  score
# 0  Alice     math     90
# 1    Bob     math     80
# 2  Alice  english     85
# 3    Bob  english     75

# ── cross-tabulation ───────────────────────────────────────────────────────────

pd.crosstab(df["region"], df["product"])               # frequency
pd.crosstab(df["region"], df["product"], margins=True) # with totals
pd.crosstab(df["region"], df["product"],
            values=df["revenue"], aggfunc="sum")        # aggregated

# ── stack / unstack ───────────────────────────────────────────────────────────

# stack: columns → row index level
stacked = pt.stack()

# unstack: row index level → columns
unstacked = stacked.unstack()
```

---

## 16. Window Functions

```python
df = pd.DataFrame({
    "date":    pd.date_range("2024-01-01", periods=10, freq="D"),
    "revenue": [100,150,130,200,180,220,190,250,210,270],
})
df = df.set_index("date")

# ── Rolling (sliding window) ───────────────────────────────────────────────────

df["ma_3"]     = df["revenue"].rolling(window=3).mean()   # 3-day moving avg
df["ma_7"]     = df["revenue"].rolling(window=7).mean()
df["roll_std"] = df["revenue"].rolling(window=3).std()
df["roll_sum"] = df["revenue"].rolling(window=3).sum()
df["roll_max"] = df["revenue"].rolling(window=3).max()
df["roll_min"] = df["revenue"].rolling(window=3).min()

# min_periods: compute even if window is not full yet
df["ma_3_mp"] = df["revenue"].rolling(window=3, min_periods=1).mean()

# centre the window
df["ma_3_c"]  = df["revenue"].rolling(window=3, center=True).mean()

# Exponentially weighted moving average (more weight on recent values)
df["ewm_span3"] = df["revenue"].ewm(span=3).mean()
df["ewm_half2"] = df["revenue"].ewm(halflife=2).mean()

# ── Expanding (cumulative from start) ─────────────────────────────────────────

df["cum_sum"]  = df["revenue"].expanding().sum()
df["cum_mean"] = df["revenue"].expanding().mean()
df["cum_max"]  = df["revenue"].expanding().max()

# Shorthand for cumulative functions
df["cumsum"]   = df["revenue"].cumsum()
df["cumprod"]  = df["revenue"].cumprod()
df["cummax"]   = df["revenue"].cummax()
df["cummin"]   = df["revenue"].cummin()

# ── Shift and diff ────────────────────────────────────────────────────────────

df["prev_day"]    = df["revenue"].shift(1)    # lag by 1
df["next_day"]    = df["revenue"].shift(-1)   # lead by 1
df["day_on_day"]  = df["revenue"].diff(1)     # revenue[t] - revenue[t-1]
df["pct_change"]  = df["revenue"].pct_change()

# ── Rank within a window ───────────────────────────────────────────────────────

df["rank_all"] = df["revenue"].rank()
df["rank_pct"] = df["revenue"].rank(pct=True)
```

---

## 17. Time Series

```python
import pandas as pd

# ── Date ranges ────────────────────────────────────────────────────────────────

pd.date_range("2024-01-01", periods=12, freq="ME")  # month-end
pd.date_range("2024-01-01", "2024-12-31", freq="W")  # weekly
pd.date_range("2024-01-01", periods=5, freq="BH")    # business hours
pd.date_range("2024-01-01", periods=5, freq="QE")    # quarter-end

# Common frequency aliases
# D   = calendar day        B  = business day
# W   = week               ME = month-end
# MS  = month-start        QE = quarter-end
# YE  = year-end           H  = hour
# T/min = minute           S  = second

# ── Parsing and converting ─────────────────────────────────────────────────────

df = pd.DataFrame({
    "date_str": ["2024-01-15", "2024-06-20", "2024-12-01"],
    "value":    [100, 200, 150],
})
df["date"] = pd.to_datetime(df["date_str"])
df = df.set_index("date")

# Infer format automatically
s = pd.to_datetime(["Jan 15 2024", "15/06/2024", "2024.12.01"],
                   infer_datetime_format=True)

# ── Date component extraction ──────────────────────────────────────────────────

df["year"]       = df.index.year
df["month"]      = df.index.month
df["day"]        = df.index.day
df["hour"]       = df.index.hour
df["dayofweek"]  = df.index.dayofweek    # 0=Monday
df["dayname"]    = df.index.day_name()
df["quarter"]    = df.index.quarter
df["weekofyear"] = df.index.isocalendar().week
df["is_month_end"] = df.index.is_month_end

# ── Resampling ─────────────────────────────────────────────────────────────────

# Downsample: hourly → daily
hourly = pd.DataFrame({
    "value": range(48)
}, index=pd.date_range("2024-01-01", periods=48, freq="h"))

daily  = hourly.resample("D").sum()
weekly = hourly.resample("W").mean()

# Multiple aggregations
hourly.resample("D").agg({"value": ["sum","mean","max","min"]})

# Upsample: daily → hourly (fill gaps)
daily.resample("h").ffill()     # forward fill
daily.resample("h").interpolate()

# ── Timezone handling ──────────────────────────────────────────────────────────

df_utc = df.copy()
df_utc.index = df_utc.index.tz_localize("UTC")
df_ist = df_utc.index.tz_convert("Asia/Kolkata")

# ── Partial string indexing ────────────────────────────────────────────────────

df["2024"]          # all of 2024
df["2024-06"]       # all of June 2024
df["2024-01":"2024-06"]  # date range slice

# ── Period index ───────────────────────────────────────────────────────────────

pi = pd.period_range("2024Q1", periods=4, freq="Q")
pf = pd.DataFrame({"gdp": [100, 105, 110, 108]}, index=pi)
pf.to_timestamp()   # convert to DatetimeIndex
```

---

## 18. String Operations

All string methods are accessed via the `.str` accessor on a Series.

```python
s = pd.Series(["Alice Smith", "BOB JONES", "  carol white  ", None, "dave"])

# ── Case ──────────────────────────────────────────────────────────────────────

s.str.lower()
s.str.upper()
s.str.title()
s.str.capitalize()
s.str.swapcase()

# ── Whitespace ────────────────────────────────────────────────────────────────

s.str.strip()
s.str.lstrip()
s.str.rstrip()
s.str.strip(".")

# ── Search and match ──────────────────────────────────────────────────────────

s.str.contains("alice", case=False, na=False)
s.str.startswith("Alice")
s.str.endswith("Smith")
s.str.match(r"^[A-Z]")       # regex match at start
s.str.fullmatch(r"[A-Z][a-z]+ [A-Z][a-z]+")

s.str.find("Smith")           # position of first occurrence (-1 if not found)
s.str.count("o")              # count occurrences

# ── Replace ───────────────────────────────────────────────────────────────────

s.str.replace("Smith", "Johnson")
s.str.replace(r"\s+", "_", regex=True)       # regex replace
s.str.replace(r"[^a-zA-Z ]", "", regex=True) # remove non-alpha

# ── Split and join ────────────────────────────────────────────────────────────

s.str.split(" ")                      # returns list per element
s.str.split(" ", expand=True)         # returns DataFrame of columns
s.str.split(" ", n=1, expand=True)    # split at most once

s.str.cat(sep=", ")                   # join all strings in Series to one

# ── Extraction ────────────────────────────────────────────────────────────────

emails = pd.Series(["user@domain.com", "other@test.org"])
emails.str.extract(r"(\w+)@(\w+)\.(\w+)")   # 3 capture groups → 3 columns

s.str.slice(0, 3)             # first 3 characters
s.str.get(0)                  # first character

# ── Padding and alignment ─────────────────────────────────────────────────────

s.str.pad(width=20, side="right", fillchar=" ")
s.str.center(20, fillchar="-")
s.str.zfill(10)              # zero-pad on left

# ── Other ─────────────────────────────────────────────────────────────────────

s.str.len()                  # length of each string
s.str.isdigit()              # True if all characters are digits
s.str.isalpha()
s.str.isnumeric()
s.str.isspace()

# Null-safe: most .str methods propagate NaN without raising
s.str.upper()   # None → NaN in result
```

---

## 19. Categorical Data

```python
# ── Create categorical Series ──────────────────────────────────────────────────

s = pd.Categorical(["bronze","silver","gold","bronze","platinum"],
                   categories=["bronze","silver","gold","platinum"],
                   ordered=True)

s_series = pd.Series(s)
print(s_series.dtype)    # category

df = pd.DataFrame({"tier": pd.Categorical(
    ["bronze","silver","gold","bronze"],
    categories=["bronze","silver","gold","platinum"],
    ordered=True,
)})

# ── Access cat attributes ─────────────────────────────────────────────────────

df["tier"].cat.categories          # Index with the category labels
df["tier"].cat.codes               # integer codes
df["tier"].cat.ordered             # True

# ── Modify categories ─────────────────────────────────────────────────────────

df["tier"].cat.add_categories(["diamond"])
df["tier"].cat.remove_categories(["platinum"])
df["tier"].cat.rename_categories({"bronze": "Bronze"})
df["tier"].cat.reorder_categories(["platinum","gold","silver","bronze"])
df["tier"].cat.remove_unused_categories()

# ── Ordered comparison (requires ordered=True) ────────────────────────────────

df["tier"] > "silver"    # [False, False, True, False]

# ── Memory saving — why categoricals matter ────────────────────────────────────

s_obj = pd.Series(["bronze","silver","gold"] * 100_000)
s_cat = s_obj.astype("category")

print(s_obj.memory_usage(deep=True))   # ~24 MB (object dtype)
print(s_cat.memory_usage(deep=True))   # ~0.1 MB (integer codes + lookup)
```

---

## 20. Reading and Writing Files

### CSV

```python
# Read
df = pd.read_csv("data.csv")
df = pd.read_csv("data.csv",
    sep="\t",                          # tab-delimited
    header=0,                          # row to use as column names
    names=["a","b","c"],               # override column names
    index_col="id",                    # use this column as index
    usecols=["name","salary"],         # load only these columns
    dtype={"age": "Int32"},            # specify dtypes
    parse_dates=["signup_date"],       # parse as datetime
    na_values=["NA","N/A","--"],       # additional null strings
    nrows=1000,                        # read only first 1000 rows
    skiprows=[1,2],                    # skip rows by position
    chunksize=10_000,                  # returns iterator for large files
    encoding="utf-8",
    compression="gzip",                # read .csv.gz directly
)

# Chunked read for large files
chunks = pd.read_csv("large.csv", chunksize=50_000)
result = pd.concat([chunk.query("value > 0") for chunk in chunks])

# Write
df.to_csv("output.csv", index=False)
df.to_csv("output.csv.gz", index=False, compression="gzip")
```

### Parquet

```python
# Read
df = pd.read_parquet("data.parquet")
df = pd.read_parquet("data.parquet",
    engine="pyarrow",                  # or "fastparquet"
    columns=["name","salary"],         # column pruning
    filters=[("salary", ">", 50000)],  # predicate pushdown
)

# Write
df.to_parquet("output.parquet",
    engine="pyarrow",
    compression="snappy",              # snappy / gzip / brotli / zstd
    index=False,
)
```

### JSON

```python
df = pd.read_json("data.json")
df = pd.read_json("data.jsonl", lines=True)   # JSON Lines format

df.to_json("output.json", orient="records", lines=True)
# orient: 'records', 'index', 'columns', 'values', 'table', 'split'
```

### Excel

```python
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")
df = pd.read_excel("data.xlsx", sheet_name=0, header=0, usecols="A:D")

df.to_excel("output.xlsx", sheet_name="Results", index=False)

# Multiple sheets
with pd.ExcelWriter("multi.xlsx") as writer:
    df1.to_excel(writer, sheet_name="Sales")
    df2.to_excel(writer, sheet_name="Costs")
```

### SQL

```python
import sqlalchemy

engine = sqlalchemy.create_engine("sqlite:///mydb.sqlite")

df = pd.read_sql("SELECT * FROM orders WHERE amount > 100", engine)
df = pd.read_sql_table("customers", engine)
df = pd.read_sql_query("SELECT id, name FROM customers LIMIT 10", engine)

df.to_sql("output_table", engine,
          if_exists="replace",    # or 'append', 'fail'
          index=False,
          chunksize=500,
          dtype={"id": sqlalchemy.Integer()})
```

### HDF5 / Feather

```python
# HDF5 (fast binary format)
df.to_hdf("data.h5", key="df", mode="w", complevel=9)
df = pd.read_hdf("data.h5", key="df")

# Feather (Arrow IPC format — very fast)
df.to_feather("data.feather")
df = pd.read_feather("data.feather", columns=["name","salary"])
```

---

## 21. Performance and Memory

```python
# ── Reduce memory at load time ─────────────────────────────────────────────────

df = pd.read_csv("data.csv", dtype={
    "id":      "int32",       # not int64
    "age":     "int8",
    "score":   "float32",     # not float64
    "city":    "category",
})

# ── Downcast existing columns ──────────────────────────────────────────────────

df["age"]   = pd.to_numeric(df["age"],   downcast="integer")   # smallest int
df["score"] = pd.to_numeric(df["score"], downcast="float")     # smallest float

# ── Category for string columns ────────────────────────────────────────────────

for col in df.select_dtypes("object").columns:
    if df[col].nunique() / len(df) < 0.5:    # low-cardinality heuristic
        df[col] = df[col].astype("category")

# ── Avoid unnecessary copies ───────────────────────────────────────────────────

# Bad — creates a copy of the whole DataFrame
df2 = df[["a", "b"]]
df2["c"] = 1              # SettingWithCopyWarning

# Good — work on a copy explicitly
df2 = df[["a", "b"]].copy()
df2["c"] = 1

# ── Use vectorised operations — never row-loop ─────────────────────────────────

# Slow
for i, row in df.iterrows():
    df.at[i, "new"] = row["a"] + row["b"]

# Fast
df["new"] = df["a"] + df["b"]

# ── eval() and query() for large DataFrames ────────────────────────────────────

# Avoids creating intermediate arrays
df.eval("profit = revenue - cost", inplace=True)
df.query("profit > 0 and region == 'North'")

# ── Memory usage report ────────────────────────────────────────────────────────

mem = df.memory_usage(deep=True)
print(mem)
print(f"Total: {mem.sum() / 1e6:.2f} MB")

# ── convert_dtypes() — auto-upgrade to nullable types ─────────────────────────

df_better = df.convert_dtypes()   # int64→Int64, object→StringDtype, etc.
```

---

## 22. Apache Arrow Backend

Pandas and Apache Arrow are related but distinct — Pandas uses NumPy as its default internal storage engine. Arrow integration is opt-in.

### Default backend: NumPy

```python
df = pd.DataFrame({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]})
print(df["a"].values)         # numpy.ndarray
print(type(df["a"].array))    # PandasArray (wraps NumPy)
```

### Where Arrow IS used in Pandas by default

| Operation | Arrow involvement |
|---|---|
| `pd.read_parquet()` | PyArrow (or fastparquet) reads the file; result converted to NumPy-backed DataFrame |
| `pd.read_feather()` | PyArrow reads; result converted |
| `pd.read_orc()` | PyArrow reads; result converted |
| `df.to_parquet()` | PyArrow writes the Parquet file |
| `pa.Table.from_pandas(df)` | Explicit conversion — usually copies data |

### ArrowDtype — opt-in Arrow storage per column (Pandas 2.0+)

```python
import pyarrow as pa
import pandas as pd

# Create a Series backed by Arrow memory
s = pd.Series([1, 2, None, 4], dtype=pd.ArrowDtype(pa.int32()))
print(s.dtype)   # int32[pyarrow]

# Create a full DataFrame with Arrow backend
df_arrow = pd.read_parquet("data.parquet", dtype_backend="pyarrow")
df_arrow = pd.read_csv("data.csv",         dtype_backend="pyarrow")

# Or convert an existing DataFrame
df_numpy = pd.DataFrame({"a": [1,2,3], "b": ["x","y","z"]})
df_arrow = df_numpy.convert_dtypes(dtype_backend="pyarrow")

print(df_arrow.dtypes)
# a      int64[pyarrow]
# b     string[pyarrow]
```

### NumPy backend vs ArrowDtype — comparison

| Feature | NumPy backend (default) | ArrowDtype (opt-in) |
|---|---|---|
| Native null support for integers | No — null forces cast to `float64` | Yes — true integer nulls (`<NA>`) |
| Native null support for strings | No — stored as `object` dtype | Yes |
| Nested types (lists, structs) | Not supported | Supported via `pa.list_()`, `pa.struct()` |
| Memory efficiency | Moderate | Better — bit-packed nulls, no object overhead |
| Large string support | Limited (`object`) | `large_string` type |
| Zero-copy to Arrow Table | Needs conversion with copy | Near zero-copy |
| Computation speed | NumPy-optimised | PyArrow compute functions |

### Conversion between Pandas and Arrow

| Direction | Method | Copies? |
|---|---|---|
| Pandas → Arrow Table | `pa.Table.from_pandas(df)` | Usually yes (NumPy-backed) |
| Pandas ArrowDtype → Arrow Table | `pa.Table.from_pandas(df)` | Near zero-copy |
| Arrow Table → Pandas | `table.to_pandas()` | Usually yes |
| Arrow Table → Pandas (zero-copy) | `table.to_pandas(zero_copy_only=True)` | Raises if copy needed |
| Arrow Array → NumPy | `arr.to_numpy(zero_copy_only=True)` | Zero-copy if contiguous |

### Pandas vs Arrow — which to use

| Use Arrow when | Use Pandas when |
|---|---|
| Sharing data between languages or processes | Staying entirely within Python |
| Sending data over a network (Flight) | Interactive exploration and analysis |
| Reading/writing Parquet at scale | Complex string operations and regex |
| Columnar compute across large datasets | Excel, HTML, SQL table I/O |
| Zero-copy handoff to DuckDB, Spark, Polars | Rich datetime and period support |
| Large datasets that exceed RAM | Familiar `.groupby()` / `.merge()` API |

---

## 23. Quick Reference

```mermaid
mindmap
  root(("Pandas"))
    Create
      pd.DataFrame from dict / list
      pd.read_csv / read_parquet / read_excel
      pd.Series from list or dict
      pd.date_range
    Inspect
      df.head / tail / sample
      df.shape / dtypes / info
      df.describe / isnull().sum
    Select
      df["col"]  single column
      df[["a","b"]]  multiple columns
      df.loc label-based
      df.iloc position-based
    Filter
      df[df["col"] > 0]
      df.query
      df["col"].isin / between
      df.loc with boolean mask
    Clean
      df.dropna / fillna
      df.drop_duplicates
      df["col"].str.strip / lower
      df["col"].astype
    Transform
      vectorised assignment
      df.assign
      df["col"].map / apply
      df.apply axis=1
    GroupBy
      groupby().sum / mean / count
      groupby().agg named aggregations
      groupby().transform
      groupby().filter
    Merge & Join
      pd.merge on / how
      pd.concat ignore_index
      df.join
    Reshape
      pd.pivot_table
      df.melt id_vars
      df.stack / unstack
      df.explode
    Time Series
      pd.to_datetime
      df.resample("D").sum
      rolling / ewm
      shift / diff
    I/O
      read_csv / to_csv
      read_parquet / to_parquet
      read_excel / to_excel
      read_sql / to_sql
    Arrow Backend
      dtype_backend="pyarrow"
      pd.ArrowDtype
      convert_dtypes
```
