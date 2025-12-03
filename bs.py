import pandas as pd
import numpy as np
import duckdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ====== KNOBS ======
DB_PATH = "/store/talent/db/talent.db"
TABLE   = "fullTalentData"
SAMPLE_ROWS = 200_000           # adjust for RAM
ENG_INT = ["BY_INT04", "BY_INT44", "BY_INT47", "BY_INT67", "BY_INT89"]  # engineering-interest items (5 = very interested)
# ===================

con = duckdb.connect(DB_PATH)

# 1) Get BY_IN% feature columns (interest items) present in the table
byin_cols = [r[0] for r in con.execute(f"""
  SELECT column_name
  FROM (SHOW {TABLE})
  WHERE REGEXP_MATCHES(column_name, '^BY_R')
     OR column_name IN ({", ".join("'" + c + "'" for c in ENG_INT)})
  ORDER BY column_name
""").fetchall()]

# 2) Also check which ENG_INT columns exist (for Y and to exclude from X)
eng_cols_present = [c for c in ENG_INT if c in byin_cols]
if not eng_cols_present:
    raise ValueError(f"None of ENG_INT columns found in table: {ENG_INT}")

# 3) Build SELECT: only the needed columns + BY_SEX to filter girls
need_cols = sorted(set(byin_cols + ["BY_SEX"]))
cols_sql  = ", ".join(f'"{c}"' for c in need_cols)

df = con.execute(f"""
  SELECT {cols_sql}
  FROM {TABLE}
  WHERE BY_SEX = 2                -- girls only
""").df()

#USING SAMPLE {SAMPLE_ROWS} ROWS

con.close()
print(f"Loaded (girls-only) shape: {df.shape}")

# 4) Create binary target: interested in engineering if ANY ENG_INT == 5
y = (df[eng_cols_present].eq(5).any(axis=1)).astype(int)
df["__y__"] = y

# 5) Build X from BY_IN% features, EXCLUDING ENG_INT (no leakage)
X = df[sorted(set(byin_cols) - set(eng_cols_present))].copy()

# 6) Clean: drop all-empty/constant cols, impute mean, scale
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=True) > 1]
X = X.astype("float64", errors="ignore").fillna(X.mean())

# Align X and y (drop any rows with missing y if present)
mask = df["__y__"].notna()
X = X.loc[mask]
y = df.loc[mask, "__y__"].astype(int)

# 7) Train/test & logistic regression
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_z = scaler.fit_transform(X_train)
X_test_z  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_z, y_train)

print("Accuracy:", round(lr.score(X_test_z, y_test), 3))

# 8) Top features (by absolute coefficient magnitude)
coef = pd.Series(lr.coef_[0], index=X.columns)
top = (coef.reindex(coef.abs().sort_values(ascending=False).index)
           .head(20)
           .rename("coef"))
print("\nTop 20 features by |coef| (sign shows direction):")
print(top.to_string())

coef.to_csv("outputs/BY_R", header=False)
print("Wrote outputs/BY_R")