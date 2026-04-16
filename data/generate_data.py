"""
generate_data.py
Generates a realistic synthetic sales dataset for InsightForge BI Assistant.
Run once to create sales_data.csv before launching the app.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────────
N_ROWS        = 2_000
START_DATE    = datetime(2022, 1, 1)
END_DATE      = datetime(2024, 12, 31)

PRODUCTS = {
    "Laptop Pro":       ("Electronics",   1200, 1800),
    "Wireless Mouse":   ("Electronics",     25,   60),
    "Office Chair":     ("Furniture",      150,  400),
    "Standing Desk":    ("Furniture",      300,  700),
    "Notebook Set":     ("Stationery",      10,   25),
    "Pen Collection":   ("Stationery",       5,   20),
    "Headphones Elite": ("Electronics",    100,  350),
    "Monitor 27in":     ("Electronics",    250,  550),
    "Ergonomic Pillow": ("Furniture",       40,   90),
    "Planner 2024":     ("Stationery",      15,   35),
}

REGIONS     = ["North", "South", "East", "West", "Central"]
SEGMENTS    = ["Enterprise", "SMB", "Startup", "Individual"]
GENDERS     = ["Male", "Female", "Non-binary"]
AGE_GROUPS  = ["18-24", "25-34", "35-44", "45-54", "55+"]
CHANNELS    = ["Online", "Retail", "Direct Sales", "Partner"]

# ── Build rows ─────────────────────────────────────────────────────────────────
date_range = (END_DATE - START_DATE).days

rows = []
for i in range(N_ROWS):
    product, (category, low, high) = random.choice(list(PRODUCTS.items()))
    region    = random.choice(REGIONS)
    segment   = random.choice(SEGMENTS)
    gender    = random.choice(GENDERS)
    age_group = random.choice(AGE_GROUPS)
    channel   = random.choice(CHANNELS)

    order_date = START_DATE + timedelta(days=random.randint(0, date_range))
    quantity   = random.randint(1, 20)
    unit_price = round(random.uniform(low, high), 2)
    discount   = round(random.choice([0, 0.05, 0.10, 0.15, 0.20]), 2)
    revenue    = round(quantity * unit_price * (1 - discount), 2)
    cost       = round(revenue * random.uniform(0.45, 0.70), 2)
    profit     = round(revenue - cost, 2)

    rows.append({
        "order_id":    f"ORD-{10000 + i}",
        "order_date":  order_date.strftime("%Y-%m-%d"),
        "product":     product,
        "category":    category,
        "region":      region,
        "segment":     segment,
        "gender":      gender,
        "age_group":   age_group,
        "channel":     channel,
        "quantity":    quantity,
        "unit_price":  unit_price,
        "discount":    discount,
        "revenue":     revenue,
        "cost":        cost,
        "profit":      profit,
    })

df = pd.DataFrame(rows).sort_values("order_date").reset_index(drop=True)
df.to_csv("data/sales_data.csv", index=False)
print(f"✅  Generated {len(df):,} rows → data/sales_data.csv")
print(df.head())
