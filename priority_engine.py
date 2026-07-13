"""
priority_engine.py
───────────────────
Deterministic priority scoring + worker assignment.

Extracted out of app.py so app.py's manual "File a Complaint" form and the
LLM chat assistant (llm_workflow.py) both compute priority and assign
workers with exactly the same rules — no duplicated logic to drift apart.
"""

import random
from datetime import datetime

# ── Department priority weight (1-3), covers every class the RandomForest
#    model can predict ──────────────────────────────────────────────────
DEPT_PRIORITY = {
    "Electrical": 3, "Road Maintenance(Engg)": 3, "Water Crisis": 3,
    "Storm  Water Drain(SWD)": 3, "Storm Water Drain(SWD)": 3,
    "Solid Waste (Garbage) Related": 2, "Sanitation": 2, "Health Dept": 2,
    "Forest": 2, "Lakes": 2, "Parks and Play grounds": 2, "veterinary": 2,
    "Major Roads": 3, "Road Infrastructure": 3, "Traffic Engineer Cell (TEC)": 2,
    "Advertisement": 1, "Town Planning": 1, "Revenue Department": 1,
    "E khata / Khata services": 1, "Property Tax services": 1,
    "Welfare Schemes": 1, "Education": 1, "Estate": 1, "Markets": 1,
    "CORONA COVID19": 3, "Information Technology": 1, "Call Center": 1,
    "Indira Canteen": 1, "Projects Central": 2, "Plastic": 2,
    "Optical Fiber Cables (OFC)": 2, "BBMP Election Branch": 1, "Others": 1,
}

# Month priority (monsoon/summer peaks = higher)
MONTH_PRIORITY = {1: 2, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 2, 10: 2, 11: 1, 12: 2}


def hour_priority(hour):
    """Late-night complaints (10pm-6am) escalate; peak hours are medium; daytime is low."""
    if hour < 6 or hour >= 22:
        return 3
    if 6 <= hour < 9 or 18 <= hour < 22:
        return 2
    return 1


def compute_priority(dept, dt: datetime):
    ds = DEPT_PRIORITY.get(dept, 1)
    ms = MONTH_PRIORITY.get(dt.month, 1)
    hs = hour_priority(dt.hour)
    total = ds + ms + hs
    if total >= 8:
        return "High", total, ds, ms, hs
    elif total >= 5:
        return "Medium", total, ds, ms, hs
    else:
        return "Low", total, ds, ms, hs


def build_role_map(employees_df):
    """Department -> the most common worker designation in that department.
    Purely for display (dashboards, the priority simulator's department
    list) — derived straight from Employees.csv, not guessed by hand.
    """
    if employees_df is None or employees_df.empty:
        return {}
    return (
        employees_df.groupby("department")["role"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
        .to_dict()
    )


def assign_workers(dept, priority_label, employees_df, n=3):
    """Pick workers for a complaint, straight off Employees.csv's own
    Department column — no department->role guesswork needed since every
    employee already lists which department they belong to.
    """
    eligible = employees_df[employees_df["department"] == dept]
    if eligible.empty:
        # Nobody on record for this exact department — fall back to the
        # whole roster rather than returning no one.
        eligible = employees_df

    # Prefer workers who are free and have headroom under their max workload.
    workable = eligible[(eligible["status"] == "Available") & (eligible["workload"] < eligible["max_workload"])]
    pool = workable if not workable.empty else eligible
    pool = pool.sort_values("workload")  # least-loaded first

    if priority_label == "High":
        workers = pool.head(n)  # fastest available responders
    elif priority_label == "Medium":
        top = pool.head(min(n * 2, len(pool)))
        workers = top.sample(min(n, len(top)), random_state=42) if len(top) else top
    else:
        workers = pool.sample(min(2, len(pool)), random_state=7) if len(pool) else pool

    if not workers.empty:
        role = workers["role"].mode().iat[0]
    elif not eligible.empty:
        role = eligible["role"].mode().iat[0]
    else:
        role = "Unassigned"
    return workers, role


def ticket_id():
    return f"SUCAS-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
