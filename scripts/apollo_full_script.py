"""
Apollo Hospitals -- Appointment No-Show & Patient Engagement Analysis
Standalone script version of the final EDA notebook.

Business rules applied throughout:
    1. No-show rate = No-Show / (Completed + No-Show + Cancelled)  -> always use `resolved`
    2. Revenue                                                      -> always use `completed`
    3. Quality metrics (wait time, duration, satisfaction, utilisation) -> always use `completed`
    4. "None" in chronic condition / membership type / insurance provider / cancellation
       reason means "Not Applicable" and is kept as its own category, never dropped.

Run from the repository root so the relative `data/` path below resolves correctly:
    python apollo_full_script.py
"""

# ============================================================
# 1. Project Setup
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 150)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 4.5)

DATA_DIR = Path("data")
FACT_PATH = DATA_DIR / "apollo_appointments_fact.csv"
DIM_PATH = DATA_DIR / "apollo_doctors_dim.csv"

# ============================================================
# 2. Load the Data
# ============================================================
# keep_default_na=False, na_values=[""] -> "None" in columns like
# patient_chronic_condition means "does not apply", not "missing". Letting
# pandas auto-convert it to NaN would wrongly treat real information as missing.
fact = pd.read_csv(FACT_PATH, keep_default_na=False, na_values=[""])
doctors = pd.read_csv(DIM_PATH, keep_default_na=False, na_values=[""])

print("Fact table shape   :", fact.shape)
print("Doctors table shape:", doctors.shape)

# ============================================================
# 3. Data Understanding & Inspection
# ============================================================
print("\n--- fact.info() ---")
fact.info()

print("\n--- doctors.info() ---")
doctors.info()

# Nulls -- expect nonzero ONLY in the 4 quality columns, and only because those
# appointments never happened (no wait time / duration / satisfaction / utilisation to record)
print("\n--- Nulls in fact table ---")
print(fact.isna().sum().loc[lambda s: s > 0])

# Confirm "None" survived as a real category, not NaN
print("\n--- patient_chronic_condition value counts ---")
print(fact["patient_chronic_condition"].value_counts())

# Duplicates
print("\n--- Duplicate checks ---")
print("Duplicate appointment_id:", fact["appointment_id"].duplicated().sum())
print("Full duplicate rows     :", fact.duplicated().sum())
print("Duplicate doctor_id     :", doctors["doctor_id"].duplicated().sum())

# Outcome counts -- the core split we will keep referring back to
print("\n--- appointment_status value counts ---")
print(fact["appointment_status"].value_counts())

# Confirm nulls in the quality columns line up exactly with non-Completed rows
quality_cols = ["wait_time_minutes", "consultation_duration_min",
                "patient_satisfaction_score", "doctor_utilization_pct"]
print("\n--- Null counts in quality columns, by appointment_status ---")
print(fact.groupby("appointment_status")[quality_cols].apply(lambda x: x.isna().sum()))

# What this tells us: nulls in the 4 quality columns appear only for Cancelled,
# No-Show and Scheduled rows -- never for Completed. This confirms the business
# rule: quality metrics only exist for appointments that actually happened.
# No imputation needed; we simply filter to Completed before using these columns.

# ============================================================
# 4. Data Cleaning & Preparation
# ============================================================
df = fact.copy()

# Convert dates
df["appointment_date"] = pd.to_datetime(df["appointment_date"])
df["booking_date"] = pd.to_datetime(df["booking_date"])

print("\n--- Date conversion checks ---")
print("Unparseable appointment dates:", df["appointment_date"].isna().sum())
print("Unparseable booking dates    :", df["booking_date"].isna().sum())
print("Date range:", df["appointment_date"].min().date(), "to", df["appointment_date"].max().date())
print("Bookings made AFTER the appointment date (should be 0):",
      (df["booking_date"] > df["appointment_date"]).sum())

# --- 4.1 Business-rule DataFrames ---
# df        -> all appointments: volume, outcome split, channel mix, age distribution
# resolved  -> drops Scheduled:  every no-show rate
# completed -> Completed only:  every revenue and quality metric
resolved = df[df["appointment_status"] != "Scheduled"].copy()
completed = df[df["appointment_status"] == "Completed"].copy()

print("\n--- Business-rule DataFrame sizes ---")
print("df        :", df.shape[0], "rows")
print("resolved  :", resolved.shape[0], "rows  (75,000 minus Scheduled)")
print("completed :", completed.shape[0], "rows  (Completed only)")

# Data note (December 2024): all 2,348 Scheduled rows fall in December 2024, and
# December 2024 contains ONLY Scheduled rows. That month has no completed outcomes
# yet, so any outcome-based trend (no-show rate, revenue) naturally has no data
# point there -- this is correct, not a bug.
dec_2024 = df[(df["appointment_year"] == 2024) & (df["appointment_month"] == 12)]
print("\n--- December 2024 status check ---")
print(dec_2024["appointment_status"].value_counts())

# --- 4.2 Derived columns ---
# Only columns that are genuinely needed for a business question are created --
# everything else (month, quarter, weekday, time slot, lead days, age group)
# already exists in the file, so we reuse it as-is.

# Month / quarter labels for trend charts
df["year_month"] = df["appointment_date"].dt.to_period("M").astype(str)
df["year_quarter"] = df["appointment_year"].astype(str) + "-" + df["appointment_quarter"]

# Weekend flag (Saturday/Sunday, day_of_week 5 and 6)
df["is_weekend"] = df["appointment_day_of_week"] >= 5

# Lead-time buckets
lead_bins = [-1, 0, 1, 3, 7, 14, 30]
lead_labels = ["0 (Same day)", "1 day", "2-3 days", "4-7 days", "8-14 days", "15-30 days"]
df["lead_time_group"] = pd.cut(df["booking_lead_days"], bins=lead_bins, labels=lead_labels)

# Prior no-show history buckets (3+ is a thin tail, so grouped together)
df["prior_no_show_group"] = pd.cut(df["patient_prior_no_shows"], bins=[-1, 0, 1, 2, 100],
                                    labels=["0", "1", "2", "3+"])

# Chronic condition as a clean Yes/No flag
df["chronic_flag"] = np.where(df["patient_chronic_condition"] == "None", "No", "Yes")

# Duration bands (Completed range is 5-48 minutes)
df["duration_band"] = pd.cut(df["consultation_duration_min"], bins=[0, 10, 20, 30, 50],
                              labels=["<=10 min", "11-20 min", "21-30 min", "31+ min"])

# Propagate the same derived columns onto resolved / completed so we don't rebuild them twice
resolved = resolved.merge(df[["appointment_id", "year_month", "year_quarter", "is_weekend",
                               "lead_time_group", "prior_no_show_group", "chronic_flag"]],
                           on="appointment_id", how="left")
completed = completed.merge(df[["appointment_id", "duration_band"]], on="appointment_id", how="left")

print("\nDerived columns added.")
print(df[["year_month", "year_quarter", "is_weekend", "lead_time_group",
          "prior_no_show_group", "chronic_flag"]].head(3))

# ============================================================
# 5. Doctor Table Join
# ============================================================
# We only need attributes that exist EXCLUSIVELY in the doctors table:
# experience_years, rating, total_reviews, qualification, and the doctor's
# STANDARD fee (renamed doctor_std_fee).
#
# specialty / city / state / hospital_name are deliberately left out -- they're
# already in the fact table and identical, so re-merging would just create
# confusing duplicate columns. available_days is also left out because it does
# not reliably match the actual appointment days in this dataset.
#
# Join type: LEFT join, fact -> doctors, many-to-one. The fact table is primary;
# every one of the 75,000 appointments must be kept.
doc_cols = ["doctor_id", "doctor_name", "qualification", "experience_years",
            "rating", "total_reviews", "avg_slot_duration_min",
            "accepts_insurance", "teleconsult_enabled", "consultation_fee"]

doc_small = doctors[doc_cols].rename(columns={"consultation_fee": "doctor_std_fee"})

df = df.merge(doc_small, on="doctor_id", how="left", validate="m:1", indicator=True)

print("\n--- Doctor join validation ---")
print("Rows after join       :", df.shape[0], "(should still be 75,000)")
print("Merge match summary   :")
print(df["_merge"].value_counts())
print("Nulls in doctor columns:", df[doc_cols[1:]].isna().sum().sum())
print("appointment_id still unique:", df["appointment_id"].is_unique)

df = df.drop(columns="_merge")

# Validation passed: row count unchanged, every appointment matched a doctor, no
# nulls introduced, and appointment_id is still unique (a broken join would have
# duplicated rows). We also carry experience_years and doctor_std_fee onto
# `completed`, since Q21 (experience vs fee) uses Completed appointments.
# IMPORTANT: this merge runs AFTER the doctor join above, so df already has the
# doctor columns available to pull from -- this ordering avoids the empty/stale
# `completed` DataFrame issue.
completed = completed.merge(df[["appointment_id", "experience_years", "doctor_std_fee"]],
                             on="appointment_id", how="left")
print("\ncompleted shape (after doctor-column merge):", completed.shape)

# ============================================================
# 6. A Small Helper for No-Show Rates
# ============================================================
# Every no-show question follows the same pattern: group `resolved` by some
# column, take the mean of `no_show_flag` (which is exactly
# No-Show count / (Completed + No-Show + Cancelled) for that group), and show
# the group size alongside it so we never trust a rate built on very few rows.
def no_show_rate_by(column, data=resolved):
    """Returns no-show rate (%) and row count per category of `column`, sorted descending by rate."""
    out = data.groupby(column, observed=True).agg(
        no_show_rate=("no_show_flag", "mean"),
        appointments=("no_show_flag", "size")
    )
    out["no_show_rate"] = (out["no_show_rate"] * 100).round(1)
    return out.sort_values("no_show_rate", ascending=False)

# ============================================================
# 7. Business Rules (recap before analysis)
# ============================================================
# 1. No-show rate = No-Show / (Completed + No-Show + Cancelled) -> always use `resolved`
# 2. Revenue -> always use `completed`
# 3. Quality metrics (wait time, duration, satisfaction, utilisation) -> always use `completed`
# 4. "None" in chronic condition / membership type / insurance provider / cancellation
#    reason = "Not Applicable", kept as its own category


# ============================================================
# SECTION A -- Business Overview
# ============================================================

# ---- Q1. How is appointment volume trending across months and quarters (2022-2024)? ----
# Metric: count of appointments per month / per quarter. Filter: none -- this is
# booked volume, so every row counts (including the December 2024 Scheduled batch).
monthly_volume = df.groupby("year_month").size()

plt.figure(figsize=(12, 4.5))
monthly_volume.plot(kind="line", marker="o", markersize=3)
plt.title("Monthly Appointment Volume (2022-2024)")
plt.xlabel("Month")
plt.ylabel("Appointments")
plt.xticks(rotation=90, fontsize=7)
plt.tight_layout()
plt.show()

print("\n--- Q1: monthly_volume.describe() ---")
print(monthly_volume.describe())

quarterly_volume = df.groupby(["appointment_year", "appointment_quarter"]).size().unstack()

quarterly_volume.T.plot(kind="bar", figsize=(9, 4.5))
plt.title("Quarterly Appointment Volume by Year")
plt.xlabel("Quarter")
plt.ylabel("Appointments")
plt.legend(title="Year")
plt.tight_layout()
plt.show()

print("\n--- Q1: quarterly_volume ---")
print(quarterly_volume)

# ---- Q2. What is the split of outcomes (Completed / No-Show / Cancelled / Scheduled)? ----
# Metric: count and % of appointment_status. Filter: none, all 75,000 rows.
status_split = df["appointment_status"].value_counts()
status_pct = (df["appointment_status"].value_counts(normalize=True) * 100).round(1)

print("\n--- Q2: outcome split ---")
print(pd.DataFrame({"count": status_split, "pct": status_pct}))

plt.figure(figsize=(5.5, 5.5))
colors = sns.color_palette("Set2", len(status_split))
plt.pie(status_split, labels=status_split.index, autopct="%1.1f%%", startangle=90,
        colors=colors, wedgeprops=dict(width=0.45))
plt.title("Appointment Outcome Split")
plt.tight_layout()
plt.show()

# ---- Q3. Which booking channels drive the highest volume? ----
# Metric: count and % share per booking_channel. Filter: none.
channel_volume = df["booking_channel"].value_counts()
channel_pct = (df["booking_channel"].value_counts(normalize=True) * 100).round(1)

print("\n--- Q3: booking channel volume ---")
print(pd.DataFrame({"count": channel_volume, "pct": channel_pct}))

plt.figure(figsize=(8, 4.5))
sns.barplot(x=channel_volume.values, y=channel_volume.index, hue=channel_volume.index,
            palette="Blues_r", legend=False)
plt.title("Booking Volume by Channel")
plt.xlabel("Appointments")
plt.ylabel("")
plt.tight_layout()
plt.show()


# ============================================================
# SECTION B -- No-Show Analysis
# ============================================================

# ---- Q4. Which specialties and cities have the highest / lowest no-show rates? ----
# Metric: no-show rate per group. Filter: resolved (Scheduled excluded).
specialty_ns = no_show_rate_by("specialty")
print("\n--- Q4: no-show rate by specialty ---")
print(specialty_ns)

plt.figure(figsize=(8, 6))
sns.barplot(x="no_show_rate", y=specialty_ns.index, data=specialty_ns,
            hue=specialty_ns.index, palette="Reds_r", legend=False)
plt.title("No-Show Rate by Specialty")
plt.xlabel("No-Show Rate (%)")
plt.ylabel("")
plt.tight_layout()
plt.show()

city_ns = no_show_rate_by("city")
print("\n--- Q4: no-show rate by city ---")
print(city_ns)

plt.figure(figsize=(8, 6))
sns.barplot(x="no_show_rate", y=city_ns.index, data=city_ns,
            hue=city_ns.index, palette="Oranges_r", legend=False)
plt.title("No-Show Rate by City")
plt.xlabel("No-Show Rate (%)")
plt.ylabel("")
plt.tight_layout()
plt.show()

# ---- Q5. How does booking lead time affect no-show probability? ----
# Metric: no-show rate per lead_time_group. Filter: resolved.
lead_ns = no_show_rate_by("lead_time_group").reindex(
    ["0 (Same day)", "1 day", "2-3 days", "4-7 days", "8-14 days", "15-30 days"]
)
print("\n--- Q5: no-show rate by lead time ---")
print(lead_ns)

plt.figure(figsize=(8, 4.5))
sns.barplot(x=lead_ns.index, y="no_show_rate", data=lead_ns,
            hue=lead_ns.index, palette="Purples", legend=False)
plt.title("No-Show Rate by Booking Lead Time")
plt.xlabel("Lead Time")
plt.ylabel("No-Show Rate (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# ---- Q6. Do evening slots, weekends, and longer lead times see higher dropout? ----
# Three separate checks -- each hypothesis stands or falls on its own.
time_of_day_ns = no_show_rate_by("time_of_day")
print("\n--- Q6: no-show rate by time of day ---")
print(time_of_day_ns)

weekend_ns = no_show_rate_by("is_weekend")
weekend_ns.index = weekend_ns.index.map({True: "Weekend", False: "Weekday"})
print("\n--- Q6: no-show rate, weekday vs weekend ---")
print(weekend_ns)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

sns.barplot(x=time_of_day_ns.index, y="no_show_rate", data=time_of_day_ns,
            hue=time_of_day_ns.index, palette="crest", legend=False, ax=axes[0])
axes[0].set_title("No-Show Rate by Time of Day")
axes[0].set_xlabel("")
axes[0].set_ylabel("No-Show Rate (%)")

sns.barplot(x=weekend_ns.index, y="no_show_rate", data=weekend_ns,
            hue=weekend_ns.index, palette="crest", legend=False, ax=axes[1])
axes[1].set_title("No-Show Rate: Weekday vs Weekend")
axes[1].set_xlabel("")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()

# ---- Q7. Which appointment types and booking channels carry the most no-show risk? ----
# Metric: no-show rate per appointment_type and per booking_channel. Filter: resolved.
type_ns = no_show_rate_by("appointment_type")
print("\n--- Q7: no-show rate by appointment type ---")
print(type_ns)

channel_ns = no_show_rate_by("booking_channel")
print("\n--- Q7: no-show rate by booking channel ---")
print(channel_ns)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

sns.barplot(x=type_ns.index, y="no_show_rate", data=type_ns,
            hue=type_ns.index, palette="mako", legend=False, ax=axes[0])
axes[0].set_title("No-Show Rate by Appointment Type")
axes[0].set_xlabel("")
axes[0].set_ylabel("No-Show Rate (%)")

sns.barplot(x=channel_ns.index, y="no_show_rate", data=channel_ns,
            hue=channel_ns.index, palette="mako", legend=False, ax=axes[1])
axes[1].set_title("No-Show Rate by Booking Channel")
axes[1].set_xlabel("")
axes[1].set_ylabel("")
axes[1].tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.show()


# ============================================================
# SECTION C -- Reminder & Engagement Effectiveness
# ============================================================

# ---- Q8. How much does reminder type reduce no-show rate vs no reminder? ----
# Metric: no-show rate per reminder_type, compared to the "No Reminder" baseline. Filter: resolved.
reminder_order = ["No Reminder", "SMS Only", "SMS + WhatsApp", "SMS + WhatsApp + Call"]
reminder_ns = no_show_rate_by("reminder_type").reindex(reminder_order)

baseline = reminder_ns.loc["No Reminder", "no_show_rate"]
reminder_ns["pp_change_vs_no_reminder"] = (reminder_ns["no_show_rate"] - baseline).round(1)
print("\n--- Q8: no-show rate by reminder type ---")
print(reminder_ns)

plt.figure(figsize=(8, 4.5))
sns.barplot(x=reminder_ns.index, y="no_show_rate", data=reminder_ns,
            hue=reminder_ns.index, palette="YlGnBu", legend=False)
plt.title("No-Show Rate by Reminder Type")
plt.xlabel("")
plt.ylabel("No-Show Rate (%)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# ---- Q9. Does prior no-show history predict future no-show behaviour? ----
# Metric: no-show rate per prior_no_show_group. Filter: resolved.
prior_ns = no_show_rate_by("prior_no_show_group").reindex(["0", "1", "2", "3+"])
print("\n--- Q9: no-show rate by prior no-show count ---")
print(prior_ns)

plt.figure(figsize=(7, 4.5))
sns.barplot(x=prior_ns.index, y="no_show_rate", data=prior_ns,
            hue=prior_ns.index, palette="Reds", legend=False)
plt.title("No-Show Rate by Prior No-Show Count")
plt.xlabel("Prior No-Shows")
plt.ylabel("No-Show Rate (%)")
plt.tight_layout()
plt.show()

# ---- Q10. Do members and repeat patients show better attendance? ----
# Metric: no-show rate for member vs non-member, and repeat vs first-time. Filter: resolved.
member_ns = no_show_rate_by("apollo_member")
print("\n--- Q10: no-show rate, member vs non-member ---")
print(member_ns)

repeat_ns = no_show_rate_by("repeat_visit")
print("\n--- Q10: no-show rate, repeat vs first-time ---")
print(repeat_ns)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

sns.barplot(x=member_ns.index, y="no_show_rate", data=member_ns,
            hue=member_ns.index, palette="BuGn", legend=False, ax=axes[0])
axes[0].set_title("No-Show Rate: Apollo Member vs Not")
axes[0].set_xlabel("")
axes[0].set_ylabel("No-Show Rate (%)")

sns.barplot(x=repeat_ns.index, y="no_show_rate", data=repeat_ns,
            hue=repeat_ns.index, palette="BuGn", legend=False, ax=axes[1])
axes[1].set_title("No-Show Rate: Repeat vs First-Time Visit")
axes[1].set_xlabel("")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()


# ============================================================
# SECTION D -- Patient & Demographic Segmentation
# ============================================================

# ---- Q11. How do age group, gender, and chronic condition status influence no-show likelihood? ----
# Metric: no-show rate by age_group, patient_gender, and chronic-condition flag. Filter: resolved.
age_order = sorted(resolved["age_group"].unique(),
                    key=lambda x: resolved.loc[resolved["age_group"] == x, "patient_age"].mean())
age_ns = no_show_rate_by("age_group").reindex(age_order)
print("\n--- Q11: no-show rate by age group ---")
print(age_ns)

gender_ns = no_show_rate_by("patient_gender")
print("\n--- Q11: no-show rate by gender ---")
print(gender_ns)

chronic_ns = no_show_rate_by("chronic_flag")
print("\n--- Q11: no-show rate by chronic condition flag ---")
print(chronic_ns)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

sns.barplot(x=age_ns.index, y="no_show_rate", data=age_ns,
            hue=age_ns.index, palette="viridis", legend=False, ax=axes[0])
axes[0].set_title("No-Show Rate by Age Group")
axes[0].set_xlabel("")
axes[0].set_ylabel("No-Show Rate (%)")
axes[0].tick_params(axis="x", rotation=30)

sns.barplot(x=gender_ns.index, y="no_show_rate", data=gender_ns,
            hue=gender_ns.index, palette="viridis", legend=False, ax=axes[1])
axes[1].set_title("No-Show Rate by Gender")
axes[1].set_xlabel("")
axes[1].set_ylabel("")

sns.barplot(x=chronic_ns.index, y="no_show_rate", data=chronic_ns,
            hue=chronic_ns.index, palette="viridis", legend=False, ax=axes[2])
axes[2].set_title("No-Show Rate: Chronic Condition vs Not")
axes[2].set_xlabel("")
axes[2].set_ylabel("")

plt.tight_layout()
plt.show()

# ---- Q12. Which visit reasons are most common, and are some associated with higher dropout? ----
# Metric: volume (all rows) and no-show rate (resolved) per visit_reason.
reason_volume = df["visit_reason"].value_counts()

plt.figure(figsize=(8, 7))
sns.barplot(x=reason_volume.values, y=reason_volume.index, hue=reason_volume.index,
            palette="Blues_r", legend=False)
plt.title("Appointment Volume by Visit Reason")
plt.xlabel("Appointments")
plt.ylabel("")
plt.tight_layout()
plt.show()

reason_ns = no_show_rate_by("visit_reason")
print("\n--- Q12: no-show rate by visit reason ---")
print(reason_ns)

plt.figure(figsize=(8, 7))
sns.barplot(x="no_show_rate", y=reason_ns.index, data=reason_ns,
            hue=reason_ns.index, palette="Reds_r", legend=False)
plt.title("No-Show Rate by Visit Reason")
plt.xlabel("No-Show Rate (%)")
plt.ylabel("")
plt.tight_layout()
plt.show()

# ---- Q13. What does the patient age distribution look like within Paediatrics, Gynaecology, and Psychiatry? ----
# Filter: these 3 specialties, all statuses (age is known regardless of outcome).
three_specialties = ["Paediatrics", "Gynaecology", "Psychiatry"]
age_subset = df[df["specialty"].isin(three_specialties)]

print("\n--- Q13: patient age distribution, by specialty ---")
print(age_subset.groupby("specialty")["patient_age"].describe().round(1))

plt.figure(figsize=(8, 5))
sns.boxplot(x="specialty", y="patient_age", data=age_subset,
            hue="specialty", palette="Set3", legend=False)
plt.title("Patient Age Distribution by Specialty")
plt.xlabel("")
plt.ylabel("Patient Age")
plt.tight_layout()
plt.show()

# Data observation: Paediatrics should skew young by definition. A small number
# of patients under 13 also show up in Gynaecology and Psychiatry, which is
# unusual for these specialties -- called out here as a data quality note rather
# than silently filtered out.


# ============================================================
# SECTION E -- Financial Performance
# ============================================================

# ---- Q14. How much revenue is being lost to no-shows? ----
# No-Show rows have revenue_realized = 0 by design (nothing was actually
# collected), so this has to be ESTIMATED, not summed directly. Two views:
#   1. Billed-value-at-risk -- the listed consultation_fee for every No-Show appointment
#   2. Modeled loss -- No-Show count x average actual revenue per Completed appointment, by specialty
no_show_rows = resolved[resolved["appointment_status"] == "No-Show"]

billed_value_at_risk = no_show_rows["consultation_fee"].sum()
print(f"\n--- Q14: billed value at risk ---")
print(f"Billed value at risk (listed fee of all No-Show appointments): Rs {billed_value_at_risk:,.0f}")

avg_revenue_by_specialty = completed.groupby("specialty")["revenue_realized"].mean()
no_show_count_by_specialty = no_show_rows.groupby("specialty").size()

modeled_loss_by_specialty = (no_show_count_by_specialty * avg_revenue_by_specialty).dropna().sort_values(ascending=False)
modeled_loss_by_specialty = modeled_loss_by_specialty.round(0)

print(f"\nTotal modeled revenue loss: Rs {modeled_loss_by_specialty.sum():,.0f}")
print(modeled_loss_by_specialty)

plt.figure(figsize=(8, 6))
sns.barplot(x=modeled_loss_by_specialty.values, y=modeled_loss_by_specialty.index,
            hue=modeled_loss_by_specialty.index, palette="Reds_r", legend=False)
plt.title("Estimated Revenue Lost to No-Shows, by Specialty")
plt.xlabel("Estimated Loss (Rs)")
plt.ylabel("")
plt.tight_layout()
plt.show()

# ---- Q15. How does average revenue vary across specialties and appointment types? ----
# Metric: average revenue_realized. Filter: completed.
revenue_by_specialty = completed.groupby("specialty")["revenue_realized"].mean().sort_values(ascending=False).round(0)
print("\n--- Q15: average revenue by specialty ---")
print(revenue_by_specialty)

revenue_by_type = completed.groupby("appointment_type")["revenue_realized"].mean().sort_values(ascending=False).round(0)
print("\n--- Q15: average revenue by appointment type ---")
print(revenue_by_type)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(x=revenue_by_specialty.values, y=revenue_by_specialty.index,
            hue=revenue_by_specialty.index, palette="Greens_r", legend=False, ax=axes[0])
axes[0].set_title("Average Revenue by Specialty")
axes[0].set_xlabel("Avg Revenue (Rs)")
axes[0].set_ylabel("")

sns.barplot(x=revenue_by_type.index, y=revenue_by_type.values,
            hue=revenue_by_type.index, palette="Greens_r", legend=False, ax=axes[1])
axes[1].set_title("Average Revenue by Appointment Type")
axes[1].set_xlabel("")
axes[1].set_ylabel("Avg Revenue (Rs)")

plt.tight_layout()
plt.show()

# ---- Q16. Which cities generate the most revenue, and what is the payment mode mix? ----
# Metric: total revenue_realized per city; % share of payment_mode. Filter: completed.
revenue_by_city = completed.groupby("city")["revenue_realized"].sum().sort_values(ascending=False).round(0)
print("\n--- Q16: total revenue by city ---")
print(revenue_by_city)

plt.figure(figsize=(8, 6))
sns.barplot(x=revenue_by_city.values, y=revenue_by_city.index,
            hue=revenue_by_city.index, palette="Blues_r", legend=False)
plt.title("Total Revenue by City")
plt.xlabel("Total Revenue (Rs)")
plt.ylabel("")
plt.tight_layout()
plt.show()

payment_mix = (completed["payment_mode"].value_counts(normalize=True) * 100).round(1)
print("\n--- Q16: payment mode mix ---")
print(payment_mix)

plt.figure(figsize=(6, 6))
plt.pie(payment_mix, labels=payment_mix.index, autopct="%1.1f%%", startangle=90,
        colors=sns.color_palette("Set2", len(payment_mix)))
plt.title("Payment Mode Mix (Completed Appointments)")
plt.tight_layout()
plt.show()

# ---- Q17. How does insurance coverage affect out-of-pocket payment? ----
# Metric: average patient_out_of_pocket, and OOP as a share of the total fee. Filter: completed.
completed["oop_share_pct"] = (completed["patient_out_of_pocket"] / completed["actual_fee_charged"] * 100).round(1)

insurance_oop = completed.groupby("insurance_coverage").agg(
    avg_out_of_pocket=("patient_out_of_pocket", "mean"),
    avg_oop_share_pct=("oop_share_pct", "mean"),
    appointments=("patient_out_of_pocket", "size")
).round(1)
print("\n--- Q17: out-of-pocket payment, insured vs not ---")
print(insurance_oop)

plt.figure(figsize=(6, 4.5))
sns.barplot(x=insurance_oop.index, y="avg_out_of_pocket", data=insurance_oop,
            hue=insurance_oop.index, palette="coolwarm", legend=False)
plt.title("Average Out-of-Pocket Payment: Insured vs Not")
plt.xlabel("")
plt.ylabel("Avg Out-of-Pocket (Rs)")
plt.tight_layout()
plt.show()

# Data observation: some "insured" Completed rows show zero insurer-paid amount
insured_completed = completed[completed["insurance_coverage"] == "Yes"]
zero_covered = (insured_completed["insurance_covered_amount"] == 0).sum()
print(f"\n--- Q17: zero-covered insured appointments ---")
print(f"Insured Completed appointments where insurer paid Rs 0: {zero_covered} of {len(insured_completed)}"
      f" ({zero_covered/len(insured_completed)*100:.1f}%)")


# ============================================================
# SECTION F -- Doctor Utilisation & Service Quality
# ============================================================

# ---- Q18. Which specialties have the highest / lowest doctor utilisation? ----
# Metric: average doctor_utilization_pct. Filter: completed.
utilisation_by_specialty = completed.groupby("specialty")["doctor_utilization_pct"].mean().sort_values(ascending=False).round(1)
print("\n--- Q18: average doctor utilisation by specialty ---")
print(utilisation_by_specialty)

plt.figure(figsize=(8, 6))
sns.barplot(x=utilisation_by_specialty.values, y=utilisation_by_specialty.index,
            hue=utilisation_by_specialty.index, palette="mako", legend=False)
plt.title("Average Doctor Utilisation (%) by Specialty")
plt.xlabel("Utilisation (%)")
plt.ylabel("")
plt.tight_layout()
plt.show()

# ---- Q19. How long are patients waiting, and does it vary by specialty or time of day? ----
# Metric: mean & median wait_time_minutes. Filter: completed.
wait_by_specialty = completed.groupby("specialty")["wait_time_minutes"].agg(["mean", "median"]).round(1).sort_values("mean", ascending=False)
print("\n--- Q19: wait time by specialty ---")
print(wait_by_specialty)

wait_by_time = completed.groupby("time_of_day")["wait_time_minutes"].agg(["mean", "median"]).round(1)
print("\n--- Q19: wait time by time of day ---")
print(wait_by_time)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(x=wait_by_specialty["mean"].values, y=wait_by_specialty.index,
            hue=wait_by_specialty.index, palette="OrRd", legend=False, ax=axes[0])
axes[0].set_title("Average Wait Time by Specialty")
axes[0].set_xlabel("Avg Wait (min)")
axes[0].set_ylabel("")

sns.barplot(x=wait_by_time.index, y=wait_by_time["mean"].values,
            hue=wait_by_time.index, palette="OrRd", legend=False, ax=axes[1])
axes[1].set_title("Average Wait Time by Time of Day")
axes[1].set_xlabel("")
axes[1].set_ylabel("Avg Wait (min)")

plt.tight_layout()
plt.show()

# ---- Q20. Is there a relationship between consultation duration and patient satisfaction? ----
# Metric: correlation + average satisfaction per duration band. Filter: completed.
corr, p_value = stats.pearsonr(completed["consultation_duration_min"], completed["patient_satisfaction_score"])
print(f"\n--- Q20: correlation, duration vs satisfaction ---")
print(f"Correlation (duration vs satisfaction): r = {corr:.3f}, p-value = {p_value:.4f}")

duration_order = ["<=10 min", "11-20 min", "21-30 min", "31+ min"]
satisfaction_by_duration = completed.groupby("duration_band", observed=True)["patient_satisfaction_score"].mean().reindex(duration_order).round(2)
print("\n--- Q20: average satisfaction by duration band ---")
print(satisfaction_by_duration)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sample = completed.sample(min(3000, len(completed)), random_state=42)
sns.regplot(x="consultation_duration_min", y="patient_satisfaction_score", data=sample,
            scatter_kws={"alpha": 0.15, "s": 12}, line_kws={"color": "red"}, ax=axes[0])
axes[0].set_title("Consultation Duration vs Satisfaction (sampled)")
axes[0].set_xlabel("Duration (min)")
axes[0].set_ylabel("Satisfaction Score")

sns.barplot(x=satisfaction_by_duration.index, y=satisfaction_by_duration.values,
            hue=satisfaction_by_duration.index, palette="crest", legend=False, ax=axes[1])
axes[1].set_title("Avg Satisfaction by Duration Band")
axes[1].set_xlabel("")
axes[1].set_ylabel("Avg Satisfaction")

plt.tight_layout()
plt.show()

# ---- Q21. How does doctor experience relate to the fee charged? ----
# We use the doctor's STANDARD fee (doctor_std_fee from the doctors table), not
# the appointment-level consultation_fee. The appointment-level fee is the
# standard fee multiplied by a fixed factor depending on appointment type
# (Video Consult x 0.85, In-Clinic x 1.0, Home Visit x 1.5) -- so it reflects a
# pricing rule, not the doctor's actual value. The doctor-level view (320
# doctors) is the cleanest way to see this relationship.
corr_fee, p_value_fee = stats.pearsonr(doctors["experience_years"], doctors["consultation_fee"])
print(f"\n--- Q21: correlation, experience vs standard fee ---")
print(f"Correlation (experience vs standard fee): r = {corr_fee:.3f}, p-value = {p_value_fee:.4f}")

exp_bins = [0, 5, 10, 20, 40]
exp_labels = ["2-5 yrs", "6-10 yrs", "11-20 yrs", "21+ yrs"]
doctors["experience_band"] = pd.cut(doctors["experience_years"], bins=exp_bins, labels=exp_labels)

fee_by_experience = doctors.groupby("experience_band", observed=True)["consultation_fee"].mean().round(0)
print("\n--- Q21: average fee by experience band ---")
print(fee_by_experience)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.regplot(x="experience_years", y="consultation_fee", data=doctors,
            scatter_kws={"alpha": 0.5, "s": 25}, line_kws={"color": "red"}, ax=axes[0])
axes[0].set_title("Doctor Experience vs Standard Fee (one point = one doctor)")
axes[0].set_xlabel("Experience (years)")
axes[0].set_ylabel("Standard Fee (Rs)")

sns.barplot(x=fee_by_experience.index, y=fee_by_experience.values,
            hue=fee_by_experience.index, palette="Greens", legend=False, ax=axes[1])
axes[1].set_title("Average Fee by Experience Band")
axes[1].set_xlabel("")
axes[1].set_ylabel("Avg Fee (Rs)")

plt.tight_layout()
plt.show()

print("\nAll 21 questions executed successfully.")