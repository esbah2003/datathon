import csv
import random

def label_row(study_hours, screen_time, stress_level, physical_activity, number_of_courses):
    risk_score = (
        0.3 * study_hours +
        0.25 * screen_time +
        0.5 * stress_level +
        0.4 * number_of_courses -
        0.25 * physical_activity
    )
    if risk_score >= 12:
        return "high"
    elif risk_score >= 7:
        return "medium"
    else:
        return "low"

rows = []
for _ in range(500):
    # Sample with reasonable student-like distributions
    study_hours = round(random.uniform(0, 10), 1)
    screen_time = round(random.uniform(0, 10), 1)
    stress_level = random.randint(1, 10)
    physical_activity = round(random.uniform(0, 14), 1)
    number_of_courses = random.randint(1, 8)

    burnout_risk = label_row(
        study_hours,
        screen_time,
        stress_level,
        physical_activity,
        number_of_courses
    )

    rows.append([
        study_hours,
        screen_time,
        stress_level,
        physical_activity,
        number_of_courses,
        burnout_risk
    ])

with open("burnout_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "study_hours",
        "screen_time",
        "stress_level",
        "physical_activity",
        "number_of_courses",
        "burnout_risk"
    ])
    writer.writerows(rows)
