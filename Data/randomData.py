import csv
import random

def label_row_noisy(study_hours, screen_time, stress_level, physical_activity):
    w_s = random.normalvariate(0.3, 0.03)
    w_t = random.normalvariate(0.25, 0.025)
    w_str = random.normalvariate(0.5, 0.05)
    w_p = random.normalvariate(0.25, 0.025)
    base_courses = 2.4

    risk_score = (
        w_s * study_hours +
        w_t * screen_time +
        w_str * stress_level -
        w_p * physical_activity +
        base_courses
    )
    if risk_score >= 13:
        label = "high"
    elif risk_score >= 6:
        label = "medium"
    else:
        label = "low"

    if 8 <= risk_score < 10 or 14 <= risk_score < 16:
        if random.random() < 0.15:
            if label == "medium":
                label = random.choice(["low", "high"])
            elif label == "low":
                label = "medium"
            elif label == "high":
                label = "medium"

    return label

rows = []
for _ in range(150):
    study_hours = round(random.uniform(0, 10), 1)
    screen_time = round(random.uniform(0, 10), 1)
    stress_level = random.randint(1, 10)
    physical_activity = round(random.uniform(0, 14), 1)
    number_of_courses = 6

    burnout_risk = label_row_noisy(
        study_hours,
        screen_time,
        stress_level,
        physical_activity
    )

    rows.append([
        study_hours,
        screen_time,
        stress_level,
        physical_activity,
        number_of_courses,
        burnout_risk
    ])

with open("burnout_150rows_noisy.csv", "w", newline="") as f:
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
