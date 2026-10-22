from pathlib import Path

app_path = Path("app.py")
backup_path = Path("app.py.backup_before_severity_fix")

if not app_path.exists():
    raise SystemExit("ERROR: app.py was not found.")

if not backup_path.exists():
    raise SystemExit("ERROR: Original backup was not found.")

text = app_path.read_text(encoding="utf-8")

old = '''        if (
            row["Voltage_V"] < 190 or row["Voltage_V"] > 260
            or row["Current_A"] > 20 or row["Power_W"] > 5000
        ):
            return "Critical"
        if (
            row["Voltage_V"] < 207 or row["Voltage_V"] > 253
            or row["Current_A"] > 10 or row["Power_W"] > 3000
            or row["Demo_Anomaly"]
        ):
            return "High"
        if row["AI_Anomaly"]:
            return "Medium"
        return "Low"
'''

new = '''        if (
            row["Voltage_V"] < 190 or row["Voltage_V"] > 260
            or row["Current_A"] > 20 or row["Power_W"] > 5000
        ):
            return "Critical"
        if (
            row["Voltage_V"] < 207 or row["Voltage_V"] > 253
            or row["Current_A"] > 10 or row["Power_W"] > 3000
        ):
            return "High"
        if row["AI_Anomaly"] or row["Demo_Anomaly"]:
            return "Medium"
        return "Low"
'''

if text.count(old) != 1:
    raise SystemExit(
        "ERROR: Expected severity code was not found exactly once. "
        "app.py was not changed."
    )

print("Patch prepared successfully.")
print("Original backup exists.")
print("No files have been modified yet.")
