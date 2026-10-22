def assess_severity(voltage_v, current_a, power_w,
                    demo_anomaly=False, ai_anomaly=False):
    """Prototype demonstration rules, not safety protection settings."""

    if (voltage_v < 190 or voltage_v > 260
            or current_a > 20 or power_w > 5000):
        return "Critical"

    if (voltage_v < 207 or voltage_v > 253
            or current_a > 10 or power_w > 3000):
        return "High"

    if demo_anomaly or ai_anomaly:
        return "Medium"

    return "Normal"


tests = [
    ("Normal reading", 230, 2, 460, False, False, "Normal"),
    ("Simulated event", 230, 1.4, 320, True, False, "Medium"),
    ("High current", 230, 14, 3200, True, False, "High"),
    ("Very low voltage", 185, 8, 1450, True, False, "Critical"),
    ("AI anomaly", 230, 2, 460, False, True, "Medium"),
]

failures = 0

for name, voltage, current, power, demo, ai, expected in tests:
    actual = assess_severity(
        voltage, current, power,
        demo_anomaly=demo, ai_anomaly=ai
    )
    passed = actual == expected
    print(f'{"PASS" if passed else "FAIL"}: {name} '
          f'| Expected={expected} | Actual={actual}')
    if not passed:
        failures += 1

print(f"\nTests passed: {len(tests) - failures}/{len(tests)}")
if failures:
    raise SystemExit(1)

print("All severity tests passed. Database was not accessed.")
