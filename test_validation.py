# Quick test of validation
from scoring import ScoreParser

parser = ScoreParser()
formula = "(6 Month Performance * 6 Month Sharpe)"

print(f"Testing formula: {formula}")
print(f"Metrics list has {len(parser.allowed_metrics)} metrics")
print(f"'6 Month Performance' in metrics: {'6 Month Performance' in parser.allowed_metrics}")
print(f"'6 Month Sharpe' in metrics: {'6 Month Sharpe' in parser.allowed_metrics}")

valid, msg = parser.validate_formula(formula)
print(f"Valid: {valid}, Message: {msg}")

# Test preprocessing
processed = parser._preprocess_formula(formula)
print(f"Preprocessed: {processed}")

# Test replacement
test_formula = processed
for metric in sorted(parser.allowed_metrics, key=len, reverse=True):
    if metric in test_formula:
        old_formula = test_formula
        test_formula = test_formula.replace(metric, "(1)")
        if old_formula != test_formula:
            print(f"Replaced '{metric}' -> result: {test_formula}")

print(f"Final test formula: {test_formula}")

# Try to eval
try:
    result = eval(test_formula)
    print(f"Eval result: {result}")
except Exception as e:
    print(f"Eval error: {e}")
