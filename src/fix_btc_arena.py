from pathlib import Path

file_path = Path("src/btc_predictor_all_in_one.py")

print("Looking for:", file_path)

if not file_path.exists():
    print("File not found!")
    exit()

code = file_path.read_text(encoding="utf-8", errors="ignore")

replacements = {
    "ΓÇö": "-",
    "ΓöÇ": "",
    "Γ£à": "✔",
    "Γ¥î": "✖",
    "≡ƒôê": "🟢",
    "≡ƒôë": "🔴",
    "≡ƒÜ¿": "🚨",
}

for bad, good in replacements.items():
    code = code.replace(bad, good)

file_path.write_text(code, encoding="utf-8")

print("✔ btc_predictor_all_in_one.py repaired successfully")
