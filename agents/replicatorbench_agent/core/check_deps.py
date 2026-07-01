#!/usr/bin/env python3
import sys

mods = sys.argv[1:]
missing = []
for name in mods:
    try:
        __import__(name)
    except Exception as e:
        missing.append((name, str(e)))

if missing:
    print("ERROR: Missing Python packages:")
    for n, msg in missing:
        print(f"  {n}  ({msg})")
    print("\nInstall them with:")
    print("  pip install -r requirements-dev.txt")
    print("  OR make install-deps")
    sys.exit(1)

print("All required imports available.")
