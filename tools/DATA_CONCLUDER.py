#!/usr/bin/env python3
import json
from collections import Counter
import sys

def main():
    results_file = "classification_results.json"
    try:
        with open(results_file, "r") as f:
            classifications = json.load(f)
    except Exception as e:
        print(f"Error loading '{results_file}': {e}")
        sys.exit(1)
    
    # Count the occurrences of each classification
    counts = Counter(classifications.values())
    
    print("Classification Summary:")
    for label in ["1", "2", "3", "4"]:
        count = counts.get(label, 0)
        print(f"Classification {label}: {count}")

if __name__ == "__main__":
    main()
