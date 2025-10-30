import os
import json
import fire
import re

def extract_category_number(filename):
    match = re.search(r'category_(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

def summarize_duo_scores(input_dir):
    output_file = os.path.join(input_dir, "summary.md")

    all_scores = []
    all_total = 0
    all_fives = 0

    rows = []
    file_list = sorted(
        [f for f in os.listdir(input_dir) if f.endswith(".jsonl")],
        key=extract_category_number
    )

    for filename in file_list:
        file_path = os.path.join(input_dir, filename)
        scores = []
        fives = 0
        total = 0

        with open(file_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    score = data.get("duo_score", None)
                    if isinstance(score, (int, float)):
                        scores.append(score)
                        total += 1
                        if score == 5:
                            fives += 1
                except Exception as e:
                    print(f"[⚠️ Error] Skipped line in {filename}: {e}")
                    continue

        if total > 0:
            avg = sum(scores) / total
            ratio = fives / total
            all_scores.extend(scores)
            all_total += total
            all_fives += fives
            category_num = extract_category_number(filename)
            rows.append((category_num, filename, avg, ratio))
        else:
            print(f"[{filename}] ⚠️ No valid entries.")

    # Sort by category number again to be sure
    rows.sort(key=lambda x: x[0])

    # Write to markdown file
    with open(output_file, "w") as out:
        out.write("| Category | Avg Score | Score=5 Ratio |\n")
        out.write("|----------|-----------|----------------|\n")
        for cat, fname, avg, ratio in rows:
            out.write(f"| {cat} | {avg:.3f} | {ratio:.3f} |\n")
        
        if all_total > 0:
            overall_avg = sum(all_scores) / all_total
            overall_ratio = all_fives / all_total
        else:
            overall_avg = 0
            overall_ratio = 0
        out.write(f"| **Overall** | **{overall_avg:.3f}** | **{overall_ratio:.3f}** |\n")

    print(f"\n✅ Markdown summary saved to: {output_file}")

if __name__ == "__main__":
    fire.Fire(summarize_duo_scores)
