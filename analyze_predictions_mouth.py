import pandas as pd
from collections import defaultdict

# Load CSV
# df = pd.read_csv("/home/as4288/asl_acoustic_data_model/experiments/data/6foldsign_mouth_combos_poi_300_360_th_50ch4_fusion_withcsvs/reloading/test_results_combined_full.csv")

df = pd.read_csv("/home/as4288/asl_acoustic_data_model/experiments/data/sign_mouth_combos_poi_300_360_th_50ch4_3fold_mouth/fold_2/results.csv")

# ASL sign locations
asl_sign_locations = {
    "black": "upper face", "summer": "upper face", "dontmind": "lower face", "dontcare": "lower face", "dry": "lower face",
    "mother": "lower face", "father": "upper face", "nephew": "upper face", "niece": "lower face", "man": "upper face",
    "woman": "lower face", "understand": "upper face", "sick": "body", "disease": "body", "red": "lower face",
    "cute": "lower face", "dorm": "lower face", "home": "lower face", "girl": "lower face", "aunt": "lower face",
    "twin": "lower face", "restaurant": "lower face", "see": "upper face", "watch": "upper face", "ant": "body",
    "daily": "lower face", "sunday": "upper face", "wonderful": "upper face", "star": "upper face", "socks": "body",
    "my": "body", "I": "body", "you": "body", "your": "body", "short": "body", "child": "body", "family": "body",
    "class": "body", "electric": "body", "physics": "body", "teach": "upper face", "none": "body", "sit": "body",
    "chair": "body", "nice": "body", "clean": "body", "train": "body", "paper": "body", "school": "body",
    "read": "body", "discuss": "body", "late": "body", "open": "body", "run": "body", "write": "body",
    "carry": "body", "sign": "body", "drive": "body", "bicycle": "body", "study": "body"
}
# Extract full sign (e.g., "girl(mm)") from file name
df["Sign"] = df["File"].str.extract(r'acoustic_diff_\d+_(.+)\.npy')

# Extract base ASL word (e.g., "girl")
df["BaseWord"] = df["Sign"].str.extract(r'^([^\(]+)')

# Extract mouth morpheme (e.g., "mm")
df["MouthMorph"] = df["Sign"].str.extract(r'\((.*?)\)')

# Rename for consistency
df.rename(columns={"True Label": "Truth", "Predicted Label": "Predicted"}, inplace=True)

# Compute BaseWord + Morpheme combos columns
df['BaseWord_MouthMorph'] = df['BaseWord'] + ' (' + df['MouthMorph'] + ')'
df['Truth_MouthMorph'] = df['Truth'] + ' (' + df['MouthMorph'] + ')'
df['Predicted_MouthMorph'] = df['Predicted'] + ' (' + df['MouthMorph'] + ')'

# Filter to only mistaken predictions
mistakes = df[df["Truth"] != df["Predicted"]].copy()

# Total mistakes
total_mistakes = len(mistakes)
print(f"❌ Total number of mistakes: {total_mistakes}\n")

# Top 5 most mistaken basewords
top_baseword_mistakes = (
    mistakes.groupby("BaseWord")
    .size()
    .sort_values(ascending=False)
    .head(5)
)

# What each baseword was mistaken as
mistaken_by_baseword = (
    mistakes[mistakes["BaseWord"].isin(top_baseword_mistakes.index)]
    .groupby(["BaseWord", "Truth"])["Predicted"]
    .value_counts()
)

# Top 5 mistaken truth expressions
top_truth_mistakes = (
    mistakes.groupby("Truth")
    .size()
    .sort_values(ascending=False)
    .head(5)
)

# What each truth expression was mistaken as
mistaken_by_truth = (
    mistakes[mistakes["Truth"].isin(top_truth_mistakes.index)]
    .groupby("Truth")["Predicted"]
    .value_counts()
)

# Top 5 predicted expressions that were wrong
top_predicted_mistakes = (
    mistakes.groupby("Predicted")
    .size()
    .sort_values(ascending=False)
    .head(5)
)

# What each wrong predicted label should've been
mistaken_by_predicted = (
    mistakes[mistakes["Predicted"].isin(top_predicted_mistakes.index)]
    .groupby("Predicted")["Truth"]
    .value_counts()
)

# Print results
# print("🔢 Top 5 BaseWords with most mistakes:\n", top_baseword_mistakes, '\n')
# print("🔍 What those base words were usually mistaken as:\n", mistaken_by_baseword, '\n')

# print("🔢 Top 5 Truth facial expressions most often misclassified:\n", top_truth_mistakes, '\n')
# print("🔍 What those truth expressions were usually mistaken as:\n", mistaken_by_truth, '\n')

# print("🔢 Top 5 Predicted expressions that were most often wrong:\n", top_predicted_mistakes, '\n')
# print("🔍 What those predicted expressions actually should have been:\n", mistaken_by_predicted, '\n')

# Assign SignArea using .loc to avoid SettingWithCopyWarning
mistakes.loc[:, "SignArea"] = mistakes["BaseWord"].map(asl_sign_locations)

# Mistakes by sign area
mistakes_by_area = mistakes["SignArea"].value_counts()
print("📊 Mistakes by Signing Area:")
for area, count in mistakes_by_area.items():
    print(f"  • {area}: {count} / {total_mistakes} mistakes")
print()


# OVERVIEW: facial expression × signing area
print("📊 OVERVIEW: Total Mistakes by Facial Expression and Signing Area\n")

# Collect data in a nested dict: morph → area → count
overview = defaultdict(lambda: defaultdict(int))

# Count total mistakes per morph
morph_totals = mistakes["MouthMorph"].value_counts().to_dict()

# Populate nested dict with counts
for _, row in mistakes.iterrows():
    morph = row["MouthMorph"]
    area = row["SignArea"]
    overview[morph][area] += 1

# Print a nice summary
areas = ["upper face", "lower face", "body"]
header = f"{'Morpheme':<10} | {'Total':<5} | " + " | ".join([f"{a:<11}" for a in areas])
print(header)
print("-" * len(header))
for morph in sorted(overview.keys()):
    total = morph_totals.get(morph, 0)
    counts = [overview[morph].get(area, 0) for area in areas]
    line = f"{morph:<10} | {total:<5} | " + " | ".join([f"{c:<11}" for c in counts])
    print(line)


# Compute and print aggregate percentages for all facial expressions
total_upper = sum(overview[m].get("upper face", 0) for m in overview)
total_lower = sum(overview[m].get("lower face", 0) for m in overview)
total_body  = sum(overview[m].get("body", 0) for m in overview)
total_all   = total_upper + total_lower + total_body

if total_all > 0:
    percent_upper = total_upper / total_all * 100
    percent_lower = total_lower / total_all * 100
    percent_upper_lower = (total_upper + total_lower) / total_all * 100

    print("🧠 Overall Mistake Distribution by Signing Area (all morphemes):")
    print(f"  • Upper Face: {total_upper} mistakes ({percent_upper:.1f}%)")
    print(f"  • Lower Face: {total_lower} mistakes ({percent_lower:.1f}%)")
    print(f"  • Combined Upper + Lower: {total_upper + total_lower} mistakes ({percent_upper_lower:.1f}%)")
    print(f"  • Body: {total_body} mistakes ({total_body / total_all * 100:.1f}%)")
else:
    print("⚠️ No mistakes to compute area percentages.\n")

print("\n🧠 Mistake Distribution by Signing Area for Each Facial Expression:")
for morph, areas in overview.items():
    total = sum(areas.values())
    upper = areas.get("upper face", 0)
    lower = areas.get("lower face", 0)
    body  = areas.get("body", 0)
    combined = upper + lower

    if total > 0:
        percent_upper = upper / total * 100
        percent_lower = lower / total * 100
        percent_combined = combined / total * 100
        percent_body = body / total * 100

        print(f"\n😶 '{morph}': {total} total mistakes")
        print(f"  • Upper Face: {upper} ({percent_upper:.1f}%)")
        print(f"  • Lower Face: {lower} ({percent_lower:.1f}%)")
        print(f"  • Combined Upper + Lower: {combined} ({percent_combined:.1f}%)")
        print(f"  • Body: {body} ({percent_body:.1f}%)")
    else:
        print(f"\n😶 '{morph}': 0 total mistakes")


print("\n" + "=" * 60 + "\n")

# PART 2 — DETAILED BREAKDOWN for each morph
print("🧠 DETAILED MISTAKE BREAKDOWN by Facial Expression and Signing Area\n")

for morph in sorted(mistakes["MouthMorph"].dropna().unique()):
    morph_mistakes = mistakes[mistakes["MouthMorph"] == morph]
    total_morph_mistakes = len(morph_mistakes)

    print(f"😶 Facial Expression: '{morph}' — {total_morph_mistakes} total mistakes")

    area_counts = morph_mistakes["SignArea"].value_counts()
    for area, count in area_counts.items():
        print(f"  • {area}: {count} / {total_morph_mistakes} mistakes")

    print("\n🔍 Detailed mistakes grouped by signing area:")
    for area in area_counts.index:
        area_df = morph_mistakes[morph_mistakes["SignArea"] == area]
        sign_counts = area_df["Sign"].value_counts().head(10)

        print(f"\n📍 {area.upper()} — Top mistaken signs:")
        for sign, count in sign_counts.items():
            print(f"  • {sign}: {count} mistakes")

            subdf = area_df[area_df["Sign"] == sign]
            mistaken_as = subdf["Predicted_MouthMorph"].value_counts().head(5)

            for pred_label, pred_count in mistaken_as.items():
                print(f"    ↳ Mistaken as: {pred_label} ({pred_count} times)")

    print("-" * 60)
