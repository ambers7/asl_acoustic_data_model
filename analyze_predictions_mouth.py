import pandas as pd

# Load CSV
# df = pd.read_csv("/home/as4288/asl_acoustic_data_model/experiments/data/6foldsign_mouth_combos_poi_300_360_th_50ch4_fusion_withcsvs/reloading/test_results_combined_full.csv")

df = pd.read_csv("/home/as4288/asl_acoustic_data_model/experiments/data/sign_mouth_combos_poi_300_360_th_50ch4_3fold_mouth/fold_1/results.csv")

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
print("🔢 Top 5 BaseWords with most mistakes:\n", top_baseword_mistakes, '\n')
print("🔍 What those base words were usually mistaken as:\n", mistaken_by_baseword, '\n')

print("🔢 Top 5 Truth facial expressions most often misclassified:\n", top_truth_mistakes, '\n')
print("🔍 What those truth expressions were usually mistaken as:\n", mistaken_by_truth, '\n')

print("🔢 Top 5 Predicted expressions that were most often wrong:\n", top_predicted_mistakes, '\n')
print("🔍 What those predicted expressions actually should have been:\n", mistaken_by_predicted, '\n')

# Assign SignArea using .loc to avoid SettingWithCopyWarning
mistakes.loc[:, "SignArea"] = mistakes["BaseWord"].map(asl_sign_locations)

# Mistakes by sign area
mistakes_by_area = mistakes["SignArea"].value_counts()
print("📊 Mistakes by Signing Area:")
for area, count in mistakes_by_area.items():
    print(f"  • {area}: {count} / {total_mistakes} mistakes")
print()

# Top 3 mistaken signs per signing area (with full labels like dorm(mm))
print("🔍 Top 10 Mistaken Signs per Signing Area:\n")
for area in mistakes_by_area.index:
    area_mistakes = mistakes[mistakes["SignArea"] == area]
    top_sign_labels = area_mistakes["Sign"].value_counts().head(10)
    
    print(f"📍 {area.upper()} — Top 10 mistaken signs:")
    for sign_label, count in top_sign_labels.items():
        subdf = area_mistakes[area_mistakes["Sign"] == sign_label]
        mistaken_as = (
            subdf.groupby("Truth")["Predicted"]
            .value_counts()
            .head(10)
        )
        print(f"  • {sign_label} — {count} mistakes")
        print(f"    ↳ Most often mistaken as:")
        for (_, pred_label), pred_count in mistaken_as.items():
            print(f"      - {pred_label}")
        print()
