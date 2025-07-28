import pandas as pd
import re

# Load new CSV
# fold 1, 300-600
# df = pd.read_csv("/home/as4288/asl_acoustic_data_model/experiments/data/sign_mouth_combos_poi_300_600_th_280ch4_3fold_signs/fold_1/results.csv")

# fold 1, 350-600
# df = pd.read_csv("/home/as4288/asl_acoustic_data_model/experiments/data/sign_mouth_combos_poi_350_600_th_240ch4_3fold_signs_350/fold_1/results.csv")
df = pd.read_csv("/home/as4288/asl_acoustic_data_model/experiments/data/sign_mouth_combos_poi_350_600_th_240ch4_fusion_signs4sesh/all_results.csv")
# experiments/data/sign_mouth_combos_poi_350_600_th_240ch4_fusion_signs4sesh/all_results.csv
# Map base ASL words to their signing area
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
lst = [
    "black", "summer", "dontmind", "dontcare", "dry",
    "mother", "father", "nephew", "niece", "man",
    "woman", "understand", "sick", "disease", "red",
    "cute", "dorm", "home", "girl", "aunt",
    "twin", "restaurant", "see", "watch", "ant",
    "daily", "sunday", "wonderful", "star", "socks",
    "my", "I", "you", "your", "short",
    "child", "family", "class", "electric", "physics",
    "teach", "none", "sit", "chair", "nice",
    "clean", "train", "paper", "school", "read",
    "discuss", "late", "open", "run", "write",
    "carry", "sign", "drive", "bicycle", "study"
]
two_handed = ["sick", "disease", "sunday", "wonderful", "star", "socks", "family","class","electric","physics","teach","none","sit","chair","nice","clean","train","paper","school","read","write","discuss","open","run", "carry", "sign", "drive", "bicycle", "study"]
symmetric = ["sunday", "wonderful", "star", "socks", "family","class","electric","physics","teach","none","open","carry", "sign", "drive", "bicycle"]
asymmetric = ["sick", "disease", "sit","study","chair","nice","clean","train","paper","school","read","write","discuss","run"]

def with_signing_area(word):
    area = asl_sign_locations.get(word, "unknown")
    return f"{word} ({area})"

# Extract base word and morpheme using regex
def extract_base_and_morpheme(sign_str):    
    # match = re.search(r'diff_\d+_(\w+)\(([^)]+)\)\.npy$', fname)
    match = re.match(r'(\w+)\(([^)]+)\)', sign_str)

    if match:
        baseword = match.group(1)
        morpheme = match.group(2)
    else:
        baseword = None
        morpheme = None
    return baseword, morpheme

# Apply extraction
df['BaseWord'], df['Morpheme'] = zip(*df['Sign'].map(extract_base_and_morpheme))
total_predictions_by_morpheme = df['Morpheme'].value_counts()

# Rename for consistency
df.rename(columns={"True Label": "Truth", "Predicted Label": "Predicted"}, inplace=True)

# Compute BaseWord + Morpheme combos columns
df['BaseWord_Morpheme'] = df['BaseWord'] + ' (' + df['Morpheme'] + ')'
df['Truth_Morpheme'] = df['Truth'] + ' (' + df['Morpheme'] + ')'
df['Predicted_Morpheme'] = df['Predicted'] + ' (' + df['Morpheme'] + ')'

# Filter mistakes
mistakes = df[df["Truth"] != df["Predicted"]].copy()
total_mistakes = len(mistakes)
total_predictions = len(df)
print(f"❌ Total number of mistakes: {total_mistakes}/{total_predictions}\n")

# Top 5 truth labels most misclassified
top_truth_mistakes = (
    mistakes.groupby("Truth")
    .size()
    .sort_values(ascending=False)
    .head(5)
)

mistaken_by_truth = (
    mistakes[mistakes["Truth"].isin(top_truth_mistakes.index)]
    .groupby("Truth")["Predicted"]
    .value_counts()
)

# Top 5 incorrect predictions
top_predicted_mistakes = (
    mistakes.groupby("Predicted")
    .size()
    .sort_values(ascending=False)
    .head(5)
)

mistaken_by_predicted = (
    mistakes[mistakes["Predicted"].isin(top_predicted_mistakes.index)]
    .groupby("Predicted")["Truth"]
    .value_counts()
)

# Map signing areas
mistakes.loc[:, "SignArea"] = mistakes["BaseWord"].map(asl_sign_locations)

# Assign SignArea to the full dataset (not just mistakes)
df["SignArea"] = df["BaseWord"].map(asl_sign_locations)

# Total predictions per sign area
total_predictions_by_area = df["SignArea"].value_counts()
mistakes_by_area = mistakes["SignArea"].value_counts()

# Mistakes count per morpheme
mistakes_by_morpheme = mistakes['Morpheme'].value_counts()

# print("Mistakes by Mouth Morpheme and common mistaken predictions:\n")
# for morpheme, group_df in mistakes.groupby('Morpheme'):
#     total_morph_mistakes = len(group_df)
#     print(f"• Morpheme '{morpheme}': {total_morph_mistakes} mistakes")
    
#     baseword_counts = group_df['BaseWord'].value_counts().head(5)
#     print("  Common true basewords with this morpheme mistake:")
#     for baseword, count in baseword_counts.items():
#         print(f"    - {baseword}: {count}")
    
#     predicted_counts = group_df['Predicted'].value_counts().head(5)
#     print("  Mistaken as:")
#     for pred_label, count in predicted_counts.items():
#         # Get the set of morphemes used with this predicted label in these mistakes
#         pred_morphemes = group_df[group_df['Predicted'] == pred_label]['Morpheme'].unique()
#         morpheme_list = ', '.join(sorted(pred_morphemes))
#         print(f"    - {pred_label} ({morpheme_list}): {count}")
#     print()

from tabulate import tabulate  # Ensure it's imported

# Table for morphemes
table_rows = []
for morpheme in sorted(total_predictions_by_morpheme.index):
    total_preds = total_predictions_by_morpheme[morpheme]
    total_mistakes = mistakes_by_morpheme.get(morpheme, 0)
    error_rate = total_mistakes / total_predictions * 100 if total_preds > 0 else 0.0
    error_rate_morpheme = total_mistakes / total_preds * 100 if total_preds > 0 else 0.0


    # Get top 3 mistaken basewords
    morph_mistakes = mistakes[mistakes['Morpheme'] == morpheme]
    top_basewords = morph_mistakes['BaseWord'].value_counts().head(3)
    baseword_str = ', '.join([f"{bw} ({ct})" for bw, ct in top_basewords.items()])

    table_rows.append([
        morpheme,
        f"{total_mistakes} / {total_preds} ({error_rate_morpheme:.1f}%)",
        f"{total_mistakes} / {total_predictions} ({error_rate:.1f}%)",
        baseword_str
    ])

print("\n📊 Mistake Summary by Mouth Morpheme\n")
print(tabulate(
    table_rows,
    headers=["Morpheme", "Predictions", "Mistakes (Count/Total %)", "Top Mistaken BaseWords"],
    tablefmt="github"
))


print("Mistakes by Mouth Morpheme and common mistaken predictions:\n")

for morpheme, group_df in mistakes.groupby('Morpheme'):
    total_morph_mistakes = len(group_df)
    total_morph_predictions = total_predictions_by_morpheme.get(morpheme, 0)
    percent = (total_morph_mistakes / total_predictions * 100) if total_predictions > 0 else 0.0
    print(f"• Morpheme '{morpheme}': {total_morph_mistakes} / {total_predictions} predictions ({percent:.1f}%)")
    
    baseword_counts = group_df['BaseWord'].value_counts().head(5)
    print("  Common true basewords with this morpheme mistake:")
    for baseword, count in baseword_counts.items():
        print(f"    - {baseword}: {count} mistakes")
    
    predicted_counts = group_df['Predicted'].value_counts().head(5)
    print("  Mistaken as:")
    for pred_label, count in predicted_counts.items():
        pred_morphemes = group_df[group_df['Predicted'] == pred_label]['Morpheme'].unique()
        morpheme_list = ', '.join(sorted(pred_morphemes))
        print(f"    - {pred_label} ({morpheme_list}): {count}")
    
    print()


    

print("📊 Mistakes by Signing Area:")
for area, count in mistakes_by_area.items():
    total_area_predictions = total_predictions_by_area.get(area, 0)
    print(f"  • {area}: {count} mistakes out of {total_area_predictions} predictions ({count/total_area_predictions*100:.2f}%)")

print()

print("\n📊 Mistake Summary by BaseWord\n")

# Total predictions and mistakes per BaseWord
total_preds_by_baseword = df['BaseWord'].value_counts()
mistakes_by_baseword = mistakes['BaseWord'].value_counts()

baseword_rows = []
for baseword in sorted(total_preds_by_baseword.index):
    total = total_preds_by_baseword[baseword]
    mistakes_count = mistakes_by_baseword.get(baseword, 0)
    percent = mistakes_count / total_predictions * 100 if total_predictions > 0 else 0.0

    baseword_rows.append([
        baseword,
        total,
        f"{mistakes_count} / {total_predictions} ({percent:.1f}%)"
    ])

print(tabulate(
    baseword_rows,
    headers=["BaseWord", "Predictions", "Mistakes (Count/Total %)"],
    tablefmt="github"
))


print("🔢 Top 5 Truth signs most often misclassified:\n")
for truth_label, count in top_truth_mistakes.items():
    print(f"  • {with_signing_area(truth_label)}: {count}")

print("🔍 What those truth expressions were usually mistaken as:\n")
last_truth = None
for (truth_label, pred_label), count in mistaken_by_truth.items():
    if truth_label != last_truth:
        print(f"  • {with_signing_area(truth_label)}:")
        last_truth = truth_label
    print(f"    - {with_signing_area(pred_label)}: {count}")
print()

print("🔢 Top 5 Predicted signs that were most often wrong:\n")
for pred_label, count in top_predicted_mistakes.items():
    print(f"  • {with_signing_area(pred_label)}: {count}")
print()

print("🔍 What those predicted expressions actually should have been:\n")
last_pred = None
for (pred_label, truth_label), count in mistaken_by_predicted.items():
    if pred_label != last_pred:
        print(f"  • {with_signing_area(pred_label)}:")
        last_pred = pred_label
    print(f"    - {with_signing_area(truth_label)}: {count}")
print()

print("Top mistaken BaseWord + Morpheme combos:")
top_mistakes_bwm = mistakes['BaseWord_Morpheme'].value_counts().head(10)
print(top_mistakes_bwm)


# Analyze specific morphemes like 'shake' and 'th'
target_morphemes = ['shake', 'th']

for morph in target_morphemes:
    print(f"\n🔍 Morpheme: '{morph}' — Detailed Error Breakdown per Sign")

    morph_df = df[df['Morpheme'] == morph]
    morph_mistakes = mistakes[mistakes['Morpheme'] == morph]

    grouped_total = morph_df['BaseWord'].value_counts()
    grouped_errors = morph_mistakes['BaseWord'].value_counts()

    for baseword in grouped_total.index:
        total = grouped_total[baseword]
        mistakes_count = grouped_errors.get(baseword, 0)
        error_rate = mistakes_count / total * 100 if total > 0 else 0

        status = "✅ partial errors" if 0 < mistakes_count < total else (
            "❌ all misclassified" if mistakes_count == total else "✔️ all correct"
        )

        print(f"  • {baseword}: {mistakes_count} / {total} errors ({error_rate:.1f}%) → {status}")
target_morphemes = ['shake', 'th']
sign_areas = ['upper face', 'lower face', 'body']

print("\n📊 Error Breakdown for Specific Morphemes by Sign Region\n")

for morph in target_morphemes:
    print(f"\n🔍 Morpheme: '{morph}'")
    morph_df = df[df['Morpheme'] == morph]
    morph_mistakes = mistakes[mistakes['Morpheme'] == morph]

    for area in sign_areas:
        total_preds = len(morph_df[morph_df['SignArea'] == area])
        total_errors = len(morph_mistakes[morph_mistakes['SignArea'] == area])
        error_rate = (total_errors / total_preds * 100) if total_preds > 0 else 0.0

        print(f"  • {area:<11}: {total_errors} / {total_preds} errors ({error_rate:.1f}%)")


# Print lengths of lists to verify
print(f"Length of two_handed list: {len(two_handed)}")
print(f"Length of symmetric list: {len(symmetric)}")
print(f"Length of asymmetric list: {len(asymmetric)}")

# Add a column to df and mistakes indicating handedness
df['Handedness'] = df['BaseWord'].apply(lambda w: 'Two-handed' if w in two_handed else 'One-handed')
mistakes['Handedness'] = mistakes['BaseWord'].apply(lambda w: 'Two-handed' if w in two_handed else 'One-handed')

# Add a column to df and mistakes indicating symmetry for two-handed only
def get_symmetry(baseword):
    if baseword in two_handed:
        if baseword in symmetric:
            return 'symmetric'
        elif baseword in asymmetric:
            return 'asymmetric'
        else:
            return 'unknown'
    else:
        return 'one-handed'

mistakes['Symmetry'] = mistakes['BaseWord'].apply(get_symmetry)
df['Symmetry'] = df['BaseWord'].apply(get_symmetry)

# Recompute symmetry for all rows
# df['Symmetry'] = df.apply(get_symmetry, axis=1)

# # df['Symmetry'] = df['BaseWord'].apply(get_symmetry)
# # mistakes['Symmetry'] = mistakes['BaseWord'].apply(get_symmetry)
# mistakes['Symmetry'] = mistakes.apply(get_symmetry, axis=1)
# Filter subsets# Accuracy Breakdown by Symmetry Category (corrected to avoid reindexing warning)
one_handed_mask = df['Symmetry'] == 'one-handed'
sym_mask = df['Symmetry'] == 'symmetric'
asym_mask = df['Symmetry'] == 'asymmetric'

one_handed_mistakes = mistakes[mistakes['Symmetry'] == 'one-handed']
sym_mistakes = mistakes[mistakes['Symmetry'] == 'symmetric']
asym_mistakes = mistakes[mistakes['Symmetry'] == 'asymmetric']

print("\n📊 Overall Accuracy Breakdown\n")
print(f"  • One-handed            : {len(one_handed_mistakes)} mistakes / {one_handed_mask.sum()} samples → {(1 - len(one_handed_mistakes) / one_handed_mask.sum()) * 100:.2f}% accuracy")
print(f"  • Two-handed (Symmetric): {len(sym_mistakes)} mistakes / {sym_mask.sum()} samples → {(1 - len(sym_mistakes) / sym_mask.sum()) * 100:.2f}% accuracy")
print(f"  • Two-handed (Asymmetric): {len(asym_mistakes)} mistakes / {asym_mask.sum()} samples → {(1 - len(asym_mistakes) / asym_mask.sum()) * 100:.2f}% accuracy")

# total_predictions = len(df)
# total_mistakes = len(mistakes)

# print("\n📊 Accuracy by Handedness:")
# for handedness in ['One-handed', 'Two-handed']:
#     total_handedness = len(df[df['Handedness'] == handedness])
#     mistakes_handedness = len(mistakes[mistakes['Handedness'] == handedness])
#     accuracy = 100 * (1 - mistakes_handedness / total_handedness) if total_handedness > 0 else 0
#     print(f"  • {handedness}: {accuracy:.2f}% accuracy ({mistakes_handedness} mistakes / {total_handedness} samples)")

# print("\n📊 Accuracy by Symmetry (Two-handed signs only):")
# for symm in ['Symmetric', 'Asymmetric']:
#     total_symm = len(df[df['Symmetry'] == symm])
#     mistakes_symm = len(mistakes[mistakes['Symmetry'] == symm])
#     accuracy = 100 * (1 - mistakes_symm / total_symm) if total_symm > 0 else 0
#     print(f"  • {symm}: {accuracy:.2f}% accuracy ({mistakes_symm} mistakes / {total_symm} samples)")

# # ✅ Detailed summary by hand/symmetry breakdown
# print("\n📊 Overall Accuracy Breakdown\n")

# # Helper to format output
# def print_accuracy(label, correct, total):
#     accuracy = 100 * (correct / total) if total > 0 else 0
#     print(f"  • {label:<22}: {total - correct} mistakes / {total} samples → {accuracy:.2f}% accuracy")

# # One-handed
# df_one = df[df["Handedness"] == "One-handed"]
# mistakes_one = mistakes[mistakes["Handedness"] == "One-handed"]
# print_accuracy("One-handed", len(df_one) - len(mistakes_one), len(df_one))

# # Two-handed symmetric
# df_sym = df[df["Symmetry"] == "Symmetric"]
# mistakes_sym = mistakes[mistakes["Symmetry"] == "Symmetric"]
# print_accuracy("Two-handed (Symmetric)", len(df_sym) - len(mistakes_sym), len(df_sym))

# # Two-handed asymmetric
# df_asym = df[df["Symmetry"] == "Asymmetric"]
# mistakes_asym = mistakes[mistakes["Symmetry"] == "Asymmetric"]
# print_accuracy("Two-handed (Asymmetric)", len(df_asym) - len(mistakes_asym), len(df_asym))

# # Sanity check: unknown/NA category
# df_other = df[df["Symmetry"] == "N/A"]
# if not df_other.empty:
#     print_accuracy("Two-handed (Unknown)", len(df_other) - len(mistakes[mistakes["Symmetry"] == "N/A"]), len(df_other))
# # Sanity check for missing symmetry classification
# missing_symmetry = [
#     sign for sign in two_handed
#     if sign not in symmetric and sign not in asymmetric
# ]

# print(f"🚨 Two-handed signs missing symmetry classification: {missing_symmetry}")
# # Identify which two-handed signs are being labeled as "unknown"
# unknown_symmetry_basewords = set(
#     df[df['Symmetry'] == 'unknown']['BaseWord']
# ).intersection(two_handed)

# print("⚠️ Two-handed basewords with unknown symmetry:")
# print(sorted(unknown_symmetry_basewords))
