import os

TARGET = os.path.join(
    os.path.dirname(__file__),
    "src",
    "classification",
    "denoising_classification_experiment.py"
)

if not os.path.exists(TARGET):
    raise FileNotFoundError(f"Could not find:\n{TARGET}")

with open(TARGET, "r", encoding="utf-8") as f:
    original = f.read()

text = original

old = '''    if condition == "gaussian_noisy":

        return os.path.join(

            DENOISED_ROOT,

            split,

            class_name,

            "gaussian",

            "gaussian_noisy",

            filename

        )'''

new = '''    if condition == "gaussian_noisy":

        return os.path.join(

            NOISY_ROOT,

            split,

            class_name,

            "gaussian",

            filename

        )'''

if old in text:
    text = text.replace(old, new, 1)
elif 'NOISY_ROOT = os.path.join(' in text:
    print("The path section already appears corrected.")
else:
    raise RuntimeError("Could not locate the old gaussian_noisy path.")

if 'NOISY_ROOT = os.path.join(' not in text:
    marker = '''DENOISED_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "denoised_fast"
)'''
    addition = marker + '''

NOISY_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "noisy"
)'''
    if marker not in text:
        raise RuntimeError("Could not locate DENOISED_ROOT section.")
    text = text.replace(marker, addition, 1)

backup = TARGET + ".backup_before_step11_fix"
if not os.path.exists(backup):
    with open(backup, "w", encoding="utf-8") as f:
        f.write(original)

with open(TARGET, "w", encoding="utf-8") as f:
    f.write(text)

print("=" * 70)
print("STEP 11 PATH FIX APPLIED")
print("=" * 70)
print("Gaussian noisy path is now:")
print("data/processed/noisy/<train_or_val>/<Control_or_DM>/gaussian/<filename>")
print("Backup:", backup)
print("=" * 70)
