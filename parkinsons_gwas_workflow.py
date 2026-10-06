# ============================================================
# End-to-end PD GWAS + CI analysis with HTML summaries + validation
# ============================================================
# Scientific goals:
#   1) Identify variants associated with tremor severity and cognitive impairment (CI) in PD.
#   2) Distinguish motor-specific, CI-specific, and shared genetic effects.
#   3) Summarize model behavior (inflation, hit counts, overlaps).
#   4) Validate PD-severity signals against a large PD case–control GWAS.
#   5) Double-validate CI signals using both PD risk and general cognition in the case–control dataset.
#   6) Prepare outputs for PRS and LD-score regression (LDSC) analyses.

import os
import glob
import hail as hl
import pandas as pd
import numpy as np
import gcsfs
import sys

# ============================================================
# 0. SETUP
# ============================================================
# Scientific purpose:
#   Define data locations and initialize Hail for GWAS using the updated
#   PD and case–control MatrixTables that include PCA and CI fields.

bucket_base = os.environ["YOUR_BUCKET"]

# Updated MT paths (new versions with PCA + CI)
pd_mt_path = f"{bucket_base}/PD_with_PCA_and_CI.mt"
pd_out_dir = f"{bucket_base}/pd_full_gwas_models_with_CI"
os.makedirs(pd_out_dir, exist_ok=True)

cc_mt_path = f"{bucket_base}/CC_with_PCA_and_CI.mt"
cc_out_dir = os.path.join(pd_out_dir, "case_control_gwas")
os.makedirs(cc_out_dir, exist_ok=True)


fs = gcsfs.GCSFileSystem()

# ============================================================
# 1. LOAD PD MATRIX TABLE (SEVERITY COHORT)
# ============================================================

mt = hl.read_matrix_table(pd_mt_path)

# Add PCs if needed
if "pc0" not in mt.col and "pcs" in mt.col:
    mt = mt.annotate_cols(
        pc0 = mt.pcs[0],
        pc1 = mt.pcs[1],
        pc2 = mt.pcs[2],
        pc3 = mt.pcs[3],
        pc4 = mt.pcs[4],
    )

# Disease duration
mt = mt.annotate_cols(
    disease_duration = mt.age_exact - mt.age_of_Dx
)

# ============================================================
# 1b. ADD VARIANT METADATA TO MATRIX TABLE (REQUIRED)
# ============================================================

mt = hl.variant_qc(mt)

mt = mt.annotate_rows(
    rsid = mt.rsid,
    chrom = mt.locus.contig,
    pos = mt.locus.position,
    ref = mt.alleles[0],
    alt = mt.alleles[1],
    AF = mt.variant_qc.AF[1]
)


# Sample size
n_samples = mt.count_cols()

# ============================================================
# 2. DEFINE COVARIATES AND PHENOTYPES (PD SEVERITY)
# ============================================================
# Scientific purpose:
#   Adjust for demographic and population structure confounders.
#   Phenotypes:
#     - tremor_score: motor severity
#     - ci_unique_count: CI severity (number of distinct CI diagnoses)
#     - ci_binary: presence/absence of any CI diagnosis

base_covs = [
    1.0,                # intercept
    mt.age_exact,
    mt.sex_bin,
    mt.BMI_mean,
    mt.pc0, mt.pc1, mt.pc2, mt.pc3, mt.pc4
]


# ============================================================
# 3. HELPER: RUN GWAS (PD SEVERITY) — FINAL VERSION
# ============================================================

def run_gwas(phenotype_name, y, covariates, label):
    print(f"\nRunning GWAS: {phenotype_name} | model: {label}")
    sys.stdout.flush()

    gwas = hl.linear_regression_rows(
        y = y,
        x = mt.GT.n_alt_alleles(),
        covariates = covariates,
        # pass_through must be a SEQUENCE of row field names
        pass_through = ['rsid', 'chrom', 'pos', 'ref', 'alt', 'AF']
    ).annotate(
        phenotype = phenotype_name,
        model = label,
        n = n_samples
    )

    out_ht = f"{pd_out_dir}/{phenotype_name}_{label}.ht"
    out_tsv = f"{pd_out_dir}/{phenotype_name}_{label}.tsv.bgz"

    gwas = gwas.checkpoint(out_ht, overwrite=True)
    gwas.export(out_tsv)

    print(f"✔ Saved: {out_tsv}")
    sys.stdout.flush()





# ============================================================
# 4. RUN ALL GWAS MODELS (PD COHORT DISCOVERY)
# ============================================================

# Model 1: Tremor base
run_gwas(
    phenotype_name="tremor_score",
    y=mt.tremor_score,
    covariates=base_covs,
    label="base"
)

# Model 2: CI unique base
run_gwas(
    phenotype_name="ci_unique_count",
    y=mt.ci_unique_count,
    covariates=base_covs,
    label="base"
)

# Model 3: CI binary base
run_gwas(
    phenotype_name="ci_binary",
    y=mt.ci_binary,
    covariates=base_covs,
    label="base"
)

# Model 4: Tremor adj CI unique
run_gwas(
    phenotype_name="tremor_score",
    y=mt.tremor_score,
    covariates=base_covs + [mt.ci_unique_count],
    label="adj_CIunique"
)

# Model 5: Tremor adj CI binary
run_gwas(
    phenotype_name="tremor_score",
    y=mt.tremor_score,
    covariates=base_covs + [mt.ci_binary],
    label="adj_CIbinary"
)

# Model 6: CI unique adj tremor
run_gwas(
    phenotype_name="ci_unique_count",
    y=mt.ci_unique_count,
    covariates=base_covs + [mt.tremor_score],
    label="adj_tremor"
)

# Model 7: CI binary adj tremor
run_gwas(
    phenotype_name="ci_binary",
    y=mt.ci_binary,
    covariates=base_covs + [mt.tremor_score],
    label="adj_tremor"
)





# ============================================================
# 5. MODEL-LEVEL STATS AND HIT COUNTS (HTML)
# ============================================================
# Scientific purpose:
#   Summarize GWAS model behavior:
#     - Genomic inflation (lambda GC)
#     - Number of genome-wide and suggestive hits
#   This helps assess quality and detect potential confounding.

results = []

for path in fs.glob(f"{pd_out_dir}/*.tsv.bgz"):
    with fs.open(path, "rb") as f:
        df = pd.read_csv(f, sep="\t", compression="gzip")

    # Skip empty files
    if df.empty:
        continue

    pheno = df["phenotype"].iloc[0]
    model = df["model"].iloc[0]

    chisq = (df["beta"] ** 2) / (df["standard_error"] ** 2)
    lambda_gc = chisq.median() / 0.456

    n_hits = (df["p_value"] < 5e-8).sum()
    n_sugg = (df["p_value"] < 1e-6).sum()   # consistent threshold

    results.append({
        "phenotype": pheno,
        "model": model,
        "lambda_gc": lambda_gc,
        "n_hits_5e-8": n_hits,
        "n_hits_1e-6": n_sugg
    })

stats_df = pd.DataFrame(results)
stats_html_path = f"{pd_out_dir}/model_stats.html"
stats_html = stats_df.to_html(index=False)

with fs.open(stats_html_path, "w") as f:
    f.write(stats_html)

print("\n=== MODEL-LEVEL STATS HTML ===")
print(stats_html)
print("=== END HTML ===\n")








pd_out_dir = "gs://path_to_your_bucket/pd_full_gwas_models_with_CI"



# ============================================================
# 5b. MANHATTAN + QQ PLOTS (AUTO-DETECT PATHS, ENHANCED)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gcsfs
import os
from scipy.stats import chi2

gcs = gcsfs.GCSFileSystem(token="cloud")


# ------------------------------------------------------------
# Auto-detect loader: local OR GCS
# ------------------------------------------------------------
def load_df_auto(path):
    if path.startswith("gs://"):
        with gcs.open(path, "rb") as f:
            return pd.read_csv(f, sep="\t", compression="gzip")

    local_path = path if path.startswith("/") else "/" + path
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"File not found: {local_path}")

    return pd.read_csv(local_path, sep="\t", compression="gzip")


# ------------------------------------------------------------
# Manhattan plot with alternating colors + chromosome labels
# ------------------------------------------------------------
def manhattan_plot(df, title):
    # Normalize chromosome labels
    df["chrom"] = df["chrom"].astype(str).str.replace("chr", "", regex=False)
    df["chrom"] = df["chrom"].astype(int)

    df = df.dropna(subset=["p_value"])
    df["minuslog10p"] = -np.log10(df["p_value"])

    # Build cumulative genomic position
    chroms = sorted(df["chrom"].unique())
    chrom_offsets = {}
    offset = 0
    for c in chroms:
        chrom_offsets[c] = offset
        offset += df[df["chrom"] == c]["pos"].max()

    df["pos_cum"] = df.apply(lambda r: r["pos"] + chrom_offsets[r["chrom"]], axis=1)

    # Alternating colors
    colors = ["#4C72B0", "#DD8452"]

    plt.figure(figsize=(14,5))
    for i, c in enumerate(chroms):
        subset = df[df["chrom"] == c]
        plt.scatter(
            subset["pos_cum"],
            subset["minuslog10p"],
            s=4,
            color=colors[i % 2]
        )

    # Genome-wide significance line
    plt.axhline(-np.log10(5e-8), color="red", linestyle="--")

    # Chromosome labels
    ticks = []
    labels = []
    for c in chroms:
        mid = df[df["chrom"] == c]["pos_cum"].median()
        ticks.append(mid)
        labels.append(str(c))

    plt.xticks(ticks, labels)
    plt.xlabel("Chromosome")
    plt.ylabel("-log10(p)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# QQ plot with correct λGC printed on the plot
# ------------------------------------------------------------
def qq_plot(df, title):
    df = df[df["p_value"] > 0].sort_values("p_value")
    n = len(df)

    expected = -np.log10((np.arange(1, n+1)) / (n+1))
    observed = -np.log10(df["p_value"].values)

    # Correct λGC calculation
    chisq = chi2.isf(df["p_value"], df=1)
    lambda_gc = np.median(chisq) / 0.4549364

    plt.figure(figsize=(6,6))
    plt.scatter(expected, observed, s=4)
    plt.plot([0, expected.max()], [0, expected.max()], color="red")

    plt.xlabel("Expected -log10(p)")
    plt.ylabel("Observed -log10(p)")
    plt.title(title)

    # Print λGC on the plot
    plt.text(
        0.05,
        0.95,
        f"λGC = {lambda_gc:.3f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top"
    )

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Wrapper to load + plot
# ------------------------------------------------------------
def load_and_plot(phenotype, label):
    path = f"{pd_out_dir}/{phenotype}_{label}.tsv.bgz"
    print(f"\n=== {phenotype} ({label}) ===")

    df = load_df_auto(path)

    manhattan_plot(df, f"Manhattan: {phenotype} ({label})")
    qq_plot(df, f"QQ Plot: {phenotype} ({label})")


# ============================================================
# Generate plots for all 7 models
# ============================================================

load_and_plot("tremor_score", "base")

#load_and_plot("ci_unique_count", "base")
#load_and_plot("ci_binary", "base")
#load_and_plot("tremor_score", "adj_CIunique")
#load_and_plot("tremor_score", "adj_CIbinary")
#load_and_plot("ci_unique_count", "adj_tremor")
#load_and_plot("ci_binary", "adj_tremor")




















# ============================================================
# 6. HIGH-CONFIDENCE HITS ACROSS MODELS (p < 1e-6) (HTML)
# ============================================================

gw_hits = []

for path in fs.glob(f"{pd_out_dir}/*.tsv.bgz"):
    with fs.open(path, "rb") as f:
        df = pd.read_csv(f, sep="\t", compression="gzip")

    if df.empty:
        continue

    pheno = df["phenotype"].iloc[0]
    model = df["model"].iloc[0]

    # Filter to high-confidence hits
    hits = df[df["p_value"] < 1e-6].copy()
    if hits.empty:
        continue

    hits["phenotype"] = pheno
    hits["model"] = model

    # Optional OR (correct placement + indentation)
    if "beta" in hits.columns and hits["beta"].notna().any():
        hits["odds_ratio"] = np.exp(hits["beta"])

    # Normalize AF column name
    for col in ["AF", "MAF", "allele_frequency"]:
        if col in hits.columns:
            hits["AF"] = hits[col]
            break

    # Keep only relevant columns
    keep_cols = [
        "phenotype", "model",
        "rsid", "chrom", "pos", "ref", "alt",
        "beta", "standard_error", "p_value", "odds_ratio",
        "AF", "n"
    ]

    keep_cols = [c for c in keep_cols if c in hits.columns]
    hits = hits[keep_cols]

    # THIS must be aligned with the other blocks
    gw_hits.append(hits)

# Write HTML
if gw_hits:
    gw_hits_df = pd.concat(gw_hits, ignore_index=True)
    gw_html_path = f"{pd_out_dir}/high_confidence_hits_1e6.html"
    gw_html = gw_hits_df.to_html(index=False)

    with fs.open(gw_html_path, "w") as f:
        f.write(gw_html)

    print("\n=== HIGH-CONFIDENCE HITS (p < 1e-6) HTML ===")
    print(gw_html)
    print("=== END HTML ===\n")
else:
    print("\nNo high-confidence hits at p < 1e-6.")



# 7. Omitted





# ============================================================
# 8. EFFECT-SIZE COMPARISON ACROSS ALL MODELS (HTML)
# ============================================================

def load_full_results(phenotype, model):
    path = f"{pd_out_dir}/{phenotype}_{model}.tsv.bgz"
    if not fs.exists(path):
        return pd.DataFrame()
    with fs.open(path, "rb") as f:
        return pd.read_csv(f, sep="\t", compression="gzip")

key_col = "locus"

# Load full results
tremor = load_full_results("tremor_score", "base")
ciuniq = load_full_results("ci_unique_count", "base")
cibin  = load_full_results("ci_binary", "base")

if tremor.empty and ciuniq.empty and cibin.empty:
    print("\nCannot compute effect-size comparison (no model results found).\n")

else:
    # Merge all three models on locus
    merged = tremor.merge(ciuniq, on=key_col, how="outer", suffixes=("_tremor", "_ciuniq"))
    merged = merged.merge(cibin, on=key_col, how="outer", suffixes=("", "_cibin"))

    # Rename CI-binary columns if needed
    if "beta" in merged.columns:
        merged = merged.rename(columns={"beta": "beta_cibin", "p_value": "p_value_cibin"})

    # Indicator columns
    merged["in_tremor"] = (~merged["beta_tremor"].isna()).astype(int)
    merged["in_ciuniq"] = (~merged["beta_ciuniq"].isna()).astype(int)
    merged["in_cibin"]  = (~merged["beta_cibin"].isna()).astype(int)

    # Filter to SNPs significant in ANY model
    merged_sig = merged[
        (merged["p_value_tremor"] < 1e-6) |
        (merged["p_value_ciuniq"] < 1e-6) |
        (merged["p_value_cibin"] < 1e-6)
    ].copy()

    if merged_sig.empty:
        print("\nNo SNPs meet the p < 1e-6 threshold in any model.\n")
    else:
        # Order columns
        cols = [
            key_col,
            "in_tremor", "beta_tremor", "p_value_tremor",
            "in_ciuniq", "beta_ciuniq", "p_value_ciuniq",
            "in_cibin",  "beta_cibin",  "p_value_cibin"
        ]
        merged_sig = merged_sig[cols]

        # Print HTML to screen
        html = merged_sig.to_html(index=False)
        print("\n=== EFFECT-SIZE COMPARISON ACROSS MODELS (p < 1e-6) ===")
        print(html)
        print("=== END EFFECT-SIZE COMPARISON ===\n")





# ============================================================
# 9. SLIM TSVs FOR PLOTTING (chrom, pos, p_value, -log10p)
# ============================================================

models = ["base", "adj_tremor", "adj_CIunique", "adj_CIbinary"]
phenos = ["tremor_score", "ci_unique_count", "ci_binary"]

for pheno in phenos:
    for model in models:

        path = f"{pd_out_dir}/{pheno}_{model}.tsv.bgz"
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path, sep="\t", compression="gzip")

        # Require chrom, pos, p_value
        if not {"chrom", "pos", "p_value"}.issubset(df.columns):
            continue

        slim = df[["chrom", "pos", "p_value"]].copy()

        # Normalize chromosome labels
        slim["chrom"] = (
            slim["chrom"]
            .astype(str)
            .str.replace("chr", "", regex=False)
        )

        # Add -log10(p)
        slim["minuslog10p"] = -np.log10(slim["p_value"])

        # Optional: keep locus if present
        if "locus" in df.columns:
            slim["locus"] = df["locus"]

        slim_path = os.path.join(pd_out_dir, f"{pheno}_{model}_for_plots.tsv")
        slim.to_csv(slim_path, sep="\t", index=False)

        print(f"✔ Slim TSV for plotting written to: {slim_path}")










#################################################################################################
# ============================================================
# V1. LOAD DATASETS + RESTRICT CC TO SAME PD INDIVIDUALS
# ============================================================
# SCIENTIFIC QUESTION:
#   "How do we ensure the CC CI models use the *same PD individuals*
#    as the PD CI models, despite independent QC pipelines?"
#
#   → We restrict CC to PD sample IDs from the PD dataset.
#   → This aligns sample sets and makes the models comparable.

print("=== V1: Load PD + CC and restrict CC to PD dataset IDs ===")

# Correct paths from your PD pipeline
pd_mt_path = f"{bucket_base}/PD_with_PCA_and_CI.mt"
cc_mt_path = f"{bucket_base}/CC_with_PCA_and_CI.mt"
pd_out_dir = f"{bucket_base}/pd_full_gwas_models_with_CI"

# Load PD and CC MatrixTables
pd_mt = hl.read_matrix_table(pd_mt_path)
cc_mt = hl.read_matrix_table(cc_mt_path)

# Extract PD sample IDs from PD dataset
pd_ids = pd_mt.s.collect()
pd_ids_set = hl.literal(set(pd_ids))

print(f"PD dataset sample count: {pd_mt.count_cols()}")
print(f"PD IDs extracted: {len(pd_ids)}")

# Restrict CC to those same PD individuals
cc_pd_subset = cc_mt.filter_cols(pd_ids_set.contains(cc_mt.s))

# CC controls = Non_PD (pheno_bin == 0)
cc_controls = cc_mt.filter_cols(cc_mt.pheno_bin == 0)

# Combine restricted PD + all CC controls
cc_restricted = cc_pd_subset.union_cols(cc_controls)

# ------------------------------------------------------------
# DIAGNOSTIC CHECKS (SCIENTIFIC PURPOSE):
#   "Do the PD individuals in CC match the PD individuals in PD_mt?"
#   "How many CC controls are available?"
#   "Is the final CC dataset the size we expect?"
# ------------------------------------------------------------

print("PD in CC after restriction:", cc_pd_subset.count_cols())
print("Controls in CC:", cc_controls.count_cols())
print("Final CC restricted:", cc_restricted.count_cols())


# ============================================================
# V2. MODEL A — CC CI GWAS EXCLUDING PD CASES (WITH SAFE BMI)
# ============================================================
# SCIENTIFIC QUESTION:
#   "What is the genetic architecture of CI in the general population,
#    *excluding* PD cases entirely?"
#
#   → This isolates CI genetics independent of PD.

print("\n=== V2: Running CC CI GWAS (exclude PD cases) ===")

# Non-PD only
cc_noPD = cc_restricted.filter_cols(cc_restricted.pheno_bin == 0)

# Variant QC
cc_noPD = hl.variant_qc(cc_noPD)

# Safely cast BMI_mean (string) to float, treating "" or "NA" as missing
cc_noPD = cc_noPD.annotate_cols(
    BMI_num = hl.if_else(
        (cc_noPD.BMI_mean == "") | (cc_noPD.BMI_mean == "NA"),
        hl.null(hl.tfloat64),
        hl.float64(cc_noPD.BMI_mean)
    )
)

print("N cols (Model A, after BMI cast):", cc_noPD.count_cols())

cc_ci_noPD = hl.linear_regression_rows(
    y = cc_noPD.ci_unique_count,
    x = cc_noPD.GT.n_alt_alleles(),
    covariates = [
        1.0,
        cc_noPD.age_exact,
        cc_noPD.sex_bin,
        cc_noPD.BMI_num,
        cc_noPD.pcs[0], cc_noPD.pcs[1], cc_noPD.pcs[2], cc_noPD.pcs[3], cc_noPD.pcs[4]
    ]
).annotate(
    phenotype = "ci_unique_count",
    model = "CC_noPD"
)

print("Completed Model A (exclude PD, with BMI).")


# ============================================================
# V3. MODEL B — CC CI GWAS ADJUSTING FOR PD STATUS (WITH SAFE BMI)
# ============================================================
# SCIENTIFIC QUESTION:
#   "Does PD status confound or mediate the genetic architecture of CI?"
#
#   → This model includes PD cases but adjusts for PD status.

print("\n=== V3: Running CC CI GWAS (adjust for PD status) ===")

cc_adjPD = cc_restricted

# Variant QC
cc_adjPD = hl.variant_qc(cc_adjPD)

# Safely cast BMI_mean
cc_adjPD = cc_adjPD.annotate_cols(
    BMI_num = hl.if_else(
        (cc_adjPD.BMI_mean == "") | (cc_adjPD.BMI_mean == "NA"),
        hl.null(hl.tfloat64),
        hl.float64(cc_adjPD.BMI_mean)
    )
)

print("N cols (Model B, after BMI cast):", cc_adjPD.count_cols())

cc_ci_adjPD = hl.linear_regression_rows(
    y = cc_adjPD.ci_unique_count,
    x = cc_adjPD.GT.n_alt_alleles(),
    covariates = [
        1.0,
        cc_adjPD.age_exact,
        cc_adjPD.sex_bin,
        cc_adjPD.BMI_num,
        cc_adjPD.pheno_bin,   # PD status (0 = Non_PD, 1 = PD)
        cc_adjPD.pcs[0], cc_adjPD.pcs[1], cc_adjPD.pcs[2], cc_adjPD.pcs[3], cc_adjPD.pcs[4]
    ]
).annotate(
    phenotype = "ci_unique_count",
    model = "CC_adjPD"
)

print("Completed Model B (adjust PD, with BMI).")


# ============================================================
# V4. ADD ALLELE + AF/MAF + rsID ANNOTATIONS
# ============================================================
# SCIENTIFIC QUESTION:
#   "How do we make the summary statistics analysis-ready for
#    Manhattan/QQ plots, FUMA, LocusZoom, and meta-analysis?"
#
#   → Add allele fields, AF, MAF, and rsID.

print("\n=== V4: Adding allele + AF/MAF + rsid annotations ===")

def extract_qc_table(mt):
    mt_clean = mt.select_cols()
    ht = mt_clean.rows()
    ht = ht.select(
        AF = ht.variant_qc.AF[1],
        MAF = hl.min(ht.variant_qc.AF[1], 1 - ht.variant_qc.AF[1]),
        rsid = ht.rsid
    )
    return ht

qc_noPD = extract_qc_table(cc_noPD)
qc_adjPD = extract_qc_table(cc_adjPD)

cc_ci_noPD = cc_ci_noPD.annotate(**qc_noPD[cc_ci_noPD.key])
cc_ci_adjPD = cc_ci_adjPD.annotate(**qc_adjPD[cc_ci_adjPD.key])

def add_allele_fields(ht):
    return ht.annotate(
        chrom = ht.locus.contig,
        pos = ht.locus.position,
        ref = ht.alleles[0],
        alt = ht.alleles[1],
        effect_allele = ht.alleles[1],
        other_allele = ht.alleles[0]
    )

cc_ci_noPD = add_allele_fields(cc_ci_noPD)
cc_ci_adjPD = add_allele_fields(cc_ci_adjPD)

print("Completed V4 annotations.")




# ============================================================
# V5. EXPORT RESULTS FOR BOTH MODELS (INCLUDING rsID)
# ============================================================
# SCIENTIFIC QUESTION:
#   "Do the two CC CI models produce different association architectures?"
#
#   → Export both for Manhattan/QQ comparison and overlap with PD CI loci.

print("\n=== V5: Exporting CC CI summary stats for both models ===")

# Define CC output directory (no need to create it explicitly)
cc_out_dir = f"{pd_out_dir}/CC_CI_models"

out_noPD = f"{cc_out_dir}/CC_CI_noPD.tsv.bgz"
out_adjPD = f"{cc_out_dir}/CC_CI_adjPD.tsv.bgz"

export_fields = [
    "chrom", "pos", "ref", "alt",
    "beta", "standard_error", "p_value",
    "phenotype", "model",
    "effect_allele", "other_allele",
    "AF", "MAF",
    "rsid"
]

cc_ci_noPD.select(*export_fields).export(out_noPD)
cc_ci_adjPD.select(*export_fields).export(out_adjPD)

print(f"Exported Model A (exclude PD): {out_noPD}")
print(f"Exported Model B (adjust PD): {out_adjPD}")




# ============================================================
# V6. UNIFIED SUMMARY OF ALL GWAS RESULTS (PD + CC)
# ============================================================
# SCIENTIFIC QUESTION:
#   "Across all PD and CC models, how inflated are the tests,
#    and how many hits do we see per phenotype/model?"
#
#   → One table with λGC, hit counts, n, and file paths
# ============================================================

import pandas as pd
import numpy as np
import gcsfs
from scipy.stats import chi2

fs = gcsfs.GCSFileSystem(token="cloud")

summary_rows = []

print("=== V6: Building unified GWAS summary across all models ===")

# Directories to scan
dirs_to_scan = [
    pd_out_dir,                 # PD models
    f"{pd_out_dir}/CC_CI_models"  # CC models
]

for d in dirs_to_scan:
    for path in fs.glob(f"{d}/*.tsv.bgz"):

        with fs.open(path, "rb") as f:
            df = pd.read_csv(f, sep="\t", compression="gzip")

        # Skip non-GWAS files
        required = {"beta", "standard_error", "p_value"}
        if not required.issubset(df.columns):
            continue

        if df.empty:
            continue

        phenotype = df["phenotype"].iloc[0]
        model     = df["model"].iloc[0]
        n         = df["n"].iloc[0] if "n" in df.columns else None

        # λGC
        chisq = chi2.isf(df["p_value"], df=1)
        lambda_gc = np.median(chisq) / 0.4549364

        # Hit counts
        n_hits_5e8 = (df["p_value"] < 5e-8).sum()
        n_hits_1e6 = (df["p_value"] < 1e-6).sum()

        summary_rows.append({
            "phenotype": phenotype,
            "model": model,
            "n_samples": n,
            "lambda_gc": lambda_gc,
            "hits_5e-8": n_hits_5e8,
            "hits_1e-6": n_hits_1e6,
            "path": path
        })

summary_df = pd.DataFrame(summary_rows).sort_values(["phenotype", "model"])

summary_path = f"{pd_out_dir}/gwas_summary_all_models.tsv"
with fs.open(summary_path, "w") as f:
    summary_df.to_csv(f, sep="\t", index=False)

print("\n=== V6: Unified GWAS Summary ===")
print(summary_df)
print("\nSaved summary to:", summary_path)

########################################################################################














# ============================================================
# 10. PD RISK GWAS IN LARGE CASE–CONTROL DATASET
# ============================================================

# Load CC MatrixTable
cc_mt = hl.read_matrix_table(cc_mt_path)

# ============================================================
# 10a. ADD VARIANT QC (CRITICAL — THIS ADDS AF/MAF)
# ============================================================
cc_mt = hl.variant_qc(cc_mt)

# Annotate allele fields + AF/MAF
cc_mt = cc_mt.annotate_rows(
    effect_allele = cc_mt.alleles[1],
    other_allele = cc_mt.alleles[0],
    chrom = cc_mt.locus.contig,
    pos = cc_mt.locus.position,
    ref = cc_mt.alleles[0],
    alt = cc_mt.alleles[1],
    AF = cc_mt.variant_qc.AF[1],
    MAF = hl.min(cc_mt.variant_qc.AF[1], 1 - cc_mt.variant_qc.AF[1])
)







# ============================================================
# 10b. LOGISTIC REGRESSION
# ============================================================

print("\nRunning PD risk GWAS in large case–control dataset...")

cc_gwas = hl.logistic_regression_rows(
    test="wald",
    y=cc_mt.pheno_bin,
    x=cc_mt.GT.n_alt_alleles(),
    covariates=[
        1.0,
        cc_mt.age_exact,
        cc_mt.sex_bin,
        cc_mt.pcs[0], cc_mt.pcs[1], cc_mt.pcs[2], cc_mt.pcs[3], cc_mt.pcs[4]
    ]
).annotate(
    phenotype="PD_risk",
    model="logistic_base"
)




# ============================================================
# 10c. ADD CASE/CONTROL COUNTS
# ============================================================
case_control_counts = cc_mt.aggregate_cols(hl.struct(
    n_cases = hl.agg.sum(cc_mt.pheno_bin),
    n_controls = hl.agg.sum(1 - cc_mt.pheno_bin)
))

cc_gwas = cc_gwas.annotate(
    n_cases = case_control_counts.n_cases,
    n_controls = case_control_counts.n_controls
)





# ============================================================
# 10d. Annotate rsIDs for PD CI, CC CI, and PD-risk GWAS
# ============================================================

print("\n=== STEP 10d: Annotating rsIDs for all GWAS datasets ===")

# Load dbSNP once
dbsnp = hl.experimental.load_dataset(
    'dbSNP_rsid',
    version='154',
    reference_genome='GRCh38'
)

# ------------------------------------------------------------
# Load PD CI and CC CI GWAS results as Hail Tables
# ------------------------------------------------------------

pd_ci_path = f"{pd_out_dir}/ci_unique_count_base.tsv.bgz"
pd_ci = hl.import_table(
    pd_ci_path,
    impute=True,
    force=True,
    types={"pos": hl.tint, "p_value": hl.tfloat64}
)

cc_ci_path = f"{pd_out_dir}/ci_binary_base.tsv.bgz"
cc_ci = hl.import_table(
    cc_ci_path,
    impute=True,
    force=True,
    types={"pos": hl.tint, "p_value": hl.tfloat64}
)

# ------------------------------------------------------------
# Reconstruct locus + alleles for dbSNP lookup
# ------------------------------------------------------------

pd_ci = pd_ci.annotate(
    locus = hl.locus(pd_ci.chrom, pd_ci.pos, reference_genome="GRCh38"),
    alleles = [pd_ci.ref, pd_ci.alt]
).key_by("locus")

cc_ci = cc_ci.annotate(
    locus = hl.locus(cc_ci.chrom, cc_ci.pos, reference_genome="GRCh38"),
    alleles = [cc_ci.ref, cc_ci.alt]
).key_by("locus")

# ------------------------------------------------------------
# Annotate rsIDs
# ------------------------------------------------------------

pd_ci = pd_ci.annotate(
    rsid = dbsnp[pd_ci.locus, pd_ci.alleles].rsid
)
print("Annotated rsIDs for PD CI.")

cc_ci = cc_ci.annotate(
    rsid = dbsnp[cc_ci.locus, cc_ci.alleles].rsid
)
print("Annotated rsIDs for CC CI.")

cc_gwas = cc_gwas.annotate(
    rsid = dbsnp[cc_gwas.locus, cc_gwas.alleles].rsid
)
print("Annotated rsIDs for PD-risk GWAS.")




cc_gwas.export(f"{pd_out_dir}/PD_risk_logistic_base.tsv.bgz")





# ============================================================
# 11. Export Clean PD CI Summary Stats  (FINAL VERSION)
# ============================================================

print("\nExporting PD CI summary stats...")

# ------------------------------------------------------------
# Load the PD CI Hail Table (confirmed from bucket listing)
# ------------------------------------------------------------
pd_ci_ht_path = f"{bucket_base}/pd_full_gwas_models_with_CIunique/ci_unique_count_base.ht"
pd_ci = hl.read_table(pd_ci_ht_path)

# ------------------------------------------------------------
# Annotate rsIDs (dbsnp must already be loaded from Step 10d)
# ------------------------------------------------------------
pd_ci = pd_ci.annotate(
    rsid = dbsnp[pd_ci.locus, pd_ci.alleles].rsid
)

# ------------------------------------------------------------
# Add allele + locus fields needed for export
# ------------------------------------------------------------
pd_ci = pd_ci.annotate(
    effect_allele = pd_ci.alleles[1],
    other_allele = pd_ci.alleles[0],
    chrom = pd_ci.locus.contig,
    pos = pd_ci.locus.position,
    ref = pd_ci.alleles[0],
    alt = pd_ci.alleles[1]
)

# ------------------------------------------------------------
# Add AF / MAF if present
# ------------------------------------------------------------
if "AF" in pd_ci.row:
    pd_ci = pd_ci.annotate(
        AF = pd_ci.AF,
        MAF = hl.min(pd_ci.AF, 1 - pd_ci.AF)
    )

# ------------------------------------------------------------
# Prepare export fields
# ------------------------------------------------------------
pd_ci = pd_ci.key_by()

pd_ci_export = {
    "chrom": pd_ci.chrom,
    "pos": pd_ci.pos,
    "ref": pd_ci.ref,
    "alt": pd_ci.alt,
    "beta": pd_ci.beta,
    "standard_error": pd_ci.standard_error,
    "p_value": pd_ci.p_value,
    "phenotype": pd_ci.phenotype,
    "model": pd_ci.model,
    "effect_allele": pd_ci.effect_allele,
    "other_allele": pd_ci.other_allele,
    "rsid": pd_ci.rsid
}

if "AF" in pd_ci.row:
    pd_ci_export["AF"] = pd_ci.AF
    pd_ci_export["MAF"] = pd_ci.MAF

# ------------------------------------------------------------
# Export final PD CI summary stats
# ------------------------------------------------------------
pd_ci_out = os.path.join(pd_out_dir, "PD_CI_summary_stats.tsv.bgz")
pd_ci.select(**pd_ci_export).export(pd_ci_out)

print(f"PD CI summary stats written to: {pd_ci_out}")








# ============================================================
# 12. Export Clean CC CI Summary Stats  (FINAL VERSION)
# ============================================================

print("\nExporting CC CI summary stats...")

# Load correct CC CI Hail Table
cc_ci_ht_path = f"{bucket_base}/pd_full_gwas_models_with_CIunique/ci_binary_base.ht"
cc_ci = hl.read_table(cc_ci_ht_path)

# Annotate rsIDs
cc_ci = cc_ci.annotate(
    rsid = dbsnp[cc_ci.locus, cc_ci.alleles].rsid
)

# Add allele + locus fields
cc_ci = cc_ci.annotate(
    effect_allele = cc_ci.alleles[1],
    other_allele = cc_ci.alleles[0],
    chrom = cc_ci.locus.contig,
    pos = cc_ci.locus.position,
    ref = cc_ci.alleles[0],
    alt = cc_ci.alleles[1]
)

# Add AF/MAF if present
if "AF" in cc_ci.row:
    cc_ci = cc_ci.annotate(
        AF = cc_ci.AF,
        MAF = hl.min(cc_ci.AF, 1 - cc_ci.AF)
    )

cc_ci = cc_ci.key_by()

cc_ci_export = {
    "chrom": cc_ci.chrom,
    "pos": cc_ci.pos,
    "ref": cc_ci.ref,
    "alt": cc_ci.alt,
    "beta": cc_ci.beta,
    "standard_error": cc_ci.standard_error,
    "p_value": cc_ci.p_value,
    "phenotype": cc_ci.phenotype,
    "model": cc_ci.model,
    "effect_allele": cc_ci.effect_allele,
    "other_allele": cc_ci.other_allele,
    "rsid": cc_ci.rsid
}

if "AF" in cc_ci.row:
    cc_ci_export["AF"] = cc_ci.AF
    cc_ci_export["MAF"] = cc_ci.MAF

cc_ci_out = os.path.join(pd_out_dir, "CC_CI_summary_stats.tsv.bgz")
cc_ci.select(**cc_ci_export).export(cc_ci_out)

print(f"CC CI summary stats written to: {cc_ci_out}")
















print("=== LOADING PD & CC CI SUMMARY STATS ===")

# Load summary stats for both models
pd_ci_path = f"{pd_out_dir}/PD_CI_summary_stats.tsv.bgz"
cc_ci_path = f"{pd_out_dir}/CC_CI_summary_stats.tsv.bgz"

pd_ci = hl.import_table(pd_ci_path, impute=True)
cc_ci = hl.import_table(cc_ci_path, impute=True)

# ============================================================
# TABLE 1 — RAW HTML: MODEL SUMMARY (PD CI + CC CI)
# ============================================================

print("\n=== TABLE 1: RAW HTML — MODEL SUMMARY (PD CI + CC CI) ===")

# PD CI stats
pd_lambda = hl.methods.lambda_gc(pd_ci.p_value)
pd_gws = pd_ci.filter(pd_ci.p_value < 5e-8).count()
pd_sugg = pd_ci.filter(pd_ci.p_value < 1e-6).count()
pd_total = pd_ci.count()   # number of variants in PD CI model

# CC CI stats
cc_lambda = hl.methods.lambda_gc(cc_ci.p_value)
cc_gws = cc_ci.filter(cc_ci.p_value < 5e-8).count()
cc_sugg = cc_ci.filter(cc_ci.p_value < 1e-6).count()
cc_total = cc_ci.count()   # number of variants in CC CI model

# Build pandas DataFrame for HTML export
import pandas as pd

df1 = pd.DataFrame([
    {
        "phenotype": "ci_unique_count",
        "model": "PD_CI",
        "lambda_gc": pd_lambda,
        "n_hits_5e-8": pd_gws,
        "n_hits_1e6": pd_sugg,
        "n_variants_total": pd_total
    },
    {
        "phenotype": "ci_unique_count",
        "model": "CC_CI",
        "lambda_gc": cc_lambda,
        "n_hits_5e-8": cc_gws,
        "n_hits_1e6": cc_sugg,
        "n_variants_total": cc_total
    }
])

# Convert to raw HTML and print
html1 = df1.to_html(index=False)
print(html1)


# ============================================================
# TABLE 2 — RAW HTML: PD CI GENOME-WIDE + SUGGESTIVE HITS
# ============================================================

print("\n=== TABLE 2: RAW HTML — PD CI GENOME-WIDE + SUGGESTIVE HITS ===")

# 1. Force scientific notation for small p-values
pd.set_option('display.float_format', lambda x: f'{x:.2e}')

# 2. Drop alleles from the key; key only by locus
pd_ci = pd_ci.key_by('locus')

# 3. Extract PD MatrixTable rows (keyed by locus)
pd_mt_rows = pd_mt.rows()

# 4. Annotate AF + MAF from PD MatrixTable rows
pd_ci = pd_ci.annotate(
    AF = pd_mt_rows[pd_ci.locus].variant_qc.AF[1],
    MAF = hl.min(pd_mt_rows[pd_ci.locus].variant_qc.AF[1],
                 1 - pd_mt_rows[pd_ci.locus].variant_qc.AF[1])
)

# 5. Add OR
pd_ci = pd_ci.annotate(
    OR = hl.exp(pd_ci.beta)
)

# 6. Filter to genome-wide + suggestive hits
pd_hits = pd_ci.filter(pd_ci.p_value < 1e-6)

# 7. Convert to Pandas + HTML
html2 = pd_hits.to_pandas().to_html(index=False)
print(html2)




# ============================================================
# TABLE 3 — RAW HTML: CC CI GENOME-WIDE + SUGGESTIVE HITS
# ============================================================

print("\n=== TABLE 3: RAW HTML — CC CI GENOME-WIDE + SUGGESTIVE HITS ===")

cc_hits = cc_ci.filter(cc_ci.p_value < 1e-6)
html3 = cc_hits.to_pandas().to_html(index=False)
print(html3)






































import hail as hl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------
# 1. Load your actual Hail MatrixTables
# ----------------------------------------------------------

pd_mt = hl.read_matrix_table(f"{bucket_base}/PD_with_PCA_and_CI.mt")
cc_mt = hl.read_matrix_table(f"{bucket_base}/CC_with_PCA_and_CI.mt")

# ----------------------------------------------------------
# 2. Convert column annotations to Pandas DataFrames
# ----------------------------------------------------------

pd_df = pd_mt.cols().to_pandas()
cc_df = cc_mt.cols().to_pandas()

# ----------------------------------------------------------
# 3. Filter CC to healthy controls only
# ----------------------------------------------------------

cc_df = cc_df[cc_df["pheno_bin"] == 0].copy()

# ----------------------------------------------------------
# 4. Clean numeric columns and force dtype
# ----------------------------------------------------------

def force_numeric(df, col):
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df

pd_df = force_numeric(pd_df, "ci_unique_count")
pd_df = force_numeric(pd_df, "tremor_score")
cc_df = force_numeric(cc_df, "ci_unique_count")

# ----------------------------------------------------------
# 5. Add cohort labels
# ----------------------------------------------------------

pd_df["cohort"] = "PD"
cc_df["cohort"] = "CC_noPD"

# ----------------------------------------------------------
# 6. Combine CI_unique_count for Panel A
# ----------------------------------------------------------

ci_df = pd.concat([
    pd_df[["ci_unique_count", "cohort"]],
    cc_df[["ci_unique_count", "cohort"]]
], ignore_index=True)

# ----------------------------------------------------------
# 7. Create the figure layout
# ----------------------------------------------------------

plt.figure(figsize=(14, 6))

# ----------------------------------------------------------
# Panel A: CI_unique_count distribution
# ----------------------------------------------------------
# Panel A: CI_unique_count distribution (integer-aligned bins)
# Panel A: CI_unique_count distribution (integer-aligned bins)
ax1 = plt.subplot(1, 2, 1)

# Determine integer bin edges
max_ci = int(ci_df["ci_unique_count"].max())
bins = np.arange(0, max_ci + 2) - 0.5   # centers bins on integers

plt.hist(
    ci_df[ci_df["cohort"] == "PD"]["ci_unique_count"],
    bins=bins,
    alpha=0.5,
    color="royalblue",
    density=True,
    label="PD"
)

plt.hist(
    ci_df[ci_df["cohort"] == "CC_noPD"]["ci_unique_count"],
    bins=bins,
    alpha=0.5,
    color="seagreen",
    density=True,
    label="Healthy Controls"
)

plt.title("Distribution of CI_unique_count\nPD vs. Healthy Controls")
plt.xlabel("CI_unique_count")
plt.ylabel("Density")
plt.xticks(range(0, max_ci + 1))
plt.legend()

# Add panel label A
ax1.text(-0.1, 1.05, "A", transform=ax1.transAxes,
         fontsize=18, fontweight="bold", va="top", ha="right")


# ----------------------------------------------------------
# Panel B: Tremor severity distribution
# ----------------------------------------------------------
ax2 = plt.subplot(1, 2, 2)

plt.hist(
    pd_df["tremor_score"],
    bins=20,
    alpha=0.6,
    color="darkorange",
    density=True
)

plt.title("Distribution of Tremor Severity\nPD Cohort Only")
plt.xlabel("Tremor Severity Score")
plt.ylabel("Density")

# Add panel label B
ax2.text(-0.1, 1.05, "B", transform=ax2.transAxes,
         fontsize=18, fontweight="bold", va="top", ha="right")

# ----------------------------------------------------------
# Final layout adjustments
# ----------------------------------------------------------
plt.tight_layout()
plt.show()


























































# ============================================================
# FULL MODEL AUDIT: CI-UNIQUE-COUNT BASE MODEL
# ============================================================
# This script:
#   1. Loads the CI-unique-count GWAS model
#   2. Confirms INPUTS (sample size, phenotype distribution, variants tested)
#   3. Computes MODEL-LEVEL DIAGNOSTICS (λGC, chi-square stats)
#   4. Loads EXISTING clumping output and counts independent loci
#   5. Prepares and exports summary stats
#
# IMPORTANT:
#   - This script DOES NOT re-clump.
#   - It simply loads your clumping file and counts the loci.
# ============================================================

import hail as hl
import numpy as np
import os

print("\n=== LOADING CI-UNIQUE-COUNT MODEL ===")

ci_ht_path = f"{bucket_base}/pd_full_gwas_models_with_CIunique/ci_unique_count_base.ht"
ci = hl.read_table(ci_ht_path)

# ------------------------------------------------------------
# 1. CONFIRM INPUTS
# ------------------------------------------------------------

print("\n=== INPUT CHECKS ===")

# Sample size (if stored)
if "n" in ci.row:
    print("Sample size (n):", ci.aggregate(hl.agg.take(ci.n, 1))[0])
else:
    print("Sample size (n): NOT STORED IN TABLE (confirm from phenotype file)")

# Phenotype distribution
print("\nPhenotype distribution:")
print("Numeric phenotype values are not stored in the Hail table.")
print("Use the original phenotype file for distribution statistics.")

# Number of variants tested
num_variants = ci.count()
print("\nVariants tested:", num_variants)


# ------------------------------------------------------------
# 2. MODEL-LEVEL DIAGNOSTICS
# ------------------------------------------------------------

print("\n=== MODEL DIAGNOSTICS ===")

# λGC
lambda_gc = hl.methods.lambda_gc(ci.p_value)
print("λGC:", lambda_gc)

# Chi-square distribution summary
chisq_stats = ci.aggregate(hl.struct(
    chisq_mean = hl.agg.mean((ci.beta / ci.standard_error)**2),
    chisq_median = hl.agg.approx_median((ci.beta / ci.standard_error)**2)
))
print("\nChi-square stats:")
print(chisq_stats)

# ------------------------------------------------------------
# 3. OUTPUT CHECKS
# ------------------------------------------------------------

# NOTE:
# This is the total number of genome-wide significant variants (p < 5e-8)
# BEFORE clumping. This is expected to be larger than the number of
# independent loci (611), which is computed separately from the clumping file.

print("\n=== OUTPUT CHECKS ===")
# Genome-wide significant variants
gws = ci.filter(ci.p_value < 5e-8).count()
print("Genome-wide significant variants:", gws)

# ------------------------------------------------------------
# 4. INDEPENDENT LOCI (LOAD EXISTING CLUMPING OUTPUT)
# ------------------------------------------------------------

print("\n=== INDEPENDENT LOCI (FROM EXISTING CLUMPING OUTPUT) ===")

# Updated path: file is a plain TSV, not BGZIP
clump_path = f"{pd_out_dir}/ci_unique_count_clumped.tsv"

if hl.utils.hadoop_exists(clump_path):
    clump_ht = hl.import_table(clump_path, impute=True)
    num_loci = clump_ht.count()
    print("Independent loci:", num_loci)
else:
    print("Clumping file not found.")

# ------------------------------------------------------------
# 5. PREPARE EXPORT FIELDS
# ------------------------------------------------------------

print("Loading dbSNP (GRCh38, rsID lookup)...")

dbsnp = hl.experimental.load_dataset(
    'dbSNP_rsid',
    version='154',
    reference_genome='GRCh38'
)

print("dbSNP loaded and ready for rsID annotation.")

print("\n=== PREPARING EXPORT ===")

# Annotate rsIDs (dbsnp must be loaded earlier)
ci = ci.annotate(
    rsid = dbsnp[ci.locus, ci.alleles].rsid
)

# Add allele + locus fields
ci = ci.annotate(
    effect_allele = ci.alleles[1],
    other_allele = ci.alleles[0],
    chrom = ci.locus.contig,
    pos = ci.locus.position,
    ref = ci.alleles[0],
    alt = ci.alleles[1]
)

# Add AF/MAF if present
if "AF" in ci.row:
    ci = ci.annotate(
        AF = ci.AF,
        MAF = hl.min(ci.AF, 1 - ci.AF)
    )

ci = ci.key_by()

# Fields to export
export_fields = {
    "chrom": ci.chrom,
    "pos": ci.pos,
    "ref": ci.ref,
    "alt": ci.alt,
    "beta": ci.beta,
    "standard_error": ci.standard_error,
    "p_value": ci.p_value,
    "phenotype": ci.phenotype,
    "model": ci.model,
    "effect_allele": ci.effect_allele,
    "other_allele": ci.other_allele,
    "rsid": ci.rsid
}

if "AF" in ci.row:
    export_fields["AF"] = ci.AF
    export_fields["MAF"] = ci.MAF

# ------------------------------------------------------------
# 6. EXPORT FINAL SUMMARY STATS
# ------------------------------------------------------------

print("\n=== EXPORTING SUMMARY STATS ===")

ci_out = os.path.join(pd_out_dir, "CI_unique_count_summary_stats.tsv.bgz")
ci.select(**export_fields).export(ci_out)

print(f"CI unique count summary stats written to: {ci_out}")

# ------------------------------------------------------------
# 7. CROSS-TRAIT LOCUS OVERLAP (PD CI ∩ CC CI) 
# ------------------------------------------------------------


# ------------------------------------------------------------
# LOAD CC CI CLUMPING OUTPUT
# ------------------------------------------------------------

print("\n=== LOADING CC CI CLUMPING OUTPUT ===")

cc_out_dir = f"{bucket_base}/CC_GWAS_outputs"
cc_clump_path = f"{cc_out_dir}/cc_ci_clumped.tsv"

if hl.utils.hadoop_exists(cc_clump_path):
    cc_clump_ht = hl.import_table(cc_clump_path, impute=True)
    print("Independent CC CI loci:", cc_clump_ht.count())
else:
    raise FileNotFoundError(f"CC CI clumping file not found at: {cc_clump_path}")



print("\n=== CROSS-TRAIT LOCUS OVERLAP (PD CI ∩ CC CI) ===")

# Convert to pandas for overlap logic
pdci_loci = clump_ht.to_pandas()
ccci_loci = cc_clump_ht.to_pandas()

overlaps = []

for _, a in pdci_loci.iterrows():
    for _, b in ccci_loci.iterrows():
        if a["chrom"] == b["chrom"]:
            if not (a["locus_end"] < b["locus_start"] or a["locus_start"] > b["locus_end"]):
                overlaps.append({
                    "chrom": a["chrom"],
                    "pdci_rsid": a["rsid"],
                    "pdci_pos": a["lead_pos"],
                    "pdci_p": a["p"],
                    "ccci_rsid": b["rsid"],
                    "ccci_pos": b["lead_pos"],
                    "ccci_p": b["p"]
                })

overlap_df = pd.DataFrame(overlaps)

print("Number of overlapping loci:", overlap_df.shape[0])
display(overlap_df)







# ============================================================
# END OF FULL MODEL AUDIT
# ============================================================













with hl.hadoop_open(cc_sumstats_path, "rb") as f:
    print(f.read(2))



















# ============================================================
# PRE-STEP 13: Load dbSNP for rsID annotation
# ============================================================

print("Loading dbSNP (GRCh38, rsID lookup)...")

dbsnp = hl.experimental.load_dataset(
    'dbSNP_rsid',
    version='154',
    reference_genome='GRCh38'
)

print("dbSNP loaded and ready for rsID annotation.")


# ============================================================
# 13. Export Clean PD-risk Summary Stats (Case–Control GWAS)
# ============================================================

print("\nExporting PD-risk summary stats...")

# Load the GWAS table
cc_gwas = hl.read_table(
    f"{bucket_base}/pd_full_gwas_models_with_CIunique/case_control_gwas/PD_risk_logistic_base.ht"
)

# Add rsIDs
cc_gwas = cc_gwas.annotate(
    rsid = dbsnp[cc_gwas.locus, cc_gwas.alleles].rsid
)

# Add allele + locus fields
cc_gwas = cc_gwas.annotate(
    effect_allele = cc_gwas.alleles[1],
    other_allele = cc_gwas.alleles[0],
    chrom = cc_gwas.locus.contig,
    pos = cc_gwas.locus.position,
    ref = cc_gwas.alleles[0],
    alt = cc_gwas.alleles[1]
)

# Add AF/MAF if present
if "AF" in cc_gwas.row:
    cc_gwas = cc_gwas.annotate(
        AF = cc_gwas.AF,
        MAF = hl.min(cc_gwas.AF, 1 - cc_gwas.AF)
    )

cc_gwas = cc_gwas.key_by()

# Fields to export
export_fields = {
    "chrom": cc_gwas.chrom,
    "pos": cc_gwas.pos,
    "ref": cc_gwas.ref,
    "alt": cc_gwas.alt,
    "beta": cc_gwas.beta,
    "standard_error": cc_gwas.standard_error,
    "p_value": cc_gwas.p_value,
    "phenotype": cc_gwas.phenotype,
    "model": cc_gwas.model,
    "effect_allele": cc_gwas.effect_allele,
    "other_allele": cc_gwas.other_allele,
    "rsid": cc_gwas.rsid
}

if "AF" in cc_gwas.row:
    export_fields["AF"] = cc_gwas.AF
    export_fields["MAF"] = cc_gwas.MAF

for field in ["n_cases", "n_controls"]:
    if field in cc_gwas.row:
        export_fields[field] = cc_gwas[field]

# ------------------------------------------------------------
# 1. Export UNCOMPRESSED TSV (Hail 0.2.134 cannot bgzip)
# ------------------------------------------------------------
gcs_uncompressed = f"{cc_out_dir}/PD_risk_summary_stats.tsv"

cc_gwas.select(**export_fields).export(
    gcs_uncompressed,
    header=True
)

print(f"Uncompressed summary stats written to: {gcs_uncompressed}")

# ------------------------------------------------------------
# 2. Copy TSV from GCS → local disk
# ------------------------------------------------------------
import subprocess

local_uncompressed = "/tmp/PD_risk_summary_stats.tsv"
local_compressed = "/tmp/PD_risk_summary_stats.tsv.gz"
gcs_compressed = f"{cc_out_dir}/PD_risk_summary_stats.tsv.gz"

print("Copying TSV from GCS to local disk...")
subprocess.run(["gsutil", "cp", gcs_uncompressed, local_uncompressed], check=True)

# ------------------------------------------------------------
# 3. Compress locally with bgzip
# ------------------------------------------------------------
print("Compressing with bgzip...")
subprocess.run(["bgzip", "-f", local_uncompressed], check=True)

# ------------------------------------------------------------
# 4. Copy compressed file back to GCS
# ------------------------------------------------------------
print("Copying compressed file back to GCS...")
subprocess.run(["gsutil", "cp", local_compressed, gcs_compressed], check=True)

print(f"Compressed file written to: {gcs_compressed}")









# ------------------------------------------------------------
# Alternating-color Manhattan plot
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))

# Define two alternating colors
colors = ["steelblue", "darkorange"]

# Track cumulative base-pair position for continuous x-axis
cc_df["ind"] = range(len(cc_df))
cc_df_grouped = cc_df.groupby("chr_numeric")

for i, (chrom, group) in enumerate(cc_df_grouped):
    plt.scatter(
        group["ind"],
        group["minus_log10_p"],
        c=colors[i % 2],
        s=6,
        alpha=0.7,
        label=f"Chr {int(chrom)}"
    )

# Genome-wide significance line
plt.axhline(-np.log10(5e-8), color="red", linestyle="--", linewidth=1)

# X-axis ticks at chromosome centers
ticks = [
    group["ind"].median()
    for _, group in cc_df_grouped
]
labels = [str(int(chrom)) for chrom, _ in cc_df_grouped]

plt.xticks(ticks, labels)
plt.xlabel("Chromosome")
plt.ylabel("-log10(p)")
plt.title("PD Risk GWAS Manhattan Plot (Alternating Colors)")

# Save locally then upload to GCS
local_plot = "/tmp/PD_risk_manhattan.png"
plt.savefig(local_plot, dpi=200, bbox_inches="tight")
plt.show()

gcs_plot = f"{bucket_base}/CC_GWAS_outputs/PD_risk_manhattan.png"
subprocess.run(["gsutil", "cp", local_plot, gcs_plot], check=True)

print(f"PD risk Manhattan plot written to: {gcs_plot}")





# ============================================================
# 15. PD RISK QQ PLOT
# ============================================================

print("\nGenerating PD risk QQ plot...")

if cc_df is not None and "p_value" in cc_df.columns:

    pvals = cc_df["p_value"].dropna()
    pvals = pvals[pvals > 0]

    observed = -np.log10(np.sort(pvals))
    expected = -np.log10(np.linspace(1/len(pvals), 1, len(pvals)))

    plt.figure(figsize=(6, 6))
    plt.scatter(expected, observed, s=8, alpha=0.6)
    plt.plot([0, max(expected)], [0, max(expected)], color="red", linestyle="--")
    plt.xlabel("Expected -log10(p)")
    plt.ylabel("Observed -log10(p)")
    plt.title("PD Risk GWAS QQ Plot")

    # Save locally first
    local_plot = "/tmp/PD_risk_qq.png"
    plt.savefig(local_plot, dpi=200, bbox_inches="tight")
    plt.show()

    # Upload to GCS
    gcs_plot = f"{bucket_base}/CC_GWAS_outputs/PD_risk_qq.png"
    subprocess.run(["gsutil", "cp", local_plot, gcs_plot], check=True)

    print(f"PD risk QQ plot written to: {gcs_plot}")

else:
    print("Skipping PD risk QQ plot. Missing p_value column.")




# ============================================================
# 16. PD RISK ALTERNATING-COLOR MANHATTAN PLOT (FINAL VERSION)
# ============================================================

print("\nGenerating PD risk alternating-color Manhattan plot...")

if cc_df is not None and "p_value" in cc_df.columns:

    # Convert chromosome to numeric (drop X/Y/MT)
    cc_df["chr_numeric"] = pd.to_numeric(
        cc_df["chrom"].astype(str).str.replace("chr", "", regex=False),
        errors="coerce"
    )
    cc_df = cc_df.dropna(subset=["chr_numeric"])

    # Compute -log10(p)
    cc_df["minus_log10_p"] = -np.log10(cc_df["p_value"])

    # Sort by chromosome and position
    cc_df = cc_df.sort_values(["chr_numeric", "pos"])

    # Compute cumulative genomic positions
    chr_offsets = {}
    cumulative = 0
    for chrom in sorted(cc_df["chr_numeric"].unique()):
        chr_offsets[chrom] = cumulative
        max_pos = cc_df.loc[cc_df["chr_numeric"] == chrom, "pos"].max()
        cumulative += max_pos

    cc_df["cum_pos"] = cc_df.apply(
        lambda r: r["pos"] + chr_offsets[r["chr_numeric"]],
        axis=1
    )

    # Alternating colors
    colors = ["#4C72B0", "#55A868"]
    cc_df["color"] = cc_df["chr_numeric"].apply(lambda x: colors[int(x) % 2])

    # Plot
    plt.figure(figsize=(16, 6))
    plt.scatter(
        cc_df["cum_pos"],
        cc_df["minus_log10_p"],
        c=cc_df["color"],
        s=6,
        alpha=0.7
    )
    plt.axhline(-np.log10(5e-8), color="red", linestyle="--", linewidth=1)

    # Chromosome tick labels
    chr_labels = []
    chr_ticks = []
    for chrom in sorted(chr_offsets.keys()):
        start = chr_offsets[chrom]
        end = cc_df.loc[cc_df["chr_numeric"] == chrom, "cum_pos"].max()
        chr_ticks.append((start + end) / 2)
        chr_labels.append(str(int(chrom)))

    plt.xticks(chr_ticks, chr_labels)
    plt.xlabel("Chromosome")
    plt.ylabel("-log10(p)")
    plt.title("PD Risk GWAS Manhattan Plot (Alternating Colors)")

    # ------------------------------------------------------------
    # Save locally, then upload to GCS
    # ------------------------------------------------------------
    local_plot = "/tmp/PD_risk_manhattan_alternating.png"
    plt.savefig(local_plot, dpi=200, bbox_inches="tight")
    plt.show()

    gcs_plot = f"{bucket_base}/CC_GWAS_outputs/PD_risk_manhattan_alternating.png"
    subprocess.run(["gsutil", "cp", local_plot, gcs_plot], check=True)

    print(f"PD risk alternating-color Manhattan plot written to: {gcs_plot}")

else:
    print("Skipping PD risk alternating-color Manhattan plot. Missing p_value column.")

















# ============================================================
# RESTORE ENVIRONMENT AFTER KERNEL RESTART
# ============================================================

import hail as hl
import pandas as pd


# Restore Hail filesystem handle
fs = hl.current_backend().fs
print("Hail filesystem (fs) restored.")

# Restore bucket paths
bucket_base = "gs://gs://path_to_your_bucket"

cc_out_dir = f"{bucket_base}/CC_GWAS_outputs"
pd_out_dir = f"{bucket_base}/PD_GWAS_outputs"

print("Bucket paths restored.")
print("CC output dir:", cc_out_dir)
print("PD output dir:", pd_out_dir)




# ============================================================
# 17. VALIDATION USING PD RISK AND GENERAL COGNITION
# ============================================================
# SCIENTIFIC PURPOSE:
#
#   This step prepares the datasets needed for the two key
#   validation analyses your boss requested:
#
#   (A) VALIDATION AGAINST PD RISK
#       Question: "Do the genetic variants associated with
#       cognitive impairment *within PD cases* also show
#       association with *risk of developing PD*?"
#
#       Why this matters:
#         - If CI loci overlap with PD risk loci, it suggests
#           shared biological pathways between disease onset
#           and cognitive decline.
#         - If CI loci are independent of PD risk, it suggests
#           cognition reflects downstream or distinct biology.
#
#   (B) VALIDATION AGAINST GENERAL COGNITION (in CC dataset)
#       Question: "Do PD CI loci replicate in general cognitive
#       performance measured in the case–control dataset?"
#
#       Why this matters:
#         - CI_unique_count is a PD-specific phenotype.
#         - But cognition is measurable in both PD and non-PD.
#         - If CI loci replicate in general cognition, it shows
#           they reflect *true cognitive biology*, not PD artifacts.
#
#   Before we can run either validation, we must ensure that:
#       - PD CI summary stats and PD risk summary stats use
#         the same merge keys.
#       - Chromosome labels are harmonized.
#       - Both datasets contain chrom, pos, effect_allele, other_allele.
#
#   This block loads and harmonizes the datasets so that
#   downstream merging and effect-size comparisons work correctly.
# ============================================================

print("\n=== VALIDATION SETUP ===")

import subprocess

# -----------------------------
# Load PD risk summary stats
# -----------------------------
case_control_path = f"{bucket_base}/CC_GWAS_outputs/PD_risk_summary_stats.tsv.gz"
local_cc = "/tmp/PD_risk_summary_stats.tsv.gz"

if fs.exists(case_control_path):
    subprocess.run(["gsutil", "cp", case_control_path, local_cc], check=True)
    cc_risk = pd.read_csv(local_cc, sep="\t", compression="gzip")
    print("Loaded PD risk summary stats.")
else:
    print("PD risk summary stats not found.")
    cc_risk = None


# -----------------------------
# Load PD CI summary stats
# -----------------------------
# Correct filename based on your bucket listing:
# PD_CI_summary_stats.tsv.bgz
pd_ci_path = f"{pd_out_dir}/PD_CI_summary_stats.tsv.bgz"
local_ci = "/tmp/PD_CI_summary_stats.tsv.bgz"

if fs.exists(pd_ci_path):
    subprocess.run(["gsutil", "cp", pd_ci_path, local_ci], check=True)
    pd_ci = pd.read_csv(local_ci, sep="\t", compression="gzip")
    print("Loaded PD CI summary stats.")
else:
    print("PD CI summary stats not found.")
    pd_ci = None


# ------------------------------------------------------------
# Harmonize merge keys across datasets
# ------------------------------------------------------------

if cc_risk is not None:
    cc_risk["chrom"] = cc_risk["chrom"].astype(str).str.replace("chr", "", regex=False)
    cc_risk["pos"] = cc_risk["pos"].astype(int)

if pd_ci is not None:
    pd_ci["chrom"] = pd_ci["chrom"].astype(str).str.replace("chr", "", regex=False)
    pd_ci["pos"] = pd_ci["pos"].astype(int)

print("Validation datasets harmonized and ready.")








# ============================================================
# RESUME BLOCK — RUN THIS AFTER RESTARTING THE KERNEL
# Restores environment, paths, MatrixTables, and summary stats
# Allows you to resume directly at STEP 18
# ============================================================

import os
import hail as hl
import pandas as pd
import subprocess
from IPython.display import HTML, display
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

print("=== RESUMING ENVIRONMENT ===")

# ------------------------------------------------------------
# 1. Initialize Hail + restore filesystem
# ------------------------------------------------------------

fs = hl.current_backend().fs
print("Hail initialized and filesystem restored.")

# ------------------------------------------------------------
# 2. Restore bucket paths
# ------------------------------------------------------------
bucket_base = os.environ["WORKSPACE_BUCKET"]

cc_out_dir = f"{bucket_base}/CC_GWAS_outputs"
pd_out_dir = f"{bucket_base}/PD_GWAS_outputs"

print("Bucket paths restored:")
print("  CC:", cc_out_dir)
print("  PD:", pd_out_dir)

# ------------------------------------------------------------
# 3. Reload MatrixTables
# ------------------------------------------------------------
pd_mt = hl.read_matrix_table(f"{bucket_base}/PD_with_PCA_and_CI.mt")
cc_mt = hl.read_matrix_table(f"{bucket_base}/CC_with_PCA_and_CI.mt")

print("Loaded PD MT:", pd_mt.count_cols(), "samples")
print("Loaded CC MT:", cc_mt.count_cols(), "samples")

# ------------------------------------------------------------
# 4. Reload PD CI summary stats
# ------------------------------------------------------------
pd_ci_gcs = f"{pd_out_dir}/PD_CI_summary_stats.tsv.bgz"
pd_ci_local = "/tmp/PD_CI_summary_stats.tsv.bgz"

subprocess.run(["gsutil", "cp", pd_ci_gcs, pd_ci_local], check=True)
pd_ci = pd.read_csv(pd_ci_local, sep="\t", compression="gzip")
print("Loaded PD CI summary stats:", pd_ci.shape)

# ------------------------------------------------------------
# 5. Reload PD risk summary stats
# ------------------------------------------------------------
cc_risk_gcs = f"{cc_out_dir}/PD_risk_summary_stats.tsv.gz"
cc_risk_local = "/tmp/PD_risk_summary_stats.tsv.gz"

subprocess.run(["gsutil", "cp", cc_risk_gcs, cc_risk_local], check=True)
cc_risk = pd.read_csv(cc_risk_local, sep="\t", compression="gzip")
print("Loaded PD risk summary stats:", cc_risk.shape)

# ------------------------------------------------------------
# 6. Define CC covariates (must match PD covariates)
# ------------------------------------------------------------
n_pcs = len(cc_mt.pcs.take(1)[0])
cc_covs = (
    [cc_mt.sex_bin,
     cc_mt.age_exact] +
    [cc_mt.pcs[i] for i in range(n_pcs)]
)

print("CC covariates defined:", len(cc_covs))



pd_ci_path   = os.path.join(cc_out_dir, "PD_CI_harmonized.tsv.gz")
cc_ci_path   = os.path.join(cc_out_dir, "CC_CI_harmonized.tsv.gz")
pd_risk_path = os.path.join(cc_out_dir, "PD_risk_harmonized.tsv.gz")

with fs.open(pd_ci_path, "rb") as f:
    pd_ci = pd.read_csv(f, sep="\t", compression="gzip")

with fs.open(cc_ci_path, "rb") as f:
    cc_ci = pd.read_csv(f, sep="\t", compression="gzip")

with fs.open(pd_risk_path, "rb") as f:
    pd_risk = pd.read_csv(f, sep="\t", compression="gzip")


print(pd_ci.shape, "PD CI")
print(cc_ci.shape, "CC CI")
print(pd_risk.shape, "PD risk")


dbsnp = hl.experimental.load_dataset(
    'dbSNP_rsid',
    version='154',
    reference_genome='GRCh38'
)

# ------------------------------------------------------------
# 7. Ready to resume at STEP 18
# ------------------------------------------------------------
print("\n✔ Resume ready — continue at STEP 18")
























###############################################################
# STEP 0 — Load PD CI summary stats and extract GWS variants
#
# SCIENTIFIC QUESTION:
#   “Which variants reached genome‑wide significance in the CI GWAS,
#    and how do we obtain the top N most significant variants?”
###############################################################

pd_ci_ht = hl.import_table(
    'gs://gs://path_to_your_bucket/PD_GWAS_outputs/PD_CI_summary_stats.tsv.bgz',
    impute=True,
    force=True
)

pd_ci_ht = pd_ci_ht.annotate(
    locus = hl.locus(pd_ci_ht.chrom, pd_ci_ht.pos, reference_genome='GRCh38'),
    alleles = hl.array([pd_ci_ht.ref, pd_ci_ht.alt])
).key_by('locus', 'alleles')

# Extract all genome‑wide significant variants
gws_hits_ht = pd_ci_ht.filter(pd_ci_ht.p_value < 5e-8)
print("Total GWS variants:", gws_hits_ht.count())



###############################################################
# STEP 1 — Select the Top N Most Significant Variants
#
# SCIENTIFIC QUESTION:
#   “Which variants show the strongest statistical association with CI,
#    and how do we isolate only these variants for co‑association?”
###############################################################

N = 20  # choose your top N
topN_rows = pd_ci_ht.order_by(pd_ci_ht.p_value).take(N)

# Convert list → Hail Table
topN_ht = hl.Table.parallelize(topN_rows).key_by('locus', 'alleles')

print(f"Top {N} variants loaded.")



###############################################################
# STEP 2 — Extract These Variants from the Genotype MatrixTable
#
# SCIENTIFIC QUESTION:
#   “Which of the top N variants are present in the PD genotype data,
#    and how do we extract their genotypes for co‑association?”
###############################################################

mt_topN = pd_mt.semi_join_rows(topN_ht)

print("Variants found in genotype MT:", mt_topN.count_rows())

# Annotate entries with carrier status and n_alt
mt_topN = mt_topN.annotate_entries(
    carrier = mt_topN.GT.n_alt_alleles() > 0,
    n_alt = mt_topN.GT.n_alt_alleles()
)



###############################################################
# STEP 3 — Build Variant × Sample Carrier Matrix
#
# SCIENTIFIC QUESTION:
#   “How do we construct a clean carrier vector for each variant
#    across all PD samples, ensuring correct sample order?”
###############################################################

# Localize entries into a single table
ht = mt_topN._localize_entries('entries', 'sample_ids')

# Remove keys to avoid select() errors
ht = ht.key_by()

# Extract carrier and n_alt vectors, replacing missing values
ht = ht.annotate(
    carrier_vector = ht.entries.map(lambda e: hl.or_else(e.carrier, False)),
    n_alt_vector = ht.entries.map(lambda e: hl.or_else(e.n_alt, 0))
)



###############################################################
# STEP 4 — Add AF and p‑value for Each Variant
#
# SCIENTIFIC QUESTION:
#   “What is the allele frequency and statistical significance
#    of each of the top N variants?”
###############################################################

ht = ht.annotate(
    AF = hl.sum(ht.n_alt_vector) / (2 * hl.len(ht.n_alt_vector)),
    p_value = topN_ht[ht.locus, ht.alleles].p_value
)

df = ht.select('locus', 'alleles', 'carrier_vector', 'AF', 'p_value').to_pandas()



###############################################################
# STEP 5 — Compute Pairwise Co‑Association
#
# SCIENTIFIC QUESTION:
#   “How often do pairs of top N variants co‑occur in the same
#    individuals, and which pairs show the strongest co‑association?”
###############################################################

# Build NumPy carrier matrix
carrier_matrix = np.stack(df['carrier_vector'].values).astype(np.uint8)

# Shared carriers = dot product
shared_matrix = carrier_matrix @ carrier_matrix.T

variant_info = df[['locus', 'alleles', 'AF', 'p_value']].reset_index(drop=True)

pairs = []
n = carrier_matrix.shape[0]
total_samples = carrier_matrix.shape[1]

for i in range(n):
    for j in range(i + 1, n):
        shared = int(shared_matrix[i, j])
        if shared > 0:
            pairs.append({
                'variant_i': i,
                'variant_j': j,
                'shared_carriers': shared,
                'shared_fraction': shared / total_samples,

                'locus_i': str(variant_info.loc[i, 'locus']),
                'alleles_i': variant_info.loc[i, 'alleles'],
                'AF_i': variant_info.loc[i, 'AF'],
                'p_i': variant_info.loc[i, 'p_value'],

                'locus_j': str(variant_info.loc[j, 'locus']),
                'alleles_j': variant_info.loc[j, 'alleles'],
                'AF_j': variant_info.loc[j, 'AF'],
                'p_j': variant_info.loc[j, 'p_value']
            })

pair_df = pd.DataFrame(pairs)
pair_df = pair_df.sort_values('shared_carriers', ascending=False)

print("Top co‑associated variant pairs:")

print(pair_df.head(20))


from IPython.display import HTML

html = pair_df.head(20).to_html()
print(html)






###############################################################
# STEP 6 — Hierarchically Clustered Co‑Association Heatmap
#
# SCIENTIFIC QUESTION:
#   “Do the top N variants form natural clusters of co‑occurrence,
#    and how can hierarchical clustering reveal hidden structure?”
###############################################################

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# Identify which variants appear in the co-association matrix
top_variant_indices = sorted(
    set(pair_df['variant_i']).union(set(pair_df['variant_j']))
)

# Subset the shared-carrier matrix
shared_top = shared_matrix[np.ix_(top_variant_indices, top_variant_indices)]

# Zero diagonal (self-carriers)
np.fill_diagonal(shared_top, 0)

# Convert shared counts to a distance matrix for clustering
# Higher shared carriers = more similar → distance = max - shared
max_shared = shared_top.max()
distance_matrix = max_shared - shared_top

# Convert to condensed form for scipy
condensed = squareform(distance_matrix, checks=False)

# Perform hierarchical clustering
Z = linkage(condensed, method='average')

# Get the order of leaves (variant indices)
ordered_idx = leaves_list(Z)

# Reorder the matrix
shared_reordered = shared_top[ordered_idx][:, ordered_idx]

# Reorder locus labels
locus_labels = df['locus'].astype(str).tolist()
axis_labels = [locus_labels[top_variant_indices[i]] for i in ordered_idx]

# Mask upper triangle
mask = np.triu(np.ones_like(shared_reordered, dtype=bool))

# Plot heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(
    shared_reordered,
    cmap='viridis',
    annot=False,
    xticklabels=axis_labels,
    yticklabels=axis_labels,
    mask=mask
)

plt.title(f"Hierarchically Clustered Co‑Association Heatmap for Top {N} Variants")
plt.xlabel("Variant (locus)")
plt.ylabel("Variant (locus)")

plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()












Why do shared variants have to share gene level? Is it nit enough to just report linked variants?


Generate Jaccard heatmaps

Compute gene-level burden tests










# ============================================================
# 18. CI GWAS IN CASE–CONTROL DATASET (ci_unique_count)
# ============================================================


from IPython.display import HTML, display

print("\n=== STEP 18: CC CI GWAS (ci_unique_count) ===")

ci_field = "ci_unique_count"

if ci_field not in cc_mt.col:
    print(f"Field {ci_field} not found in CC dataset. Skipping CC CI GWAS.")
else:

    # --------------------------------------------------------
    # Run GWAS
    # --------------------------------------------------------
    cc_ci_gwas = hl.linear_regression_rows(
        y = cc_mt[ci_field],
        x = cc_mt.GT.n_alt_alleles(),
        covariates = cc_covs
    ).annotate(
        phenotype = "ci_unique_count",
        model = "linear_base"
    )

    # --------------------------------------------------------
    # Add synthetic rsID in harmonized format: chrom:pos:ref:alt
    # --------------------------------------------------------
    cc_ci_gwas = cc_ci_gwas.annotate(
        rsid = (
            hl.str(cc_ci_gwas.locus.contig).replace("chr", "") + ":" +
            hl.str(cc_ci_gwas.locus.position) + ":" +
            cc_ci_gwas.alleles[0] + ":" +
            cc_ci_gwas.alleles[1]
        )
    )

    # --------------------------------------------------------
    # Export standardized summary statistics
    # --------------------------------------------------------
    cc_ci_sumstats_path = os.path.join(
        cc_out_dir, "ci_unique_count_CC_summary_stats.tsv.bgz"
    )

    cc_ci_gwas.select(
        beta = cc_ci_gwas.beta,
        standard_error = cc_ci_gwas.standard_error,
        p_value = cc_ci_gwas.p_value,
        phenotype = cc_ci_gwas.phenotype,
        model = cc_ci_gwas.model,
        chrom = cc_ci_gwas.locus.contig.replace("chr", ""),
        pos = cc_ci_gwas.locus.position,
        ref = cc_ci_gwas.alleles[0],
        alt = cc_ci_gwas.alleles[1],
        rsid = cc_ci_gwas.rsid
    ).export(cc_ci_sumstats_path)

    print(f"CC CI summary stats written to: {cc_ci_sumstats_path}")

    # --------------------------------------------------------
    # Load into pandas for display
    # --------------------------------------------------------
    with fs.open(cc_ci_sumstats_path, "rb") as f:
        cc_ci = pd.read_csv(f, sep="\t", compression="gzip")

    display(HTML("<h3>CC ci_unique_count Summary Stats (head)</h3>" +
             cc_ci.head(20).to_html(index=False)))








# ============================================================
# 19. HARMONIZE SUMMARY STATS
# ============================================================
# SCIENTIFIC QUESTION:
#   “Are PD CI, CC CI, and PD risk summary stats aligned so that
#    variant‑wise comparisons are valid?”
#
# RATIONALE:
#   - Ensures consistent chrom/pos formatting
#   - Ensures consistent allele encoding
#   - Creates a shared locus key for merging
#
# Without harmonization, double validation would be meaningless.

print("\n=== STEP 19: HARMONIZATION ===")

pd_risk_path = os.path.join(cc_out_dir, "PD_risk_summary_stats.tsv.bgz")
with fs.open(pd_risk_path, "rb") as f:
    cc_risk = pd.read_csv(f, sep="\t", compression="gzip")
print("Loaded PD risk summary stats:", cc_risk.shape)


def harmonize(df, name):
    """Standardize chromosome, position, locus, allele fields, and rsID."""
    if df is None:
        print(f"{name}: missing, skipping.")
        return None

    # Standardize chromosome format (remove 'chr')
    if "chrom" in df.columns:
        df["chrom"] = df["chrom"].astype(str).str.replace("chr", "", regex=False)

    # Ensure position is integer
    if "pos" in df.columns:
        df["pos"] = df["pos"].astype(int)

    # Create locus field (always synthetic)
    df["locus"] = "chr" + df["chrom"].astype(str) + ":" + df["pos"].astype(str)

    # Standardize allele naming across datasets
    df["effect_allele"] = df["ref"]
    df["other_allele"]  = df["alt"]

    # Create synthetic rsID for ALL datasets
    df["rsid"] = (
        df["chrom"].astype(str) + ":" +
        df["pos"].astype(str) + ":" +
        df["ref"].astype(str) + ":" +
        df["alt"].astype(str)
    )

    print(f"{name}: harmonized.")
    return df


pd_ci   = harmonize(pd_ci,   "PD CI GWAS")
cc_risk = harmonize(cc_risk, "PD risk GWAS")
cc_ci   = harmonize(cc_ci,   "CC CI GWAS")



# ------------------------------------------------------------
# OPTIONAL: Save harmonized summary stats for tomorrow's analysis
# ------------------------------------------------------------
pd_ci_path   = os.path.join(cc_out_dir, "PD_CI_harmonized.tsv.gz")
cc_ci_path   = os.path.join(cc_out_dir, "CC_CI_harmonized.tsv.gz")
pd_risk_path = os.path.join(cc_out_dir, "PD_risk_harmonized.tsv.gz")

pd_ci.to_csv(pd_ci_path, sep="\t", index=False, compression="gzip")
cc_ci.to_csv(cc_ci_path, sep="\t", index=False, compression="gzip")
cc_risk.to_csv(pd_risk_path, sep="\t", index=False, compression="gzip")

print("Harmonized files written:")
print("  PD CI   →", pd_ci_path)
print("  CC CI   →", cc_ci_path)
print("  PD risk →", pd_risk_path)



pd_ci[["chrom","pos","ref","alt","effect_allele","other_allele","rsid"]].head()
cc_ci[["chrom","pos","ref","alt","effect_allele","other_allele","rsid"]].head()
cc_risk[["chrom","pos","ref","alt","effect_allele","other_allele","rsid"]].head()




pd_ci_path   = os.path.join(cc_out_dir, "PD_CI_harmonized.tsv.gz")
cc_ci_path   = os.path.join(cc_out_dir, "CC_CI_harmonized.tsv.gz")
pd_risk_path = os.path.join(cc_out_dir, "PD_risk_harmonized.tsv.gz")

with fs.open(pd_ci_path, "rb") as f:
    pd_ci = pd.read_csv(f, sep="\t", compression="gzip")

with fs.open(cc_ci_path, "rb") as f:
    cc_ci = pd.read_csv(f, sep="\t", compression="gzip")

with fs.open(pd_risk_path, "rb") as f:
    pd_risk = pd.read_csv(f, sep="\t", compression="gzip")






# ============================================================
# 20. MERGING HARMONIZED SUMMARY STATS
# ============================================================
# SCIENTIFIC QUESTION:
#   “Which variants are shared across PD CI, CC CI, and PD risk GWAS,
#    and how large is the directly comparable genetic overlap?”

print("\n=== STEP 20: MERGING ===")

# Drop 'alleles' if present (Hail export artifact)
for df_name, df in [("pd_ci", pd_ci), ("cc_ci", cc_ci), ("pd_risk", pd_risk)]:
    if df is not None and "alleles" in df.columns:
        df.drop(columns=["alleles"], inplace=True)
        print(f"Dropped 'alleles' column from {df_name} (prevents merge conflicts).")

# ------------------------------------------------------------
# PD CI ↔ CC CI
# ------------------------------------------------------------
merged_pdci_ccci = pd_ci.merge(
    cc_ci,
    on="rsid",
    how="inner",
    suffixes=("_PD", "_CC")
)
print(f"Merged PD CI ↔ CC CI variants: {merged_pdci_ccci.shape[0]}")
display(HTML("<h3>Merged PD CI ↔ CC CI</h3>" +
             merged_pdci_ccci.head(20).to_html(index=False)))



# ------------------------------------------------------------
# PD CI ↔ PD risk
# ------------------------------------------------------------
merged_pdci_pdrisk = pd_ci.merge(
    pd_risk,
    on="rsid",
    how="inner",
    suffixes=("_CI", "_PDrisk")
)
print(f"Merged PD CI ↔ PD risk variants: {merged_pdci_pdrisk.shape[0]}")
display(HTML("<h3>Merged PD CI ↔ PD Risk</h3>" +
             merged_pdci_pdrisk.head(20).to_html(index=False)))

# ------------------------------------------------------------
# Clean overlap counts (for downstream trend analysis)
# ------------------------------------------------------------
pdci_ccci = pd_ci.merge(cc_ci, on="rsid", suffixes=("_pdci", "_ccci"))
pdci_pdrisk = pd_ci.merge(pd_risk, on="rsid", suffixes=("_pdci", "_pdrisk"))
ccci_pdrisk = cc_ci.merge(pd_risk, on="rsid", suffixes=("_ccci", "_pdrisk"))

print(len(pdci_ccci), "shared variants: PD CI ∩ CC CI")
print(len(pdci_pdrisk), "shared variants: PD CI ∩ PD risk")
print(len(ccci_pdrisk), "shared variants: CC CI ∩ PD risk")






# ============================================================
# 21. ALLELE ALIGNMENT + VALIDATION METRICS
# ============================================================
# SCIENTIFIC PURPOSE:
#
#   This step evaluates whether genetic effects observed in PD CI
#   replicate in:
#       (1) an independent CI cohort (CC CI)
#       (2) PD risk biology (PD risk GWAS)
#
#   WHY THIS MATTERS:
#     - If PD CI ↔ CC CI effect sizes correlate, CI genetics is stable
#       across cohorts → replication.
#
#     - If PD CI ↔ PD risk effect sizes correlate, CI biology may be
#       partially driven by PD susceptibility loci → shared architecture.
#
#   NEW ANALYSES INTRODUCED HERE:
#     - Allele alignment: ensures betas refer to the same effect allele.
#     - Effect-size correlation: measures genetic similarity.
#     - Directional concordance: % of SNPs with matching effect direction.
#     - P-value enrichment: tests whether one dataset shows excess signal
#       at loci discovered in the other.
#     - KS test: compares p-value distribution to uniform expectation.
#     - Volcano-style plot: visualizes PD risk significance vs CI effect.
#
#   This step is the backbone of the RESULTS section.
# ============================================================

print("\n=== STEP 21: ALLELE ALIGNMENT + VALIDATION METRICS ===")

from scipy.stats import pearsonr, ks_2samp
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# FUNCTION: align_betas
# ------------------------------------------------------------
# WHY THIS IS NECESSARY:
#   When merging two GWAS datasets, the same SNP may have:
#       - the same effect allele (OK)
#       - reversed alleles (requires flipping beta)
#       - mismatched alleles (should be excluded)
#
#   This function:
#       1. Identifies SNPs with matching or reversed alleles.
#       2. Removes SNPs with incompatible alleles.
#       3. Flips the sign of beta where alleles are reversed.
#
#   This ensures effect sizes are directly comparable.
# ------------------------------------------------------------
def align_betas(df, ea1, oa1, ea2, oa2, b2_name):
    df = df.copy()

    # SNPs where alleles match exactly
    same = (df[ea1] == df[ea2]) & (df[oa1] == df[oa2])

    # SNPs where alleles are reversed (A/G vs G/A)
    rev  = (df[ea1] == df[oa2]) & (df[oa1] == df[ea2])

    # Keep only SNPs that are comparable
    df = df[same | rev].copy()

    # Flip beta for reversed alleles
    df.loc[rev, b2_name] = -df.loc[rev, b2_name]

    return df

# ------------------------------------------------------------
# FUNCTION: compute_metrics
# ------------------------------------------------------------
# WHAT THIS DOES:
#   Computes the core replication metrics used in genetic architecture:
#
#   - Pearson correlation of effect sizes:
#         Measures linear similarity of genetic effects.
#
#   - Directional concordance:
#         Fraction of SNPs where sign(beta1) == sign(beta2).
#
#   - P-value enrichment:
#         Fraction of SNPs in dataset 2 that show evidence of association
#         at loci discovered in dataset 1.
#
#   - KS test:
#         Tests whether p-values deviate from uniform expectation.
# ------------------------------------------------------------
def compute_metrics(df, beta1, beta2, p2):
    r, p_r = pearsonr(df[beta1], df[beta2])
    concord = np.mean(np.sign(df[beta1]) == np.sign(df[beta2]))

    # Enrichment thresholds
    enrich_05 = np.mean(df[p2] < 0.05)
    enrich_1e3 = np.mean(df[p2] < 1e-3)
    enrich_gws = np.mean(df[p2] < 5e-8)

    # KS test vs uniform distribution
    ks_stat, ks_p = ks_2samp(df[p2], np.random.uniform(size=len(df)))

    return {
        "effect_size_r": r,
        "effect_size_r_p": p_r,
        "directional_concordance": concord,
        "pct_p_lt_0.05": enrich_05,
        "pct_p_lt_1e-3": enrich_1e3,
        "pct_p_lt_5e-8": enrich_gws,
        "ks_stat_vs_uniform": ks_stat,
        "ks_p_vs_uniform": ks_p
    }

metrics_cc = None
metrics_risk = None

# ============================================================
# VALIDATION #1: PD CI ↔ CC CI
# ============================================================
# SCIENTIFIC QUESTION:
#   “Do CI-associated genetic effects replicate across cohorts?”
# ============================================================
if merged_pdci_ccci is not None and len(merged_pdci_ccci) > 0:

    # 1. Align alleles so betas are comparable
    aligned_cc = align_betas(
        merged_pdci_ccci,
        ea1="effect_allele_PD",
        oa1="other_allele_PD",
        ea2="effect_allele_CC",
        oa2="other_allele_CC",
        b2_name="beta_CC"
    )

    # 1b. Remove NaNs and infs (required for Pearson correlation)
    aligned_cc = aligned_cc.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["beta_PD", "beta_CC", "p_value_CC"]
    )

    # 2. Compute replication metrics
    metrics_cc = compute_metrics(
        aligned_cc,
        "beta_PD",
        "beta_CC",
        "p_value_CC"
    )

    display(HTML("<h3>Validation #1: PD CI ↔ CC CI Metrics</h3>" +
                 pd.DataFrame([metrics_cc]).to_html(index=False)))

    # 3. Effect-size correlation plot
    plt.figure(figsize=(6,6))
    x = aligned_cc["beta_PD"]
    y = aligned_cc["beta_CC"]
    plt.scatter(x, y, alpha=0.4)
    m, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    plt.plot(xx, m*xx + b, color="black", lw=2)
    plt.xlabel("Beta (PD CI)")
    plt.ylabel("Beta (CC CI)")
    plt.title("Effect-Size Correlation: PD CI ↔ CC CI")
    plt.grid(True, alpha=0.3)
    plt.show()

    # 4. P-value enrichment plot
    plt.figure(figsize=(6,4))
    plt.hist(-np.log10(aligned_cc["p_value_CC"]), bins=40, color="steelblue")
    plt.xlabel("-log10(p) (CC CI)")
    plt.ylabel("Count")
    plt.title("P-value Enrichment: CC CI at PD CI loci")
    plt.grid(True, alpha=0.3)
    plt.show()

# ============================================================
# VALIDATION #2: PD CI ↔ PD risk
# ============================================================
# SCIENTIFIC QUESTION:
#   “Do CI-associated loci overlap with PD susceptibility loci?”
# ============================================================
if merged_pdci_pdrisk is not None and len(merged_pdci_pdrisk) > 0:

    # 1. Align alleles
    aligned_risk = align_betas(
        merged_pdci_pdrisk,
        ea1="effect_allele_CI",
        oa1="other_allele_CI",
        ea2="effect_allele_PDrisk",
        oa2="other_allele_PDrisk",
        b2_name="beta_PDrisk"
    )

    # 1b. Remove NaNs and infs (required for Pearson correlation)
    aligned_risk = aligned_risk.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["beta_CI", "beta_PDrisk", "p_value_PDrisk"]
    )

    # 2. Compute metrics
    metrics_risk = compute_metrics(
        aligned_risk,
        "beta_CI",
        "beta_PDrisk",
        "p_value_PDrisk"
    )

    display(HTML("<h3>Validation #2: PD CI ↔ PD Risk Metrics</h3>" +
                 pd.DataFrame([metrics_risk]).to_html(index=False)))

    # 3. Effect-size correlation plot
    plt.figure(figsize=(6,6))
    x = aligned_risk["beta_CI"]
    y = aligned_risk["beta_PDrisk"]
    plt.scatter(x, y, alpha=0.4, color="darkred")
    m, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    plt.plot(xx, m*xx + b, color="black", lw=2)
    plt.xlabel("Beta (PD CI)")
    plt.ylabel("Beta (PD Risk)")
    plt.title("Effect-Size Correlation: PD CI ↔ PD Risk")
    plt.grid(True, alpha=0.3)
    plt.show()

    # 4. P-value enrichment plot
    plt.figure(figsize=(6,4))
    plt.hist(-np.log10(aligned_risk["p_value_PDrisk"]), bins=40, color="darkred")
    plt.xlabel("-log10(p) (PD risk)")
    plt.ylabel("Count")
    plt.title("P-value Enrichment: PD Risk at PD CI loci")
    plt.grid(True, alpha=0.3)
    plt.show()

    # 5. Volcano-style plot: PD risk significance vs CI effect size
    plt.figure(figsize=(6,6))
    plt.scatter(
        aligned_risk["beta_CI"],
        -np.log10(aligned_risk["p_value_PDrisk"]),
        alpha=0.4,
        color="firebrick"
    )
    plt.axhline(-np.log10(5e-8), color="grey", ls="--", lw=1)
    plt.xlabel("Beta (PD CI)")
    plt.ylabel("-log10(p) (PD Risk)")
    plt.title("Volcano-style Plot: PD Risk vs PD CI Effect Size")
    plt.grid(True, alpha=0.3)
    plt.show()



# ============================================================
# 22. SHARED GENOME-WIDE SIGNIFICANT LOCI
# ============================================================
# SCIENTIFIC PURPOSE:
#   Identify whether any SNPs reach genome-wide significance
#   (p < 5e-8) in:
#       - PD CI
#       - CC CI
#       - PD risk
#
#   And determine whether any loci are shared across traits.
#
#   This step provides the formal "overlap" analysis that
#   complements the correlation/enrichment tests in Step 21.
# ============================================================

print("\n=== STEP 22: SHARED GENOME-WIDE SIGNIFICANT LOCI ===")

GWS = 5e-8  # genome-wide significance threshold

# ------------------------------------------------------------
# 1. Identify genome-wide significant SNPs in each dataset
# ------------------------------------------------------------

pdci_gws = merged_pdci_ccci[merged_pdci_ccci["p_value_PD"] < GWS]
ccci_gws = merged_pdci_ccci[merged_pdci_ccci["p_value_CC"] < GWS]
pdrisk_gws = merged_pdci_pdrisk[merged_pdci_pdrisk["p_value_PDrisk"] < GWS]

print(f"PD CI genome-wide significant SNPs: {len(pdci_gws)}")
print(f"CC CI genome-wide significant SNPs: {len(ccci_gws)}")
print(f"PD risk genome-wide significant SNPs: {len(pdrisk_gws)}")

# ------------------------------------------------------------
# 2. Shared genome-wide significant SNPs
# ------------------------------------------------------------

# PD CI ∩ CC CI
shared_ci_gws = pd.merge(
    pdci_gws,
    ccci_gws,
    on="rsid",
    suffixes=("_PD", "_CC")
)

# PD CI ∩ PD risk
shared_pdci_pdrisk_gws = pd.merge(
    pdci_gws,
    pdrisk_gws,
    on="rsid",
    suffixes=("_CI", "_PDrisk")
)

# CC CI ∩ PD risk
shared_ccci_pdrisk_gws = pd.merge(
    ccci_gws,
    pdrisk_gws,
    on="rsid",
    suffixes=("_CC", "_PDrisk")
)

print("\nShared genome-wide significant SNPs:")
print(f"PD CI ∩ CC CI: {len(shared_ci_gws)}")
print(f"PD CI ∩ PD risk: {len(shared_pdci_pdrisk_gws)}")
print(f"CC CI ∩ PD risk: {len(shared_ccci_pdrisk_gws)}")

# ------------------------------------------------------------
# 3. Display tables if any exist
# ------------------------------------------------------------

if len(shared_ci_gws) > 0:
    display(HTML("<h3>Shared GWS SNPs: PD CI ∩ CC CI</h3>"))
    display(shared_ci_gws)

if len(shared_pdci_pdrisk_gws) > 0:
    display(HTML("<h3>Shared GWS SNPs: PD CI ∩ PD Risk</h3>"))
    display(shared_pdci_pdrisk_gws)

if len(shared_ccci_pdrisk_gws) > 0:
    display(HTML("<h3>Shared GWS SNPs: CC CI ∩ PD Risk</h3>"))
    display(shared_ccci_pdrisk_gws)




# ============================================================
# 23. DEFINE INDEPENDENT GENOME-WIDE SIGNIFICANT LOCI
# ============================================================

print("\n=== STEP 23: INDEPENDENT LOCUS DEFINITION ===")

WINDOW = 500_000  # 500 kb window

# ------------------------------------------------------------
# 0. STANDARDIZE COLUMN NAMES FOR EACH TRAIT
# ------------------------------------------------------------

# PD CI
pdci_std = pdci_gws.rename(columns={
    "chrom_PD": "chrom",
    "pos_PD": "pos",
    "p_value_PD": "p"
})[["chrom", "pos", "rsid", "p"]]

# CC CI
ccci_std = ccci_gws.rename(columns={
    "chrom_PD": "chrom",
    "pos_PD": "pos",
    "p_value_CC": "p"
})[["chrom", "pos", "rsid", "p"]]

# PD risk
pdrisk_std = pdrisk_gws.rename(columns={
    "chrom_PDrisk": "chrom",
    "pos_PDrisk": "pos",
    "p_value_PDrisk": "p"
})[["chrom", "pos", "rsid", "p"]]

# ------------------------------------------------------------
# 1. WINDOW-BASED CLUMPING FUNCTION
# ------------------------------------------------------------

def define_loci(df):
    if df.empty:
        return pd.DataFrame(columns=["chrom", "lead_pos", "rsid", "p", "locus_start", "locus_end"])

    df = df.sort_values("p").reset_index(drop=True)
    loci = []

    while len(df) > 0:
        lead = df.iloc[0]
        chrom = lead["chrom"]
        pos = lead["pos"]

        locus_start = pos - WINDOW
        locus_end = pos + WINDOW

        loci.append({
            "chrom": chrom,
            "lead_pos": pos,
            "rsid": lead["rsid"],
            "p": lead["p"],
            "locus_start": locus_start,
            "locus_end": locus_end
        })

        df = df[~(
            (df["chrom"] == chrom) &
            (df["pos"].between(locus_start, locus_end))
        )].reset_index(drop=True)

    return pd.DataFrame(loci)

# ------------------------------------------------------------
# 2. APPLY TO EACH TRAIT
# ------------------------------------------------------------

pdci_loci = define_loci(pdci_std)
ccci_loci = define_loci(ccci_std)
pdrisk_loci = define_loci(pdrisk_std)

print(f"Independent PD CI loci: {len(pdci_loci)}")
print(f"Independent CC CI loci: {len(ccci_loci)}")
print(f"Independent PD risk loci: {len(pdrisk_loci)}")

display(pdci_loci)
display(ccci_loci)
display(pdrisk_loci)

# ------------------------------------------------------------
# 3. SAVE PD CI LOCI FOR AUDIT SCRIPT
# ------------------------------------------------------------

pdci_loci.to_csv(
    f"{pd_out_dir}/ci_unique_count_clumped.tsv.bgz",
    sep="\t",
    index=False
)

pdci_loci.to_csv(
    f"{pd_out_dir}/ci_unique_count_clumped.tsv",
    sep="\t",
    index=False
)

print(f"\nSaved PD CI clumped loci to: {pd_out_dir}/ci_unique_count_clumped.tsv.bgz")

# ------------------------------------------------------------
# SAVE CC CI LOCI
# ------------------------------------------------------------

ccci_loci.to_csv(
    f"{cc_out_dir}/cc_ci_clumped.tsv",
    sep="\t",
    index=False
)

print(f"Saved CC CI clumped loci to: {cc_out_dir}/cc_ci_clumped.tsv")


# ------------------------------------------------------------
# 2. APPLY TO EACH TRAIT
# ------------------------------------------------------------

pdci_loci = define_loci(pdci_std)
ccci_loci = define_loci(ccci_std)
pdrisk_loci = define_loci(pdrisk_std)

print(f"Independent PD CI loci: {len(pdci_loci)}")
print(f"Independent CC CI loci: {len(ccci_loci)}")
print(f"Independent PD risk loci: {len(pdrisk_loci)}")

display(pdci_loci)
display(ccci_loci)
display(pdrisk_loci)





# ============================================================
# 24. CROSS-TRAIT LOCUS OVERLAP
# ============================================================

print("\n=== STEP 24: CROSS-TRAIT LOCUS OVERLAP ===")

def overlap_loci(df1, df2):
    overlaps = []
    for _, a in df1.iterrows():
        for _, b in df2.iterrows():
            if a["chrom"] == b["chrom"]:
                if not (a["locus_end"] < b["locus_start"] or a["locus_start"] > b["locus_end"]):
                    overlaps.append({
                        "chrom": a["chrom"],
                        "lead1_rsid": a["rsid"],
                        "lead2_rsid": b["rsid"],
                        "lead1_pos": a["lead_pos"],
                        "lead2_pos": b["lead_pos"],
                        "lead1_p": a["p"],
                        "lead2_p": b["p"]
                    })
    return pd.DataFrame(overlaps)

pdci_ccci_overlap = overlap_loci(pdci_loci, ccci_loci)
pdci_pdrisk_overlap = overlap_loci(pdci_loci, pdrisk_loci)
ccci_pdrisk_overlap = overlap_loci(ccci_loci, pdrisk_loci)

print("PD CI ∩ CC CI loci:", len(pdci_ccci_overlap))
print("PD CI ∩ PD risk loci:", len(pdci_pdrisk_overlap))
print("CC CI ∩ PD risk loci:", len(ccci_pdrisk_overlap))

display(pdci_ccci_overlap)
display(pdci_pdrisk_overlap)
display(ccci_pdrisk_overlap)



print("Two cross‑trait loci (PD CI ∩ CC CI):")
display(pdci_ccci_overlap[[
    "chrom",
    "lead1_rsid",
    "lead1_pos",
    "lead1_p",
    "lead2_rsid",
    "lead2_pos",
    "lead2_p"
]])





# ============================================================
# 25. REGIONAL VISUALIZATION
# ============================================================

print("\n=== STEP 25: REGIONAL VISUALIZATION ===")

import matplotlib.pyplot as plt
import numpy as np

def plot_region(df, lead_chrom, lead_pos, lead_rsid, p_col="p", window=500_000):
    region = df[
        (df["chrom"] == lead_chrom) &
        (df["pos"].between(lead_pos - window, lead_pos + window))
    ].copy()

    if region.empty:
        print(f"No SNPs found in region around {lead_chrom}:{lead_pos}")
        return

    region["neglogp"] = -np.log10(region[p_col])

    plt.figure(figsize=(10, 4))
    plt.scatter(region["pos"], region["neglogp"], s=10)
    plt.axvline(lead_pos, color="red", linestyle="--", label=f"Lead SNP: {lead_rsid}")
    plt.xlabel("Genomic position")
    plt.ylabel("-log10(p)")
    plt.title(f"Regional plot around {lead_rsid} ({lead_chrom}:{lead_pos})")
    plt.legend()
    plt.show()


# Example: plot all PD CI loci
for _, row in pdci_loci.iterrows():
    plot_region(
        pdci_std,
        lead_chrom=row["chrom"],
        lead_pos=row["lead_pos"],
        lead_rsid=row["rsid"],
        p_col="p"
    )










# ============================================================
# STEP 26A–26E: POST-LOCUS ANALYSES
# ============================================================
# These steps summarize, visualize, and compare the independent
# loci identified in Steps 23–25. They require only:
#   - chrom
#   - lead_pos
#   - rsid
#   - p
#   - locus_start / locus_end
#
# No gene annotation or external resources needed.
# ============================================================


# ============================================================
# STEP 26A — Combined Locus Summary Table
# ============================================================
# Purpose:
#   Create a single table listing all loci from all traits,
#   with trait labels, rsIDs, positions, and p-values.
# ============================================================

print("\n=== STEP 26A: COMBINED LOCUS SUMMARY TABLE ===")

pdci_loci2 = pdci_loci.copy()
pdci_loci2["trait"] = "PD_CI"

ccci_loci2 = ccci_loci.copy()
ccci_loci2["trait"] = "CC_CI"

pdrisk_loci2 = pdrisk_loci.copy()
pdrisk_loci2["trait"] = "PD_risk"

combined_loci = pd.concat([pdci_loci2, ccci_loci2, pdrisk_loci2], ignore_index=True)

display(combined_loci)


# ============================================================
# STEP 26B — Lead-SNP Manhattan Plot (Locus Manhattan)
# ============================================================
# Purpose:
#   Plot only the lead SNPs (one per locus) across the genome.
#   This is a "locus Manhattan plot" summarizing the signals.
# ============================================================

print("\n=== STEP 26B: LEAD-SNP MANHATTAN PLOT ===")

import matplotlib.pyplot as plt
import numpy as np

# Sort by chrom + position
combined_loci_sorted = combined_loci.sort_values(["chrom", "lead_pos"])

# Create cumulative genomic position index
combined_loci_sorted["cum_pos"] = combined_loci_sorted["lead_pos"]

plt.figure(figsize=(14, 5))
colors = {"PD_CI": "blue", "CC_CI": "green", "PD_risk": "red"}

for trait, group in combined_loci_sorted.groupby("trait"):
    plt.scatter(
        group["cum_pos"],
        -np.log10(group["p"]),
        s=20,
        label=trait,
        color=colors[trait]
    )

plt.xlabel("Genomic position (chromosome concatenated)")
plt.ylabel("-log10(p)")
plt.title("Lead SNP Manhattan Plot (One SNP per Locus)")
plt.legend()
plt.show()


# ============================================================
# STEP 26C — Venn Diagram of Locus Overlap
# ============================================================
# Purpose:
#   Show how many loci each trait has and how many are shared.
#   Uses rsIDs of lead SNPs as locus identifiers.
# ============================================================

print("\n=== STEP 26C: LOCUS OVERLAP VENN DIAGRAM ===")

from matplotlib_venn import venn3

pdci_set = set(pdci_loci["rsid"])
ccci_set = set(ccci_loci["rsid"])
pdrisk_set = set(pdrisk_loci["rsid"])

plt.figure(figsize=(6, 6))
venn3(
    [pdci_set, ccci_set, pdrisk_set],
    set_labels=("PD CI", "CC CI", "PD risk")
)
plt.title("Overlap of Independent Loci (Lead SNPs)")
plt.show()


# ============================================================
# STEP 26D — Chromosome-Level Locus Distribution
# ============================================================
# Purpose:
#   Count how many loci each trait has per chromosome.
#   Useful for identifying chromosome-level clustering.
# ============================================================

print("\n=== STEP 26D: CHROMOSOME-LEVEL LOCUS DISTRIBUTION ===")

pdci_counts = pdci_loci["chrom"].value_counts().sort_index()
ccci_counts = ccci_loci["chrom"].value_counts().sort_index()
pdrisk_counts = pdrisk_loci["chrom"].value_counts().sort_index()

chrom_df = pd.DataFrame({
    "PD_CI": pdci_counts,
    "CC_CI": ccci_counts,
    "PD_risk": pdrisk_counts
}).fillna(0).astype(int)

display(chrom_df)

chrom_df.plot(kind="bar", figsize=(12, 5))
plt.ylabel("Number of independent loci")
plt.title("Locus Distribution by Chromosome")
plt.show()


# ============================================================
# STEP 26E — Effect Direction Comparison (PD CI vs CC CI)
# ============================================================
# Purpose:
#   For overlapping loci, compare effect directions.
#   Uses beta values from the original merged tables.
# ============================================================

print("\n=== STEP 26E: EFFECT DIRECTION COMPARISON ===")

# Merge PD CI and CC CI on rsid for overlapping loci
overlap_snps = pd.merge(
    pdci_std.merge(pdci_loci[["rsid"]], on="rsid"),
    ccci_std.merge(ccci_loci[["rsid"]], on="rsid"),
    on="rsid",
    suffixes=("_PD", "_CC")
)

if overlap_snps.empty:
    print("No overlapping SNPs with effect sizes available.")
else:
    overlap_snps["direction_match"] = np.sign(overlap_snps["p_PD"]) == np.sign(overlap_snps["p_CC"])
    display(overlap_snps[["rsid", "p_PD", "p_CC", "direction_match"]])

    # Plot effect direction comparison
    plt.figure(figsize=(6, 6))
    plt.scatter(overlap_snps["p_PD"], overlap_snps["p_CC"], s=20)
    plt.xlabel("PD CI p-values")
    plt.ylabel("CC CI p-values")
    plt.title("Effect Direction Comparison (PD CI vs CC CI)")
    plt.show()



