"""
Parkinson's disease GWAS: tremor severity, cognitive impairment, and PD risk
============================================================================

Author: Andrew Winters

Scientific goals
  1. Identify variants associated with tremor severity and cognitive
     impairment (CI) in Parkinson's disease (PD).
  2. Distinguish motor-specific, CI-specific and shared genetic effects.
  3. Summarize model behavior (genomic inflation, hit counts, overlaps).
  4. Compare PD CI signals with PD risk in a large case-control dataset.
  5. Compare PD CI signals with CI in the case-control dataset.
  6. Prepare summary statistics for PRS and LD score regression (LDSC).

Environment
  Written to run top to bottom in a cloud Jupyter notebook with Hail
  already initialized and the WORKSPACE_BUCKET environment variable set.
  Tables are shown with display() and figures inline; selected tables are
  also printed as raw HTML for reports.

Inputs (in the workspace bucket)
  PD_with_PCA_and_CI.mt   PD cohort MatrixTable with PCs and CI phenotypes
  CC_with_PCA_and_CI.mt   case-control (CC) MatrixTable with PCs and CI
                          phenotypes. It contains the PD cohort individuals,
                          so PD cohort and CC results are not independent.

Conventions
  - Betas are per copy of the ALT allele (effect_allele = ALT).
  - variant_id = chrom:pos:ref:alt (used to merge datasets).
  - rsid = dbSNP 154 rsID where available.

Sections
   1. Load the PD cohort MatrixTable
   2. Covariates
   3. PD cohort GWAS (7 models)
   4. Model-level statistics
   5. Manhattan and QQ plots (PD cohort models)
   6. Suggestive hits across models
   7. Effect-size comparison across base models
   8. Slim TSVs for plotting
   9. CC CI GWAS restricted to the PD individuals plus controls
  10. CC CI GWAS in the full case-control dataset
  11. Unified summary of all GWAS models
  12. PD risk GWAS in the case-control dataset
  13. Clean summary statistics exports
  14. PD risk Manhattan and QQ plots
  15. Summary tables (raw HTML)
  16. Phenotype distribution figure
  17. Harmonization
  18. Merging harmonized summary statistics
  19. Allele alignment and cross-trait comparison metrics
  20. Genome-wide significant variants and their overlap
  21. Independent locus definition
  22. Cross-trait locus overlap
  23. Model audit: CI unique count base model
  24. Regional plots
  25. Post-locus analyses
  26. Co-association of the top CI variants
"""

import os
import subprocess

import gcsfs
import hail as hl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display
from matplotlib_venn import venn3
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import chi2, kstest, pearsonr


# ============================================================
# 0. CONFIGURATION AND HELPERS
# ============================================================

BUCKET = os.environ["WORKSPACE_BUCKET"]

PD_MT_PATH = f"{BUCKET}/PD_with_PCA_and_CI.mt"
CC_MT_PATH = f"{BUCKET}/CC_with_PCA_and_CI.mt"

PD_OUT_DIR = f"{BUCKET}/PD_GWAS_outputs"
CC_OUT_DIR = f"{BUCKET}/CC_GWAS_outputs"
CC_CI_MODELS_DIR = f"{CC_OUT_DIR}/CC_CI_models"

REFERENCE_GENOME = "GRCh38"
GWS_P = 5e-8            # genome-wide significance
SUGGESTIVE_P = 1e-6     # suggestive significance
LOCUS_WINDOW = 500_000  # +/- window for independent loci and regional plots
N_PCS = 5               # principal components used as covariates
TOP_N_COASSOC = 20      # variants used in the co-association analysis

LAMBDA_EXPECTED_MEDIAN = chi2.ppf(0.5, df=1)  # 0.4549

fs = gcsfs.GCSFileSystem(token="cloud")


def read_tsv_gz(path):
    """Read a (b)gzipped TSV from the bucket into pandas."""
    with fs.open(path, "rb") as f:
        return pd.read_csv(f, sep="\t", compression="gzip")


def write_tsv(df, path, compression=None):
    """Write a pandas DataFrame to the bucket as TSV."""
    mode = "wb" if compression else "w"
    with fs.open(path, mode) as f:
        df.to_csv(f, sep="\t", index=False, compression=compression)


def print_html(df, title):
    """Print a table as raw HTML (for copying into reports)."""
    print(f"\n=== {title} ===")
    print(df.to_html(index=False))
    print("=== END ===\n")


def chrom_to_numeric(series):
    """'chr1' / '1' -> 1. Non-autosomal labels (X, Y, MT) become NaN."""
    return pd.to_numeric(series.astype(str).str.replace("chr", "", regex=False),
                         errors="coerce")


def lambda_gc(p_values):
    """Genomic inflation: median chi-square (1 df) from p-values / 0.4549."""
    p = pd.Series(p_values).dropna()
    return np.median(chi2.isf(p, df=1)) / LAMBDA_EXPECTED_MEDIAN


def summarize_gwas(df, label, phenotype):
    """One summary row: lambda GC, hit counts and number of variants."""
    return {
        "phenotype": phenotype,
        "model": label,
        "lambda_gc": lambda_gc(df["p_value"]),
        "n_hits_5e-8": int((df["p_value"] < GWS_P).sum()),
        "n_hits_1e-6": int((df["p_value"] < SUGGESTIVE_P).sum()),
        "n_variants": len(df),
    }


def load_dbsnp():
    return hl.experimental.load_dataset(
        "dbSNP_rsid", version="154", reference_genome=REFERENCE_GENOME
    )


def finish_plot(gcs_png=None):
    """Show the current figure; if a bucket path is given, save it there too."""
    plt.tight_layout()
    if gcs_png:
        local_png = os.path.join("/tmp", os.path.basename(gcs_png))
        plt.savefig(local_png, dpi=200, bbox_inches="tight")
        subprocess.run(["gsutil", "cp", local_png, gcs_png], check=True)
        print(f"Plot written to: {gcs_png}")
    plt.show()


def manhattan_plot(df, title, colors=("#4C72B0", "#DD8452"), gcs_png=None):
    """Alternating-color Manhattan plot of autosomes (needs chrom, pos, p_value)."""
    df = df.dropna(subset=["p_value"]).copy()
    df["chr_num"] = chrom_to_numeric(df["chrom"])
    df = df.dropna(subset=["chr_num"])
    df["chr_num"] = df["chr_num"].astype(int)
    df = df.sort_values(["chr_num", "pos"])
    df["minuslog10p"] = -np.log10(df["p_value"])

    chroms = sorted(df["chr_num"].unique())
    offsets, offset = {}, 0
    for c in chroms:
        offsets[c] = offset
        offset += df.loc[df["chr_num"] == c, "pos"].max()
    df["pos_cum"] = df["pos"] + df["chr_num"].map(offsets)

    plt.figure(figsize=(14, 5))
    for i, c in enumerate(chroms):
        subset = df[df["chr_num"] == c]
        plt.scatter(subset["pos_cum"], subset["minuslog10p"], s=4, color=colors[i % 2])

    plt.axhline(-np.log10(GWS_P), color="red", linestyle="--")
    ticks = [df.loc[df["chr_num"] == c, "pos_cum"].median() for c in chroms]
    plt.xticks(ticks, [str(c) for c in chroms])
    plt.xlabel("Chromosome")
    plt.ylabel("-log10(p)")
    plt.title(title)
    finish_plot(gcs_png)


def qq_plot(df, title, gcs_png=None):
    """QQ plot of p-values with lambda GC printed on the plot."""
    p = df["p_value"].dropna()
    p = np.sort(p[p > 0].values)
    n = len(p)

    expected = -np.log10(np.arange(1, n + 1) / (n + 1))
    observed = -np.log10(p)

    plt.figure(figsize=(6, 6))
    plt.scatter(expected, observed, s=4)
    plt.plot([0, expected.max()], [0, expected.max()], color="red")
    plt.xlabel("Expected -log10(p)")
    plt.ylabel("Observed -log10(p)")
    plt.title(title)
    plt.text(0.05, 0.95, f"λGC = {lambda_gc(p):.3f}",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment="top")
    finish_plot(gcs_png)


def export_clean_sumstats(ht, out_path, dbsnp, extra_fields=()):
    """Export standardized summary statistics with rsIDs, alleles and AF/MAF."""
    ht = ht.annotate(rsid=dbsnp[ht.locus, ht.alleles].rsid).key_by()

    fields = {
        "chrom": ht.locus.contig,
        "pos": ht.locus.position,
        "ref": ht.alleles[0],
        "alt": ht.alleles[1],
        "beta": ht.beta,
        "standard_error": ht.standard_error,
        "p_value": ht.p_value,
        "phenotype": ht.phenotype,
        "model": ht.model,
        "effect_allele": ht.alleles[1],
        "other_allele": ht.alleles[0],
        "rsid": ht.rsid,
    }
    if "AF" in ht.row:
        fields["AF"] = ht.AF
        fields["MAF"] = hl.min(ht.AF, 1 - ht.AF)
    for field in extra_fields:
        if field in ht.row:
            fields[field] = ht[field]

    ht.select(**fields).export(out_path)
    print(f"Summary stats written to: {out_path}")


# ============================================================
# 1. LOAD THE PD COHORT MATRIXTABLE
# ============================================================
# Adds per-variant metadata (chrom, pos, alleles, AF) that is carried into
# the GWAS output.

mt = hl.read_matrix_table(PD_MT_PATH)

if "pc0" not in mt.col and "pcs" in mt.col:
    mt = mt.annotate_cols(**{f"pc{i}": mt.pcs[i] for i in range(N_PCS)})

mt = hl.variant_qc(mt)
mt = mt.annotate_rows(
    chrom=mt.locus.contig,
    pos=mt.locus.position,
    ref=mt.alleles[0],
    alt=mt.alleles[1],
    AF=mt.variant_qc.AF[1],
)

n_samples = mt.count_cols()


# ============================================================
# 2. COVARIATES
# ============================================================
# Adjust for demographic confounders and population structure.
# Phenotypes:
#   tremor_score     motor severity
#   ci_unique_count  CI severity (number of distinct CI diagnoses)
#   ci_binary        presence/absence of any CI diagnosis

base_covs = (
    [1.0, mt.age_exact, mt.sex_bin, mt.BMI_mean]  # 1.0 = intercept
    + [mt[f"pc{i}"] for i in range(N_PCS)]
)


# ============================================================
# 3. PD COHORT GWAS (7 MODELS)
# ============================================================

# (phenotype, model label, additional covariate)
PD_MODELS = [
    ("tremor_score",    "base",         None),
    ("ci_unique_count", "base",         None),
    ("ci_binary",       "base",         None),
    ("tremor_score",    "adj_CIunique", "ci_unique_count"),
    ("tremor_score",    "adj_CIbinary", "ci_binary"),
    ("ci_unique_count", "adj_tremor",   "tremor_score"),
    ("ci_binary",       "adj_tremor",   "tremor_score"),
]


def pd_model_path(phenotype, label, ext="tsv.bgz"):
    return f"{PD_OUT_DIR}/{phenotype}_{label}.{ext}"


def run_gwas(phenotype, label, extra_cov=None):
    print(f"\nRunning GWAS: {phenotype} | model: {label}", flush=True)

    covariates = base_covs + ([mt[extra_cov]] if extra_cov else [])

    gwas = hl.linear_regression_rows(
        y=mt[phenotype],
        x=mt.GT.n_alt_alleles(),
        covariates=covariates,
        pass_through=["rsid", "chrom", "pos", "ref", "alt", "AF"],
    ).annotate(
        phenotype=phenotype,
        model=label,
        n=n_samples,
    )

    gwas = gwas.checkpoint(pd_model_path(phenotype, label, "ht"), overwrite=True)
    gwas.export(pd_model_path(phenotype, label))
    print(f"✔ Saved: {pd_model_path(phenotype, label)}", flush=True)


for phenotype, label, extra_cov in PD_MODELS:
    run_gwas(phenotype, label, extra_cov)


# ============================================================
# 4. MODEL-LEVEL STATISTICS
# ============================================================
# Genomic inflation (lambda GC) and number of genome-wide and suggestive
# hits per model, to assess quality and detect potential confounding.

stats_df = pd.DataFrame([
    summarize_gwas(read_tsv_gz(pd_model_path(phenotype, label)), label, phenotype)
    for phenotype, label, _ in PD_MODELS
])

with fs.open(f"{PD_OUT_DIR}/model_stats.html", "w") as f:
    f.write(stats_df.to_html(index=False))
print_html(stats_df, "MODEL-LEVEL STATS")


# ============================================================
# 5. MANHATTAN AND QQ PLOTS (PD COHORT MODELS)
# ============================================================

for phenotype, label, _ in PD_MODELS:
    print(f"\n=== {phenotype} ({label}) ===")
    df = read_tsv_gz(pd_model_path(phenotype, label))
    manhattan_plot(df, f"Manhattan: {phenotype} ({label})")
    qq_plot(df, f"QQ Plot: {phenotype} ({label})")


# ============================================================
# 6. SUGGESTIVE HITS ACROSS MODELS (p < 1e-6)
# ============================================================

hit_cols = [
    "phenotype", "model",
    "rsid", "chrom", "pos", "ref", "alt",
    "beta", "standard_error", "p_value",
    "AF", "n",
]

hits = pd.concat(
    [read_tsv_gz(pd_model_path(phenotype, label)) for phenotype, label, _ in PD_MODELS],
    ignore_index=True,
)
hits = hits.loc[hits["p_value"] < SUGGESTIVE_P, [c for c in hit_cols if c in hits.columns]]

if hits.empty:
    print("\nNo variants at p < 1e-6 in any model.")
else:
    with fs.open(f"{PD_OUT_DIR}/suggestive_hits_1e6.html", "w") as f:
        f.write(hits.to_html(index=False))
    print_html(hits, "SUGGESTIVE HITS (p < 1e-6)")


# ============================================================
# 7. EFFECT-SIZE COMPARISON ACROSS BASE MODELS
# ============================================================
# Variants with p < 1e-6 in any base model, with beta and p from all three.

base_results = []
for phenotype, short in [("tremor_score", "tremor"),
                         ("ci_unique_count", "ciuniq"),
                         ("ci_binary", "cibin")]:
    df = read_tsv_gz(pd_model_path(phenotype, "base"))
    base_results.append(
        df[["locus", "alleles", "beta", "p_value"]]
        .rename(columns={"beta": f"beta_{short}", "p_value": f"p_value_{short}"})
    )

merged = base_results[0]
for df in base_results[1:]:
    merged = merged.merge(df, on=["locus", "alleles"], how="outer")

merged_sig = merged[
    (merged["p_value_tremor"] < SUGGESTIVE_P) |
    (merged["p_value_ciuniq"] < SUGGESTIVE_P) |
    (merged["p_value_cibin"] < SUGGESTIVE_P)
]

if merged_sig.empty:
    print("\nNo variants meet p < 1e-6 in any base model.\n")
else:
    print_html(merged_sig, "EFFECT-SIZE COMPARISON ACROSS BASE MODELS (p < 1e-6)")


# ============================================================
# 8. SLIM TSVs FOR PLOTTING (chrom, pos, p_value, -log10p)
# ============================================================

for phenotype, label, _ in PD_MODELS:
    df = read_tsv_gz(pd_model_path(phenotype, label))

    slim = df[["locus", "chrom", "pos", "p_value"]].copy()
    slim["chrom"] = slim["chrom"].astype(str).str.replace("chr", "", regex=False)
    slim["minuslog10p"] = -np.log10(slim["p_value"])

    slim_path = f"{PD_OUT_DIR}/{phenotype}_{label}_for_plots.tsv"
    write_tsv(slim, slim_path)
    print(f"✔ Slim TSV for plotting written to: {slim_path}")


# ============================================================
# 9. CC CI GWAS RESTRICTED TO THE PD INDIVIDUALS PLUS CONTROLS
# ============================================================
# Restricts the case-control (CC) dataset to the PD individuals in the PD
# cohort plus all CC controls, so the CC models use the same PD people as
# the PD cohort models despite independent QC pipelines.
#   CC_noPD:  CI GWAS in controls only (CI genetics without PD)
#   CC_adjPD: CI GWAS in PD + controls, adjusted for PD status

print("=== Restricting CC to the PD cohort individuals plus controls ===")

pd_mt = hl.read_matrix_table(PD_MT_PATH)
cc_mt = hl.read_matrix_table(CC_MT_PATH)

pd_ids = pd_mt.s.collect()
pd_ids_set = hl.literal(set(pd_ids))

cc_pd_subset = cc_mt.filter_cols(pd_ids_set.contains(cc_mt.s))
cc_controls = cc_mt.filter_cols((cc_mt.pheno_bin == 0) & ~pd_ids_set.contains(cc_mt.s))
cc_restricted = cc_pd_subset.union_cols(cc_controls)

print(f"PD cohort samples:             {len(pd_ids)}")
print(f"PD individuals found in CC:    {cc_pd_subset.count_cols()}")
print(f"CC controls:                   {cc_controls.count_cols()}")
print(f"Final restricted CC samples:   {cc_restricted.count_cols()}")


def prepare_cc(mt_in):
    """Variant QC + numeric BMI (BMI_mean is stored as a string in CC)."""
    mt_out = hl.variant_qc(mt_in)
    return mt_out.annotate_cols(
        BMI_num=hl.if_else(
            (mt_out.BMI_mean == "") | (mt_out.BMI_mean == "NA"),
            hl.missing(hl.tfloat64),
            hl.float64(mt_out.BMI_mean),
        )
    )


def cc_covariates(mt_in, adjust_pd=False):
    covs = [1.0, mt_in.age_exact, mt_in.sex_bin, mt_in.BMI_num]
    if adjust_pd:
        covs.append(mt_in.pheno_bin)  # PD status (0 = Non_PD, 1 = PD)
    return covs + [mt_in.pcs[i] for i in range(N_PCS)]


def extract_qc_table(mt_in):
    ht = mt_in.select_cols().rows()
    return ht.select(
        AF=ht.variant_qc.AF[1],
        MAF=hl.min(ht.variant_qc.AF[1], 1 - ht.variant_qc.AF[1]),
        rsid=ht.rsid,
    )


def add_allele_fields(ht):
    return ht.annotate(
        chrom=ht.locus.contig,
        pos=ht.locus.position,
        ref=ht.alleles[0],
        alt=ht.alleles[1],
        effect_allele=ht.alleles[1],
        other_allele=ht.alleles[0],
    )


cc_export_fields = [
    "chrom", "pos", "ref", "alt",
    "beta", "standard_error", "p_value",
    "phenotype", "model",
    "effect_allele", "other_allele",
    "AF", "MAF", "rsid",
]

cc_ci_model_paths = {}
for model_label, subset_mt, adjust_pd in [
    ("CC_noPD", cc_restricted.filter_cols(cc_restricted.pheno_bin == 0), False),
    ("CC_adjPD", cc_restricted, True),
]:
    print(f"\n=== CC CI GWAS: {model_label} ===")
    cc_model_mt = prepare_cc(subset_mt)
    print("N samples:", cc_model_mt.count_cols())

    gwas = hl.linear_regression_rows(
        y=cc_model_mt.ci_unique_count,
        x=cc_model_mt.GT.n_alt_alleles(),
        covariates=cc_covariates(cc_model_mt, adjust_pd=adjust_pd),
    ).annotate(phenotype="ci_unique_count", model=model_label)

    qc = extract_qc_table(cc_model_mt)
    gwas = add_allele_fields(gwas.annotate(**qc[gwas.key]))

    out_path = f"{CC_CI_MODELS_DIR}/CC_CI_{model_label.split('_')[1]}.tsv.bgz"
    gwas.select(*cc_export_fields).export(out_path)
    cc_ci_model_paths[model_label] = out_path
    print(f"Exported {model_label}: {out_path}")


# ============================================================
# 10. CC CI GWAS IN THE FULL CASE-CONTROL DATASET
# ============================================================
# ci_unique_count in all CC samples (PD cases and controls), adjusted for
# sex, age and all available PCs. This is the CC CI model used in the
# summary tables and the cross-trait comparisons (sections 15-25).

print("\n=== CC CI GWAS (ci_unique_count, full case-control dataset) ===")

n_cc_pcs = len(cc_mt.pcs.take(1)[0])
cc_full_covs = [1.0, cc_mt.sex_bin, cc_mt.age_exact] + [cc_mt.pcs[i] for i in range(n_cc_pcs)]

cc_ci_full_path = f"{CC_OUT_DIR}/ci_unique_count_CC_summary_stats.tsv.bgz"

cc_ci_gwas = hl.linear_regression_rows(
    y=cc_mt.ci_unique_count,
    x=cc_mt.GT.n_alt_alleles(),
    covariates=cc_full_covs,
).annotate(phenotype="ci_unique_count", model="CC_full")

cc_ci_gwas.select(
    beta=cc_ci_gwas.beta,
    standard_error=cc_ci_gwas.standard_error,
    p_value=cc_ci_gwas.p_value,
    phenotype=cc_ci_gwas.phenotype,
    model=cc_ci_gwas.model,
    chrom=cc_ci_gwas.locus.contig.replace("chr", ""),
    pos=cc_ci_gwas.locus.position,
    ref=cc_ci_gwas.alleles[0],
    alt=cc_ci_gwas.alleles[1],
).export(cc_ci_full_path)
print(f"CC CI summary stats written to: {cc_ci_full_path}")

cc_ci = read_tsv_gz(cc_ci_full_path)
display(HTML("<h3>CC ci_unique_count summary stats (head)</h3>"
             + cc_ci.head(20).to_html(index=False)))


# ============================================================
# 11. UNIFIED SUMMARY OF ALL GWAS MODELS
# ============================================================
# One table with lambda GC, hit counts, n and file path for every model.

all_model_paths = (
    [pd_model_path(p, l) for p, l, _ in PD_MODELS]
    + list(cc_ci_model_paths.values())
    + [cc_ci_full_path]
)

summary_rows = []
for path in all_model_paths:
    df = read_tsv_gz(path)
    row = summarize_gwas(df, df["model"].iloc[0], df["phenotype"].iloc[0])
    row["n_samples"] = df["n"].iloc[0] if "n" in df.columns else None
    row["path"] = path
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).sort_values(["phenotype", "model"])
summary_path = f"{PD_OUT_DIR}/gwas_summary_all_models.tsv"
write_tsv(summary_df, summary_path)

print("\n=== Unified GWAS summary ===")
print(summary_df)
print("Saved summary to:", summary_path)


# ============================================================
# 12. PD RISK GWAS IN THE CASE-CONTROL DATASET
# ============================================================

print("\nRunning PD risk GWAS in the case-control dataset...")

cc_mt_qc = hl.variant_qc(cc_mt)
cc_mt_qc = cc_mt_qc.annotate_rows(
    AF=cc_mt_qc.variant_qc.AF[1],
    MAF=hl.min(cc_mt_qc.variant_qc.AF[1], 1 - cc_mt_qc.variant_qc.AF[1]),
)

pd_risk_gwas = hl.logistic_regression_rows(
    test="wald",
    y=cc_mt_qc.pheno_bin,
    x=cc_mt_qc.GT.n_alt_alleles(),
    covariates=(
        [1.0, cc_mt_qc.age_exact, cc_mt_qc.sex_bin]
        + [cc_mt_qc.pcs[i] for i in range(N_PCS)]
    ),
    pass_through=["AF", "MAF"],
).annotate(phenotype="PD_risk", model="logistic_base")

case_control_counts = cc_mt_qc.aggregate_cols(hl.struct(
    n_cases=hl.agg.sum(cc_mt_qc.pheno_bin),
    n_controls=hl.agg.sum(1 - cc_mt_qc.pheno_bin),
))
pd_risk_gwas = pd_risk_gwas.annotate(
    n_cases=case_control_counts.n_cases,
    n_controls=case_control_counts.n_controls,
)

dbsnp = load_dbsnp()
pd_risk_gwas = pd_risk_gwas.annotate(rsid=dbsnp[pd_risk_gwas.locus, pd_risk_gwas.alleles].rsid)

pd_risk_ht_path = f"{CC_OUT_DIR}/PD_risk_logistic_base.ht"
pd_risk_gwas = pd_risk_gwas.checkpoint(pd_risk_ht_path, overwrite=True)
pd_risk_gwas.export(f"{CC_OUT_DIR}/PD_risk_logistic_base.tsv.bgz")


# ============================================================
# 13. CLEAN SUMMARY STATISTICS EXPORTS
# ============================================================

print("\nExporting clean summary stats...")

pd_ci_sumstats_path = f"{PD_OUT_DIR}/PD_CI_summary_stats.tsv.bgz"
pd_cibinary_sumstats_path = f"{PD_OUT_DIR}/PD_CIbinary_summary_stats.tsv.bgz"
pd_risk_sumstats_path = f"{CC_OUT_DIR}/PD_risk_summary_stats.tsv.gz"

# PD cohort: ci_unique_count and ci_binary base models
export_clean_sumstats(
    hl.read_table(pd_model_path("ci_unique_count", "base", "ht")),
    pd_ci_sumstats_path, dbsnp,
)
export_clean_sumstats(
    hl.read_table(pd_model_path("ci_binary", "base", "ht")),
    pd_cibinary_sumstats_path, dbsnp,
)

# PD risk: export uncompressed, then bgzip locally and copy back
pd_risk_uncompressed = f"{CC_OUT_DIR}/PD_risk_summary_stats.tsv"
export_clean_sumstats(
    hl.read_table(pd_risk_ht_path), pd_risk_uncompressed, dbsnp,
    extra_fields=("n_cases", "n_controls"),
)

local_tsv = "/tmp/PD_risk_summary_stats.tsv"
subprocess.run(["gsutil", "cp", pd_risk_uncompressed, local_tsv], check=True)
subprocess.run(["bgzip", "-f", local_tsv], check=True)
subprocess.run(["gsutil", "cp", f"{local_tsv}.gz", pd_risk_sumstats_path], check=True)
print(f"Compressed PD risk summary stats written to: {pd_risk_sumstats_path}")


# ============================================================
# 14. PD RISK MANHATTAN AND QQ PLOTS
# ============================================================

pd_risk_df = read_tsv_gz(pd_risk_sumstats_path)

qq_plot(pd_risk_df, "PD Risk GWAS QQ Plot",
        gcs_png=f"{CC_OUT_DIR}/PD_risk_qq.png")

manhattan_plot(pd_risk_df, "PD Risk GWAS Manhattan Plot",
               colors=("#4C72B0", "#55A868"),
               gcs_png=f"{CC_OUT_DIR}/PD_risk_manhattan.png")


# ============================================================
# 15. SUMMARY TABLES (RAW HTML)
# ============================================================

pd_ci_df = read_tsv_gz(pd_ci_sumstats_path)

# Table 1: CI model summary, PD cohort and case-control dataset
table1 = pd.DataFrame(
    [summarize_gwas(pd_ci_df, "PD_cohort", "ci_unique_count")]
    + [summarize_gwas(read_tsv_gz(path), label, "ci_unique_count")
       for label, path in cc_ci_model_paths.items()]
    + [summarize_gwas(cc_ci, "CC_full", "ci_unique_count")]
)
print_html(table1, "TABLE 1: CI MODEL SUMMARY (PD COHORT AND CASE-CONTROL)")

# Table 2: PD cohort CI hits (p < 1e-6)
pd.set_option("display.float_format", lambda x: f"{x:.2e}")
print_html(pd_ci_df[pd_ci_df["p_value"] < SUGGESTIVE_P],
           "TABLE 2: PD COHORT CI HITS (p < 1e-6)")

# Table 3: case-control CI hits (p < 1e-6), full CC model
print_html(cc_ci[cc_ci["p_value"] < SUGGESTIVE_P],
           "TABLE 3: CASE-CONTROL CI HITS (p < 1e-6)")


# ============================================================
# 16. PHENOTYPE DISTRIBUTION FIGURE
# ============================================================
#   A. ci_unique_count in the PD cohort vs non-PD controls
#   B. Tremor severity in the PD cohort

pd_pheno = pd_mt.cols().to_pandas()
cc_pheno = cc_mt.cols().to_pandas()
cc_pheno = cc_pheno[cc_pheno["pheno_bin"] == 0].copy()  # non-PD controls

for frame, cols in [(pd_pheno, ["ci_unique_count", "tremor_score"]),
                    (cc_pheno, ["ci_unique_count"])]:
    for col in cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")

plt.figure(figsize=(14, 6))

# Panel A: integer-aligned bins
ax1 = plt.subplot(1, 2, 1)
max_ci = int(max(pd_pheno["ci_unique_count"].max(), cc_pheno["ci_unique_count"].max()))
bins = np.arange(0, max_ci + 2) - 0.5

plt.hist(pd_pheno["ci_unique_count"].dropna(), bins=bins, alpha=0.5,
         color="royalblue", density=True, label="PD")
plt.hist(cc_pheno["ci_unique_count"].dropna(), bins=bins, alpha=0.5,
         color="seagreen", density=True, label="Non-PD controls")
plt.title("Distribution of CI_unique_count\nPD vs. Non-PD Controls")
plt.xlabel("CI_unique_count")
plt.ylabel("Density")
plt.xticks(range(0, max_ci + 1))
plt.legend()
ax1.text(-0.1, 1.05, "A", transform=ax1.transAxes,
         fontsize=18, fontweight="bold", va="top", ha="right")

# Panel B
ax2 = plt.subplot(1, 2, 2)
plt.hist(pd_pheno["tremor_score"].dropna(), bins=20, alpha=0.6,
         color="darkorange", density=True)
plt.title("Distribution of Tremor Severity\nPD Cohort Only")
plt.xlabel("Tremor Severity Score")
plt.ylabel("Density")
ax2.text(-0.1, 1.05, "B", transform=ax2.transAxes,
         fontsize=18, fontweight="bold", va="top", ha="right")

finish_plot()


# ============================================================
# 17. HARMONIZATION
# ============================================================
# Consistent chromosome/position format, allele encoding and a shared
# variant_id (chrom:pos:ref:alt) so PD CI, CC CI and PD risk can be merged.

print("\n=== HARMONIZATION ===")


def harmonize(df, name):
    df = df.copy()
    df["chrom"] = df["chrom"].astype(str).str.replace("chr", "", regex=False)
    df["pos"] = df["pos"].astype(int)
    df["locus"] = "chr" + df["chrom"] + ":" + df["pos"].astype(str)
    df["effect_allele"] = df["alt"]
    df["other_allele"] = df["ref"]
    df["variant_id"] = (df["chrom"] + ":" + df["pos"].astype(str) + ":"
                        + df["ref"].astype(str) + ":" + df["alt"].astype(str))
    if "rsid" not in df.columns:
        df["rsid"] = pd.NA
    df = df.drop(columns=["alleles"], errors="ignore")
    print(f"{name}: harmonized ({len(df)} variants).")
    return df


pd_ci = harmonize(pd_ci_df, "PD cohort CI")
cc_ci = harmonize(cc_ci, "Case-control CI")
pd_risk = harmonize(pd_risk_df, "PD risk")

for df, name in [(pd_ci, "PD_CI"), (cc_ci, "CC_CI"), (pd_risk, "PD_risk")]:
    path = f"{CC_OUT_DIR}/{name}_harmonized.tsv.gz"
    write_tsv(df, path, compression="gzip")
    print(f"  {name} → {path}")


# ============================================================
# 18. MERGING HARMONIZED SUMMARY STATISTICS
# ============================================================
# Which variants were tested in more than one GWAS, and how large is the
# directly comparable overlap?

print("\n=== MERGING ===")

merged_pdci_ccci = pd_ci.merge(cc_ci, on="variant_id", suffixes=("_PD", "_CC"))
merged_pdci_pdrisk = pd_ci.merge(pd_risk, on="variant_id", suffixes=("_CI", "_PDrisk"))
merged_ccci_pdrisk = cc_ci.merge(pd_risk, on="variant_id", suffixes=("_CC", "_PDrisk"))

print(f"Variants tested in PD CI and CC CI:   {len(merged_pdci_ccci)}")
print(f"Variants tested in PD CI and PD risk: {len(merged_pdci_pdrisk)}")
print(f"Variants tested in CC CI and PD risk: {len(merged_ccci_pdrisk)}")

display(HTML("<h3>Merged PD CI ↔ CC CI</h3>" + merged_pdci_ccci.head(20).to_html(index=False)))
display(HTML("<h3>Merged PD CI ↔ PD Risk</h3>" + merged_pdci_pdrisk.head(20).to_html(index=False)))


# ============================================================
# 19. ALLELE ALIGNMENT AND CROSS-TRAIT COMPARISON METRICS
# ============================================================
# Are PD cohort CI effects mirrored in (1) CI in the case-control dataset
# and (2) PD risk? The case-control dataset includes the PD cohort
# individuals, so these are cross-trait comparisons, not independent
# replication.
#   - Allele alignment so betas refer to the same effect allele
#   - Effect-size correlation (Pearson r) across all shared variants
#   - Directional concordance (% of variants with the same effect sign)
#   - Fraction of shared variants below three p-value thresholds
#   - One-sample KS test of p-values against Uniform(0, 1)

print("\n=== ALLELE ALIGNMENT + COMPARISON METRICS ===")


def align_betas(df, ea1, oa1, ea2, oa2, b2_name):
    """Keep variants with matching or swapped alleles; flip beta where swapped.

    variant_id includes the alleles, so swaps are not expected here; this is
    kept as a safeguard for merges on other keys.
    """
    df = df.copy()
    same = (df[ea1] == df[ea2]) & (df[oa1] == df[oa2])
    swapped = (df[ea1] == df[oa2]) & (df[oa1] == df[ea2])
    df = df[same | swapped].copy()
    swapped = swapped[df.index]
    df.loc[swapped, b2_name] = -df.loc[swapped, b2_name]
    return df


def compute_metrics(df, beta1, beta2, p2):
    r, p_r = pearsonr(df[beta1], df[beta2])
    ks_stat, ks_p = kstest(df[p2], "uniform")
    return {
        "n_variants": len(df),
        "effect_size_r": r,
        "effect_size_r_p": p_r,
        "directional_concordance": np.mean(np.sign(df[beta1]) == np.sign(df[beta2])),
        "frac_p_lt_0.05": np.mean(df[p2] < 0.05),
        "frac_p_lt_1e-3": np.mean(df[p2] < 1e-3),
        "frac_p_lt_5e-8": np.mean(df[p2] < GWS_P),
        "ks_stat_vs_uniform": ks_stat,
        "ks_p_vs_uniform": ks_p,
    }


def effect_size_plot(x, y, xlabel, ylabel, title, color=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.4, color=color)
    m, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    plt.plot(xx, m * xx + b, color="black", lw=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    finish_plot()


def p_value_hist(p_values, xlabel, title, color):
    plt.figure(figsize=(6, 4))
    plt.hist(-np.log10(p_values), bins=40, color=color)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    finish_plot()


def clean_numeric(df, cols):
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)


aligned_cc = None
aligned_risk = None

# Comparison 1: PD cohort CI ↔ case-control CI
if len(merged_pdci_ccci) > 0:
    aligned_cc = align_betas(merged_pdci_ccci, "effect_allele_PD", "other_allele_PD",
                             "effect_allele_CC", "other_allele_CC", "beta_CC")
    aligned_cc = clean_numeric(aligned_cc, ["beta_PD", "beta_CC", "p_value_CC"])

    metrics_cc = compute_metrics(aligned_cc, "beta_PD", "beta_CC", "p_value_CC")
    display(HTML("<h3>PD CI ↔ CC CI metrics</h3>"
                 + pd.DataFrame([metrics_cc]).to_html(index=False)))

    effect_size_plot(aligned_cc["beta_PD"], aligned_cc["beta_CC"],
                     "Beta (PD CI)", "Beta (CC CI)",
                     "Effect-Size Correlation: PD CI ↔ CC CI")
    p_value_hist(aligned_cc["p_value_CC"], "-log10(p) (CC CI)",
                 "CC CI p-values at variants shared with PD CI", "steelblue")

# Comparison 2: PD cohort CI ↔ PD risk
if len(merged_pdci_pdrisk) > 0:
    aligned_risk = align_betas(merged_pdci_pdrisk, "effect_allele_CI", "other_allele_CI",
                               "effect_allele_PDrisk", "other_allele_PDrisk", "beta_PDrisk")
    aligned_risk = clean_numeric(aligned_risk, ["beta_CI", "beta_PDrisk", "p_value_PDrisk"])

    metrics_risk = compute_metrics(aligned_risk, "beta_CI", "beta_PDrisk", "p_value_PDrisk")
    display(HTML("<h3>PD CI ↔ PD risk metrics</h3>"
                 + pd.DataFrame([metrics_risk]).to_html(index=False)))

    effect_size_plot(aligned_risk["beta_CI"], aligned_risk["beta_PDrisk"],
                     "Beta (PD CI)", "Beta (PD risk, log-odds)",
                     "Effect-Size Correlation: PD CI ↔ PD Risk", color="darkred")
    p_value_hist(aligned_risk["p_value_PDrisk"], "-log10(p) (PD risk)",
                 "PD risk p-values at variants shared with PD CI", "darkred")

    # Volcano-style plot: PD risk significance vs CI effect size
    plt.figure(figsize=(6, 6))
    plt.scatter(aligned_risk["beta_CI"], -np.log10(aligned_risk["p_value_PDrisk"]),
                alpha=0.4, color="firebrick")
    plt.axhline(-np.log10(GWS_P), color="grey", ls="--", lw=1)
    plt.xlabel("Beta (PD CI)")
    plt.ylabel("-log10(p) (PD risk)")
    plt.title("Volcano-style Plot: PD Risk vs PD CI Effect Size")
    plt.grid(True, alpha=0.3)
    finish_plot()


# ============================================================
# 20. GENOME-WIDE SIGNIFICANT VARIANTS AND THEIR OVERLAP
# ============================================================

print("\n=== GENOME-WIDE SIGNIFICANT VARIANTS ===")

gws_cols = ["chrom", "pos", "variant_id", "rsid", "p_value"]
pdci_gws = pd_ci.loc[pd_ci["p_value"] < GWS_P, gws_cols]
ccci_gws = cc_ci.loc[cc_ci["p_value"] < GWS_P, gws_cols]
pdrisk_gws = pd_risk.loc[pd_risk["p_value"] < GWS_P, gws_cols]

print(f"PD CI genome-wide significant variants:   {len(pdci_gws)}")
print(f"CC CI genome-wide significant variants:   {len(ccci_gws)}")
print(f"PD risk genome-wide significant variants: {len(pdrisk_gws)}")

shared_gws = {
    "PD CI ∩ CC CI": pdci_gws.merge(ccci_gws, on="variant_id", suffixes=("_PD", "_CC")),
    "PD CI ∩ PD risk": pdci_gws.merge(pdrisk_gws, on="variant_id", suffixes=("_CI", "_PDrisk")),
    "CC CI ∩ PD risk": ccci_gws.merge(pdrisk_gws, on="variant_id", suffixes=("_CC", "_PDrisk")),
}

print("\nGenome-wide significant in both:")
for name, shared in shared_gws.items():
    print(f"  {name}: {len(shared)}")
    if len(shared) > 0:
        display(HTML(f"<h3>Genome-wide significant in both: {name}</h3>"))
        display(shared)


# ============================================================
# 21. INDEPENDENT LOCUS DEFINITION
# ============================================================
# Distance-based clumping (no LD): take the most significant variant,
# define a +/- 500 kb locus around it, remove variants in that window,
# repeat.

print("\n=== INDEPENDENT LOCUS DEFINITION ===")


def define_loci(df, window=LOCUS_WINDOW):
    cols = ["chrom", "lead_pos", "variant_id", "rsid", "p", "locus_start", "locus_end"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    df = df.sort_values("p_value").reset_index(drop=True)
    loci = []
    while len(df) > 0:
        lead = df.iloc[0]
        start, end = lead["pos"] - window, lead["pos"] + window
        loci.append({
            "chrom": lead["chrom"],
            "lead_pos": lead["pos"],
            "variant_id": lead["variant_id"],
            "rsid": lead["rsid"],
            "p": lead["p_value"],
            "locus_start": start,
            "locus_end": end,
        })
        df = df[~((df["chrom"] == lead["chrom"]) & df["pos"].between(start, end))]
        df = df.reset_index(drop=True)
    return pd.DataFrame(loci, columns=cols)


pdci_loci = define_loci(pdci_gws)
ccci_loci = define_loci(ccci_gws)
pdrisk_loci = define_loci(pdrisk_gws)

print(f"Independent PD CI loci:   {len(pdci_loci)}")
print(f"Independent CC CI loci:   {len(ccci_loci)}")
print(f"Independent PD risk loci: {len(pdrisk_loci)}")

display(pdci_loci)
display(ccci_loci)
display(pdrisk_loci)

pdci_clump_path = f"{PD_OUT_DIR}/ci_unique_count_clumped.tsv"
ccci_clump_path = f"{CC_OUT_DIR}/cc_ci_clumped.tsv"
pdrisk_clump_path = f"{CC_OUT_DIR}/pd_risk_clumped.tsv"
write_tsv(pdci_loci, pdci_clump_path)
write_tsv(ccci_loci, ccci_clump_path)
write_tsv(pdrisk_loci, pdrisk_clump_path)
print(f"\nSaved loci to:\n  {pdci_clump_path}\n  {ccci_clump_path}\n  {pdrisk_clump_path}")


# ============================================================
# 22. CROSS-TRAIT LOCUS OVERLAP
# ============================================================
# Two loci overlap if they are on the same chromosome and their +/- 500 kb
# windows intersect.

print("\n=== CROSS-TRAIT LOCUS OVERLAP ===")


def overlap_loci(df1, df2):
    overlaps = []
    for _, a in df1.iterrows():
        for _, b in df2.iterrows():
            if a["chrom"] == b["chrom"] and not (
                a["locus_end"] < b["locus_start"] or a["locus_start"] > b["locus_end"]
            ):
                overlaps.append({
                    "chrom": a["chrom"],
                    "lead1_variant_id": a["variant_id"],
                    "lead1_rsid": a["rsid"],
                    "lead1_pos": a["lead_pos"],
                    "lead1_p": a["p"],
                    "lead2_variant_id": b["variant_id"],
                    "lead2_rsid": b["rsid"],
                    "lead2_pos": b["lead_pos"],
                    "lead2_p": b["p"],
                })
    return pd.DataFrame(overlaps)


locus_overlaps = {
    "PD CI ∩ CC CI": overlap_loci(pdci_loci, ccci_loci),
    "PD CI ∩ PD risk": overlap_loci(pdci_loci, pdrisk_loci),
    "CC CI ∩ PD risk": overlap_loci(ccci_loci, pdrisk_loci),
}

for name, overlap in locus_overlaps.items():
    print(f"{name} overlapping loci: {len(overlap)}")
    display(overlap)


# ============================================================
# 23. MODEL AUDIT: CI UNIQUE COUNT BASE MODEL
# ============================================================
# Confirms inputs, model-level diagnostics and the number of independent
# loci for the main CI model.

print("\n=== MODEL AUDIT: ci_unique_count (base) ===")

ci = hl.read_table(pd_model_path("ci_unique_count", "base", "ht"))

print("Sample size (n):", ci.aggregate(hl.agg.take(ci.n, 1))[0])
print("Variants tested:", ci.count())

chisq_expr = (ci.beta / ci.standard_error) ** 2
audit_stats = ci.aggregate(hl.struct(
    chisq_mean=hl.agg.mean(chisq_expr),
    chisq_median=hl.agg.approx_median(chisq_expr),
    median_chisq_from_p=hl.agg.approx_median(hl.qchisqtail(ci.p_value, 1)),
))
print("λGC:", audit_stats.median_chisq_from_p / LAMBDA_EXPECTED_MEDIAN)
print("Chi-square (beta/SE)^2 mean:  ", audit_stats.chisq_mean)
print("Chi-square (beta/SE)^2 median:", audit_stats.chisq_median)

# Genome-wide significant variants before clumping (expected to exceed the
# number of independent loci)
print("Genome-wide significant variants:", ci.filter(ci.p_value < GWS_P).count())
print("Independent loci (section 21):   ", len(pdci_loci))


# ============================================================
# 24. REGIONAL PLOTS (ALL PD CI LOCI)
# ============================================================

print("\n=== REGIONAL PLOTS ===")


def plot_region(df, lead_chrom, lead_pos, lead_label, window=LOCUS_WINDOW):
    region = df[
        (df["chrom"] == lead_chrom) &
        df["pos"].between(lead_pos - window, lead_pos + window)
    ]

    if region.empty:
        print(f"No variants found in region around {lead_chrom}:{lead_pos}")
        return

    plt.figure(figsize=(10, 4))
    plt.scatter(region["pos"], -np.log10(region["p_value"]), s=10)
    plt.axvline(lead_pos, color="red", linestyle="--", label=f"Lead variant: {lead_label}")
    plt.axhline(-np.log10(GWS_P), color="grey", linestyle=":", lw=1)
    plt.xlabel(f"Position on chromosome {lead_chrom}")
    plt.ylabel("-log10(p)")
    plt.title(f"Regional plot around {lead_label} ({lead_chrom}:{lead_pos})")
    plt.legend()
    finish_plot()


for _, row in pdci_loci.iterrows():
    label = row["rsid"] if pd.notna(row["rsid"]) else row["variant_id"]
    plot_region(pd_ci, row["chrom"], row["lead_pos"], label)


# ============================================================
# 25. POST-LOCUS ANALYSES
# ============================================================

# 25A. Combined locus summary table
print("\n=== 25A: COMBINED LOCUS SUMMARY TABLE ===")

combined_loci = pd.concat([
    pdci_loci.assign(trait="PD_CI"),
    ccci_loci.assign(trait="CC_CI"),
    pdrisk_loci.assign(trait="PD_risk"),
], ignore_index=True)
display(combined_loci)

# 25B. Lead-variant Manhattan plot (one variant per locus)
print("\n=== 25B: LEAD-VARIANT MANHATTAN PLOT ===")

chrom_lengths = (
    pd_ci.assign(chr_num=chrom_to_numeric(pd_ci["chrom"]))
    .dropna(subset=["chr_num"])
    .groupby("chr_num")["pos"].max()
    .sort_index()
)
chrom_offsets = chrom_lengths.cumsum().shift(fill_value=0)

combined_loci["chr_num"] = chrom_to_numeric(combined_loci["chrom"])
combined_loci = combined_loci.dropna(subset=["chr_num"])
combined_loci["cum_pos"] = combined_loci["lead_pos"] + combined_loci["chr_num"].map(chrom_offsets)

plt.figure(figsize=(14, 5))
trait_colors = {"PD_CI": "blue", "CC_CI": "green", "PD_risk": "red"}
for trait, group in combined_loci.groupby("trait"):
    plt.scatter(group["cum_pos"], -np.log10(group["p"].astype(float)), s=20,
                label=trait, color=trait_colors[trait])

plt.xticks((chrom_offsets + chrom_lengths / 2).values,
           [str(int(c)) for c in chrom_lengths.index])
plt.xlabel("Chromosome")
plt.ylabel("-log10(p)")
plt.title("Lead-Variant Manhattan Plot (One Variant per Locus)")
plt.legend()
finish_plot()

# 25C. Venn diagram of shared lead variants (exact same lead variant)
print("\n=== 25C: LEAD-VARIANT VENN DIAGRAM ===")

plt.figure(figsize=(6, 6))
venn3(
    [set(pdci_loci["variant_id"]), set(ccci_loci["variant_id"]), set(pdrisk_loci["variant_id"])],
    set_labels=("PD CI", "CC CI", "PD risk"),
)
plt.title("Shared Lead Variants Across Traits")
finish_plot()

# 25D. Loci per chromosome
print("\n=== 25D: CHROMOSOME-LEVEL LOCUS DISTRIBUTION ===")

chrom_df = pd.DataFrame({
    "PD_CI": pdci_loci["chrom"].value_counts(),
    "CC_CI": ccci_loci["chrom"].value_counts(),
    "PD_risk": pdrisk_loci["chrom"].value_counts(),
}).fillna(0).astype(int)
chrom_df = chrom_df.loc[sorted(chrom_df.index, key=lambda c: (len(str(c)), str(c)))]
display(chrom_df)

chrom_df.plot(kind="bar", figsize=(12, 5))
plt.xlabel("Chromosome")
plt.ylabel("Number of independent loci")
plt.title("Locus Distribution by Chromosome")
finish_plot()

# 25E. Effect direction at PD CI lead variants in the case-control CI GWAS
print("\n=== 25E: EFFECT DIRECTION AT PD CI LEAD VARIANTS ===")

if aligned_cc is None or pdci_loci.empty:
    print("No PD CI lead variants with case-control effect sizes available.")
else:
    lead_effects = aligned_cc[aligned_cc["variant_id"].isin(pdci_loci["variant_id"])].copy()
    lead_effects["direction_match"] = np.sign(lead_effects["beta_PD"]) == np.sign(lead_effects["beta_CC"])
    display(lead_effects[["variant_id", "rsid_PD", "beta_PD", "beta_CC",
                          "p_value_PD", "p_value_CC", "direction_match"]])
    print(f"Same direction: {lead_effects['direction_match'].sum()} of {len(lead_effects)}")

    plt.figure(figsize=(6, 6))
    plt.scatter(lead_effects["beta_PD"], lead_effects["beta_CC"], s=20)
    plt.axhline(0, color="grey", lw=0.8)
    plt.axvline(0, color="grey", lw=0.8)
    plt.xlabel("Beta (PD CI)")
    plt.ylabel("Beta (CC CI)")
    plt.title("Effect Direction at PD CI Lead Variants")
    finish_plot()


# ============================================================
# 26. CO-ASSOCIATION OF THE TOP CI VARIANTS
# ============================================================
# Which of the top N CI variants are carried by the same PD individuals,
# and do they form clusters of co-occurrence?

print(f"\n=== CO-ASSOCIATION OF THE TOP {TOP_N_COASSOC} CI VARIANTS ===")

pd_ci_ht = hl.import_table(pd_ci_sumstats_path, impute=True, force=True)
pd_ci_ht = pd_ci_ht.annotate(
    locus=hl.locus(pd_ci_ht.chrom, pd_ci_ht.pos, reference_genome=REFERENCE_GENOME),
    alleles=hl.array([pd_ci_ht.ref, pd_ci_ht.alt]),
).key_by("locus", "alleles")

topN_ht = pd_ci_ht.order_by(pd_ci_ht.p_value).head(TOP_N_COASSOC).key_by("locus", "alleles")

# Genotypes of these variants in the PD cohort
mt_topN = pd_mt.semi_join_rows(topN_ht)
print("Variants found in genotype MT:", mt_topN.count_rows())

mt_topN = mt_topN.annotate_entries(
    carrier=mt_topN.GT.n_alt_alleles() > 0,
    n_alt=mt_topN.GT.n_alt_alleles(),
)

# Variant × sample carrier vectors (missing genotypes treated as non-carrier)
ht = mt_topN._localize_entries("entries", "sample_ids")
ht = ht.annotate(
    carrier_vector=ht.entries.map(lambda e: hl.or_else(e.carrier, False)),
    n_alt_vector=ht.entries.map(lambda e: hl.or_else(e.n_alt, 0)),
)
ht = ht.annotate(
    AF=hl.sum(ht.n_alt_vector) / (2 * hl.len(ht.n_alt_vector)),
    p_value=topN_ht[ht.locus, ht.alleles].p_value,
).key_by()

coassoc_df = ht.select("locus", "alleles", "carrier_vector", "AF", "p_value").to_pandas()

# Pairwise shared carriers
carrier_matrix = np.stack(coassoc_df["carrier_vector"].values).astype(np.uint8)
shared_matrix = carrier_matrix @ carrier_matrix.T
variant_info = coassoc_df[["locus", "alleles", "AF", "p_value"]].reset_index(drop=True)
n_variants, total_samples = carrier_matrix.shape

pairs = []
for i in range(n_variants):
    for j in range(i + 1, n_variants):
        shared = int(shared_matrix[i, j])
        if shared > 0:
            pairs.append({
                "variant_i": i,
                "variant_j": j,
                "shared_carriers": shared,
                "shared_fraction": shared / total_samples,
                "locus_i": str(variant_info.loc[i, "locus"]),
                "alleles_i": variant_info.loc[i, "alleles"],
                "AF_i": variant_info.loc[i, "AF"],
                "p_i": variant_info.loc[i, "p_value"],
                "locus_j": str(variant_info.loc[j, "locus"]),
                "alleles_j": variant_info.loc[j, "alleles"],
                "AF_j": variant_info.loc[j, "AF"],
                "p_j": variant_info.loc[j, "p_value"],
            })

if not pairs:
    print("No pairs of top variants share carriers.")
else:
    pair_df = pd.DataFrame(pairs).sort_values("shared_carriers", ascending=False)
    print_html(pair_df.head(20), "TOP CO-ASSOCIATED VARIANT PAIRS")

    # Hierarchically clustered co-association heatmap
    top_idx = sorted(set(pair_df["variant_i"]) | set(pair_df["variant_j"]))
    shared_top = shared_matrix[np.ix_(top_idx, top_idx)].astype(float)
    np.fill_diagonal(shared_top, 0)

    distance_matrix = shared_top.max() - shared_top  # more shared carriers = closer
    np.fill_diagonal(distance_matrix, 0)
    order = leaves_list(linkage(squareform(distance_matrix, checks=False), method="average"))

    shared_reordered = shared_top[order][:, order]
    locus_labels = coassoc_df["locus"].astype(str).tolist()
    axis_labels = [locus_labels[top_idx[i]] for i in order]

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        shared_reordered,
        cmap="viridis",
        xticklabels=axis_labels,
        yticklabels=axis_labels,
        mask=np.triu(np.ones_like(shared_reordered, dtype=bool)),
        cbar_kws={"label": "Shared carriers"},
    )
    plt.title(f"Hierarchically Clustered Co-Association Heatmap, Top {TOP_N_COASSOC} CI Variants")
    plt.xlabel("Variant (locus)")
    plt.ylabel("Variant (locus)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    finish_plot()
