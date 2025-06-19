import pandas as pd

def load_spliceai_vcf(vcf_path):
    rows = []
    with open(vcf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            chrom, pos, _, ref, alt, _, _, info = parts
            spliceai_field = [x for x in info.split(";") if x.startswith("SpliceAI=")]
            if not spliceai_field:
                continue
            value = spliceai_field[0].split("=")[1]
            fields = value.split("|")
            ds_ag, ds_al, ds_dg, ds_dl = map(float, fields[2:6])
            max_ds = max(ds_ag, ds_al, ds_dg, ds_dl)
            rows.append({
                "chrom": chrom,
                "pos": int(pos),
                "ref": ref,
                "alt": alt,
                "DS_AG": ds_ag,
                "DS_AL": ds_al,
                "DS_DG": ds_dg,
                "DS_DL": ds_dl,
                "DS_MAX": max_ds
            })
    return pd.DataFrame(rows)

def compute_splice_effect_percent(df):
    if df.empty:
        return 0.0
    mean_max_ds = df["DS_MAX"].mean()
    return round(mean_max_ds * 100, 2)  # percentage

def compute_splice_score(vcf_path):
    df = load_spliceai_vcf(vcf_path)
    splice_impact_percent = compute_splice_effect_percent(df)
    print(f"🧬 Estimated Splicing Impact: {splice_impact_percent}%")

    return splice_impact_percent

if __name__=="__main__":
    vcf_path = "./outputs/spliceai_output.vcf"
    compute_splice_score(vcf_path)

