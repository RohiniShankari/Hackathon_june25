import os
import subprocess
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import pysam
from tools.splice_score import compute_splice_score
from tools.retrieve_exons import get_cached_or_fresh_exons

def delete_exon(sequence, start, end):
    """Deletes the exon between [start, end) from the input sequence (0-based)."""
    wt_seq = sequence
    del_seq = sequence[:start] + sequence[end:]
    return str(wt_seq), str(del_seq)


def save_sequence_to_fasta(seq_str, output_path, seq_id):
    """Saves a sequence string to a FASTA file."""
    record = SeqRecord(Seq(seq_str), id=seq_id, description="")
    SeqIO.write(record, output_path, "fasta")


def generate_bed_file(chrom, start, end, bed_path):
    """Generates a BED file for the deleted exon."""
    with open(bed_path, "w") as f:
        f.write(f"{chrom}\t{start}\t{end}\tEXON_DEL\t0\t+\n")


def get_base_from_fasta(fasta_path, chrom, pos):
    fasta = pysam.FastaFile(fasta_path)
    return fasta.fetch(chrom, pos - 1, pos).upper()


def generate_snv_vcf(chrom, start, end, vcf_path, fasta_path, alt_base='A'):
    """Generates a VCF with SNVs at exon splice junctions (acceptor & donor)."""
    acc_pos = start + 1  # 1-based
    don_pos = end

    fasta = pysam.FastaFile(fasta_path)
    # print("Fasta keys:", fasta.references)
    print("Extracted fasta:", fasta.get_reference_length(chrom))

    def write_variant(fh, pos):
        if pos <= 0 or pos > fasta.get_reference_length(chrom):
            raise ValueError(f"Invalid position {pos} on {chrom}")
        ref = fasta.fetch(chrom, pos - 1, pos).upper()
        alt = alt_base if alt_base != ref else ('C' if ref != 'C' else 'T')
        fh.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\n")

    with open(vcf_path, 'w') as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write(f"##contig=<ID={chrom},length={fasta.get_reference_length(chrom)}>\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        write_variant(f, acc_pos)
        write_variant(f, don_pos)


def generate_vcf_for_exon_deletion(chrom, start, end, vcf_path, fasta_fai_path="hg38.fa.fai"):
    """Alternative VCF generator using only FASTA .fai index."""
    splice_acceptor_pos = start + 1
    splice_donor_pos = end

    with open(fasta_fai_path) as fai:
        contig_lines = [
            f"##contig=<ID={line.split()[0]},length={line.split()[1]}>\n"
            for line in fai
        ]

    with open(vcf_path, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.writelines(contig_lines)
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        f.write(f"{chrom}\t{splice_acceptor_pos}\t.\tG\tA\t.\t.\t.\n")
        f.write(f"{chrom}\t{splice_donor_pos}\t.\tG\tA\t.\t.\t.\n")


def run_spliceai(bed_or_vcf_path, ref_genome_fasta, output_path, annotation="grch38"):
    """Runs SpliceAI using subprocess. Requires 'spliceai' in PATH."""
    cmd = [
        "python", "-m", "spliceai",
        "-I", bed_or_vcf_path,
        "-O", output_path,
        "-R", ref_genome_fasta,
        "-A", annotation
    ]
    print(f"[INFO] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"[INFO] SpliceAI output saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] SpliceAI failed: {e}")
        raise


def parse_spliceai_output(output_path):
    """
    Parses SpliceAI output file and returns a list of predictions.
    """
    results = []
    with open(output_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.strip().split("\t")
            try:
                info_field = fields[7]
                spliceai_tag = [x for x in info_field.split(';') if x.startswith("SpliceAI=")][0]
                data = spliceai_tag.split("=")[1].split("|")
                result = {
                    "chrom": fields[0],
                    "pos": int(fields[1]),
                    "ref": fields[3],
                    "alt": fields[4],
                    "DS_AG": float(data[2]),
                    "DS_AL": float(data[3]),
                    "DS_DG": float(data[4]),
                    "DS_DL": float(data[5]),
                    "AG_pos": int(data[6]),
                    "AL_pos": int(data[7]),
                    "DG_pos": int(data[8]),
                    "DL_pos": int(data[9])
                }
                results.append(result)
            except Exception as e:
                print(f"[WARNING] Could not parse line: {line.strip()}")
    return results

def main(
    chrom,
    exon_start,
    exon_end,
    output_dir="./outputs/api"
):
    # output_dir = os.path.join(output_base_dir, os.path.basename(gene_name).split(".")[0])
    os.makedirs(output_dir, exist_ok=True)
    ref_genome = "./data/hg38.fa"
    # # Load gene sequence
    # wt_seq = SeqIO.read(wt_file, "fasta").seq
    # delta_seq = SeqIO.read(delta_file, "fasta").seq

    # Generate VCF input for SpliceAI
    vcf_path = os.path.join(output_dir, "spliceai_input.vcf")
    generate_snv_vcf(chrom, exon_start, exon_end, vcf_path, fasta_path=ref_genome)

    # Run SpliceAI
    output_path = os.path.join(output_dir, "spliceai_output.vcf")
    run_spliceai(vcf_path, ref_genome, output_path)

    # Parse and print SpliceAI output
    results = parse_spliceai_output(output_path)
    print("\n[RESULTS] SpliceAI Δ Scores:")
    for r in results:
        print(f"{r['chrom']}:{r['pos']} | DS_AG: {r['DS_AG']} | DS_DG: {r['DS_DG']} | DS_AL: {r['DS_AL']} | DS_DL: {r['DS_DL']}")

    estimated_splice_score = compute_splice_score(output_path)

    return estimated_splice_score, output_path

def get_coords(gene_name: str):
    coords = get_cached_or_fresh_exons(gene_name)
    
    return coords

if __name__ == "__main__":
    gene_name="TP53"
    coords = get_cached_or_fresh_exons(gene_name)
    print("Coordinates:\n", coords)
    chrom = "chr17"
    try:
        start, end = map(int, input("Enter your coordinates (start) and (end) in comma separated values:").strip().split(","))
    except:
        start = coords[0][0]
        end = coords[0][1]

    main(
        chrom=chrom,
        exon_start=start,   #7673710,
        exon_end=end,    #7673710,
        gene_name=gene_name,
        output_base_dir="/home/ubuntu/exon_deletion/outputs"
    )
