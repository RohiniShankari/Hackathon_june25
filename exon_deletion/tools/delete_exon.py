import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from typing import Tuple

def parse_exon_coords(coord_str):
    """
    Parses a string of exon coordinates like "100-200,300-400" into a list of tuples.
    """
    coords = []
    for exon in coord_str.split(","):
        start, end = map(int, exon.strip().split("-"))
        coords.append((start, end))
    return coords

def run(fasta_path: str, exon_start: int, exon_end: int) -> Tuple[str, str]:
    """
    Deletes an exon from the input sequence and writes WT and delta FASTAs.
    Returns paths to the output files.
    """
    output_dir = "/home/ubuntu/exon_deletion/outputs/api"
    os.makedirs(output_dir, exist_ok=True)

    if fasta_path.endswith(".fasta"):
        record = SeqIO.read(fasta_path, "fasta")
        wt_seq = record.seq
        id_name = record.id
    else:
        wt_seq = fasta_path
        id_name = "temp_fasta"

    seq_length = len(wt_seq)
    if 0<=exon_start<=seq_length-1 and seq_length-1>=exon_end>exon_start:
        delta_seq = wt_seq[:exon_start] + wt_seq[exon_end:]
    else:
        raise ValueError(f"The given coordinates are out-of-index, the range to this sequence is 0-{seq_length-1}")

    # Save or print sequences
    wt_record = SeqRecord(wt_seq, id=id_name + "_WT", description="Wild-type sequence")
    delta_record = SeqRecord(delta_seq, id=id_name + f"_exon{str(exon_start)}-{str(exon_end)}", description="Exon-deleted sequence")

    # Save to output
    wt_path = os.path.join(output_dir,f"deleted_exon_WT.fasta")
    delta_path = os.path.join(output_dir,f"deleted_exon_DEL.fasta")

    SeqIO.write(wt_record, wt_path, "fasta")
    SeqIO.write(delta_record, delta_path, "fasta")

    print(f"WT sequence written to {wt_path}")
    print(f"Delta exon sequence written to {delta_path}")

    return wt_path, delta_path

if __name__ == "__main__":
    #Give fasta_path or fasta_sequence
    fasta_path = "AAGTGTCTTTGCAGCTGTGGTGGCTCAGAGCAGGTCAGAGGCTCTGCTGTCTGTGTAGTGAGTGCAGTTGCCTTGAGTGACTCAGGGAAGAGGTGTAGTGAGGAAACAGGGGAGATCAGGTGTTTTCATGTTTGTGTGTTTGTTTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTTTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTGCTGTCCTGCTGTTTGTTGCTGTGTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTCTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTT"
    exon_start = 1500
    exon_end = 1550
    output_prefix = "/home/ubuntu/exon_deletion/outputs/api"
    os.makedirs(output_prefix, exist_ok=True)

    run(fasta_path, exon_start, exon_end)
