import requests
from Bio import SeqIO
import json
from pathlib import Path
import shutil

def get_cached_or_fresh_exons(gene_symbol, cache_file="/tmp/exon_cache.json"):
    cache = {}
    path = Path(cache_file)
    if path.exists():
        with open(path) as f:
            cache = json.load(f)
    if gene_symbol not in cache:
        coords = get_exon_coords(gene_symbol)
        cache[gene_symbol] = coords
        with open(path, "w") as f:
            json.dump(cache, f)
    return cache[gene_symbol]

def delete_exon_from_fasta(fasta_path, coords_to_delete, output_path):
    record = SeqIO.read(fasta_path, "fasta")
    sequence = str(record.seq)
    for start, end, *_ in sorted(coords_to_delete, reverse=True):  # delete from end to start
        sequence = sequence[:start - 1] + sequence[end:]
    record.seq = sequence
    SeqIO.write(record, output_path, "fasta")

def get_exon_coords(gene_symbol):
    server = "https://rest.ensembl.org"
    ext = f"/lookup/symbol/homo_sapiens/{gene_symbol}?expand=1"
    headers = {"Content-Type": "application/json"}

    response = requests.get(server + ext, headers=headers)
    if not response.ok:
        raise Exception(f"Failed to fetch exon data for {gene_symbol}: {response.text}")
    
    data = response.json()

    # Use canonical transcript or the first one
    transcript = data['Transcript'][0]
    coords = [(e["start"], e["end"], e["strand"], e["id"]) for e in transcript["Exon"]]
    return coords

def extract_gene_name(fasta_path):
    with open(fasta_path) as f:
        header = f.readline().strip()
    if header.startswith(">"):
        gene = header[1:].split("_")[0]  # Extract 'TP53' from '>TP53_chr17...'
        return gene
    raise ValueError("Invalid FASTA header format.")

def main(fasta_path):
    gene = extract_gene_name(fasta_path)
    coords = get_cached_or_fresh_exons(gene)
    
    print(coords)

if __name__=="__main__":
    main(fasta_path="/home/ubuntu/exon_deletion/inputs/test_tp53_2.fasta",)