from pydantic import BaseModel, Field

class EnformerInput(BaseModel):
    wt_file: str = Field(..., description="Path to the wild-type FASTA file")
    delta_file: str = Field(..., description="Path to the exon-deleted FASTA file")
    exon_start: int = Field(..., description="Start coordinate of the deleted exon")
    exon_end: int = Field(..., description="End coordinate of the deleted exon")

class DeleteExonInput(BaseModel):
    sequence: str = Field(..., description="Raw DNA sequence or FASTA input")
    exon_start: int = Field(..., description="Start coordinate of the exon to delete")
    exon_end: int = Field(..., description="End coordinate of the exon to delete")


class ExonCoordInput(BaseModel):
    gene_name: str = Field(..., description="Gene name to fetch exon coordinates for")

class SpliceAIInput(BaseModel):
    chrom: str = Field(..., description="Chromosome name (e.g. '1', 'X')")
    exon_start: int = Field(..., description="Start coordinate of the exon")
    exon_end: int = Field(..., description="End coordinate of the exon")
