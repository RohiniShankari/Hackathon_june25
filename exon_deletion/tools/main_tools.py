import os
from langchain_core.tools import tool, Tool, StructuredTool
from tools.delete_exon import run as delete_exon_run
from tools.spliceai_predictor import main as get_splice_scores
from tools.spliceai_predictor import get_coords
from tools.enformer_predictor import main as enformer_prediction
from tools.models import DeleteExonInput, SpliceAIInput, ExonCoordInput, EnformerInput

def delete_exon(sequence: str, exon_start: int, exon_end: int) -> dict:
    wt_path, del_path = delete_exon_run(
        fasta_path=sequence,  # assuming delete_exon_run supports raw strings
        exon_start=exon_start,
        exon_end=exon_end
    )
    return {"wt_path": wt_path, "del_path": del_path}

def run_spliceai_tool(chrom: str, exon_start: int, exon_end: int) -> str:
    score, output_path = get_splice_scores("chr" + chrom, exon_start, exon_end)
    return f"SpliceAI output path: {output_path}, estimated score: {score}"

def get_exon_coordinates(gene_name: str) -> str:
    coords = get_coords(gene_name)
    return str(coords)

def run_enformer_tool(wt_file: str, delta_file: str, exon_start: int, exon_end: int):
    """
        Runs Enformer on WT and delta sequences in FASTA files

        Inputs: 
            wt_file (path to WT FASTA), delta_file (path to exon-deleted FASTA), exon_start (int), exon_end (int). 
        Outputs: 
            PNG and CSV paths for delta analysis
    """

    response, save_delta_png_path, save_tracks_png_path, save_delta_expr_csv =  enformer_prediction(wt_file, delta_file, exon_start, exon_end)

    return {
        'png': [save_delta_png_path, save_tracks_png_path],
        'csv': [save_delta_expr_csv],
        'response': response
    }

def retrieve_tools():
    spliceai_tool = StructuredTool.from_function(
        func=run_spliceai_tool,
        name="run_spliceai_tool",
        description="Run SpliceAI prediction given chromosome, exon_start, and exon_end.",
        args_schema=SpliceAIInput
    )

    delete_exon_tool = StructuredTool.from_function(
        func=delete_exon,
        name="delete_exon",
        description="Deletes a region between exon_start and exon_end from a given DNA sequence (raw or FASTA).",
        args_schema=DeleteExonInput
    )

    get_exon_coordinates_tool = StructuredTool.from_function(
        func=get_exon_coordinates,
        name="get_exon_coordinates",
        description="Fetch exon coordinates for a given gene name using the Ensembl API.",
        args_schema=ExonCoordInput
    )

    enformer_tool = StructuredTool.from_function(
        func=run_enformer_tool,
        name="run_enformer_tool",
        description="Compares regulatory changes between wild-type and exon-deleted sequences using Enformer.",
        args_schema=EnformerInput,
    )

    workflow_tools = [delete_exon_tool, spliceai_tool, get_exon_coordinates_tool, enformer_tool]

    return workflow_tools