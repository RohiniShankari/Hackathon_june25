import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pysam
import gzip, os
import kipoiseq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from kipoiseq import Interval
import pyfaidx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

SEQ_LEN  = 393216
HALF_LEN = SEQ_LEN // 2
MODEL_PATH = 'https://tfhub.dev/deepmind/enformer/1'
CAGE_START = 4790
CAGE_END = 5313  # exclusive
CENTER_WINDOW = 50  # bins

class Enformer:

  def __init__(self, tfhub_url):
    self._model = hub.load(tfhub_url).model

  def predict_on_batch(self, inputs):
    predictions = self._model.predict_on_batch(inputs)
    return {k: v.numpy() for k, v in predictions.items()}

  @tf.function
  def contribution_input_grad(self, input_sequence,
                              target_mask, output_head='human'):
    input_sequence = input_sequence[tf.newaxis]

    target_mask_mass = tf.reduce_sum(target_mask)
    with tf.GradientTape() as tape:
      tape.watch(input_sequence)
      prediction = tf.reduce_sum(
          target_mask[tf.newaxis] *
          self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

    input_grad = tape.gradient(prediction, input_sequence) * input_sequence
    input_grad = tf.squeeze(input_grad, axis=0)
    return tf.reduce_sum(input_grad, axis=-1)
    
def one_hot_encode_char(c):
    # A,C,G,T→0,1,2,3; everything else→all-zero (N)
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    vec = np.zeros(4, dtype=np.float32)
    if c in mapping:
        vec[mapping[c]] = 1.0
    return vec

def plot_tracks(tracks, interval, height=1.5, png_path='/tmp/tracks.png'):
  fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
  for ax, (title, y) in zip(axes, tracks.items()):
    start = int(interval.start)
    end = int(interval.end)
    ax.fill_between(np.linspace(start, end, num=len(y)), y)
    ax.set_title(title)
    sns.despine(top=True, right=True, bottom=True)
  ax.set_xlabel(str(interval))
  plt.tight_layout()
  plt.savefig(png_path)

def plot_delta(ref_pred, mut_pred, chrom, exon_start, exon_end, tracks=[41, 5110], png_path="/tmp/delta.png"):

    n_bins = ref_pred.shape[0]
    center_bin = n_bins // 2
    pos   = (exon_start + exon_end) // 2
    print("Center-bin Δ across tracks:", mut_pred[center_bin] - ref_pred[center_bin])

    diff = mut_pred - ref_pred
    n_bins = diff.shape[0]
    coords = np.arange(n_bins) * 128 - (n_bins // 2 * 128)

    plt.figure(figsize=(12, 3))
    for t in tracks:
        plt.plot(coords, diff[:, t], label=f"Track {t}")
    plt.axvline(0, color="black", linestyle="--", label="Deletion center")
    plt.xlabel("bp relative to center")
    plt.title(f"{chrom}:{pos} Δ-expression")
    plt.legend()
    plt.show()
    plt.savefig(png_path)

def seq_to_fixed_one_hot(seq, center_index):
    """
    Place `seq` so that its `center_index` lands at the center of a SEQ_LEN window.
    seq: string (WT or Δexon)
    center_index: integer index in seq to align to window center
    Returns: [1, SEQ_LEN, 4] float32 Tensor
    """
    # Initialize all-zero (Ns)
    arr = np.zeros((SEQ_LEN, 4), dtype=np.float32)
    # Compute where seq[0] should land in arr:
    start = HALF_LEN - center_index
    end   = start + len(seq)
    # Clip to [0, SEQ_LEN)
    seq_start = max(0, -start)
    arr_start = max(0, start)
    length    = min(len(seq) - seq_start, SEQ_LEN - arr_start)
    for i in range(length):
        arr[arr_start + i] = one_hot_encode_char(seq[seq_start + i])
    return tf.constant(arr[np.newaxis], dtype=tf.float32)  # batch dim

def run_enformer_exon_deletion(wt_seq, del_seq, exon_start, exon_end):
    # exon_start/end are positions in wt_seq where deletion occurs
    # Choose center_index (e.g. midpoint of deleted region in WT)
    center_idx = (exon_start + exon_end)//2

    # Prepare one-hot inputs
    wt_input = seq_to_fixed_one_hot(wt_seq, center_idx)
    del_input= seq_to_fixed_one_hot(del_seq, center_idx)

    # Load model
    model = Enformer(MODEL_PATH)
    # sequence_one_hot = one_hot_encode(fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH)))
    # predictions = model.predict_on_batch(sequence_one_hot[np.newaxis])['human'][0]

    # Run predictions (batch=1)
    ref_out = model.predict_on_batch(wt_input)['human'][0]  # shape [bins, tracks]
    del_out = model.predict_on_batch(del_input)['human'][0]

    return ref_out, del_out

def compare_gene_expression(ref_out, del_out, output_csv="outputs/delta_expression.csv"):
    center_bin = ref_out.shape[0] // 2
    wt_expr = ref_out[center_bin - CENTER_WINDOW:center_bin + CENTER_WINDOW, CAGE_START:CAGE_END].mean(axis=0)
    del_expr = del_out[center_bin - CENTER_WINDOW:center_bin + CENTER_WINDOW, CAGE_START:CAGE_END].mean(axis=0)
    delta_expr = del_expr - wt_expr

    print("\nTop Δ-expression CAGE tracks:")
    top_indices = np.argsort(np.abs(delta_expr))[::-1][:10]
    response = "\nTop Δ-expression CAGE tracks:\n"
    for i in top_indices:
        print(f"CAGE Track {CAGE_START + i}: Δ = {delta_expr[i]:.4f}")
        statement = f"CAGE Track {CAGE_START + i}: Δ = {delta_expr[i]:.4f}\n"
        response+=statement

    # Save full CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame({
        "track_index": np.arange(CAGE_START, CAGE_END),
        "wt_expression": wt_expr,
        "del_expression": del_expr,
        "delta_expression": delta_expr
    })
    df.to_csv(output_csv, index=False)
    print(f"\nFull Δ-expression CSV saved to: {output_csv}")

    return response

def main(wt_file, delta_file, exon_start, exon_end, output_dir='./outputs/api'):

    save_delta_png_path = os.path.join(output_dir,"delta_plot.png")
    save_tracks_png_path = os.path.join(output_dir,"tracks.png")
    save_delta_expr_csv = os.path.join(output_dir, "delta_expression.csv")

    wt_seq = SeqIO.read(wt_file, "fasta").seq
    del_record = SeqIO.read(delta_file, "fasta")
    del_seq = del_record.seq
    chrom = del_record.id

    ref_out, predictions = run_enformer_exon_deletion(wt_seq, del_seq, exon_start, exon_end)
    plot_delta(ref_out, predictions, chrom, exon_start, exon_end, png_path=save_delta_png_path)

    tracks = {'DNASE:CD14-positive monocyte female': predictions[:, 41],
          'DNASE:keratinocyte female': predictions[:, 42],
          'CHIP:H3K27ac:keratinocyte female': predictions[:, 706],
          'CAGE:Keratinocyte - epidermal': np.log10(1 + predictions[:, 4799])}
    
    target_interval = kipoiseq.Interval(chrom, format(exon_start, '_'), format(exon_end, '_'))
    print("Target Interval:", target_interval)
    plot_tracks(tracks, target_interval, png_path=save_tracks_png_path)

    response = compare_gene_expression(ref_out, predictions, output_csv=save_delta_expr_csv)

    return response, save_delta_png_path, save_tracks_png_path, save_delta_expr_csv

if __name__=="__main__":
    wt_file = "/home/ubuntu/exon_deletion/outputs/test_196k/deleted_exon_WT.fasta"
    delta_file = "/home/ubuntu/exon_deletion/outputs/test_196k/deleted_exon_DEL.fasta"
    exon_start = 15000
    exon_end = 15150
    main(wt_file, delta_file, exon_start, exon_end)