# 🧬 Exon Deletion Impact Predictor

This repository implements an end-to-end pipeline to simulate exon deletion from a gene, predict its effects on splicing and gene expression, and visualize the results using modern deep learning tools like **SpliceAI** and **Enformer**. It includes both a CLI and an interactive Gradio web interface.


## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```
git clone https://github.com/boltzmannlabs/Hackathon_june25.git
cd exon_deletion
```
### 2️⃣ Create the Conda Environment

```
conda env create -f ./installs/environment.yaml
conda activate exon

(or)

conda create -n exon python=3.8 -y
conda activate exon
sh ./installs/env.sh
```

### 3️⃣ Install the tools Package
```
pip install -e .
```

## 🔑 AWS Credentials
Some parts of the pipeline may access AWS-hosted model weights or services. Be sure to export your AWS credentials before running:
```
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
```

## 🌐 Launching Gradio App
To use the browser-based GUI:
```
python gradio_app.py
```

## Usage Prompts
To run the agent, use these example prompts:

**Custom gene fasta file:**
* Upload wt_file only and mention `wt_file` in the prompt and required positions
* prompt = `Delete exon from given wt_file and positions 200 and 205`

**Direct gene to splicing scores:**
* prompt = `Get exon coordinates of TP53 gene and predict splicing changes if exon at index 1 on chromosome 17 is deleted.`

**Getting exon coords for particular gene:**
* prompt = `Get exon coordinates of TP53 gene`

**Predicting only splice scores at given chromosome and positions:**
* prompt = `Predict splicing changes if exon at positions 7676521 and 7676594 on chromosome 17 is deleted`

**Enformer Predictions:**
* After download del and wt files from first tool, re-upload both files and give prompt like in this format
* prompt = `Compare regulatory changes using wt_file and delta_file between positions 200 and 500`

