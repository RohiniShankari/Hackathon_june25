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
conda env create -f environment.yaml
conda activate exon
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