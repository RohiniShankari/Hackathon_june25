Here's a more concise version of your `README.md`, keeping it professional and clear:

---

````markdown
# 🧪 LLM-Based Retrosynthesis Agent with Biocatalysis

An intelligent retrosynthesis planner built with LangGraph + LLMs, integrating biocatalysis, enzyme reasoning, and patent search to enable explainable and scalable synthesis planning.

---

## 🚀 Features

- ✅ LLM-guided retrosynthetic planning (from name or SMILES)
- 🔬 Biocatalytic reaction prediction (RetroBioCat)
- 🧬 EC number + enzyme prediction with FASTA retrieval
- 🧠 Atom mapping + reaction condition inference
- 📄 Similar patent search with LLM-based similarity reasoning
- 🧾 Summary generation in YAML/Markdown
- 🔗 Future: AlphaFold/Boltz integration for structure validation

---

## 📦 Setup

```bash
git clone https://github.com/boltzmannlabs/Hackathon_june25.git
cd synagent
conda create -n langgraph_env python=3.9 -y
conda activate langgraph_env
pip install -r tools/CLAIRE/requirements.txt
pip install -e tools/CLAIRE/rxnfp/
pip install -r tools/RetroBioCat_2/
pip install -e tools/OpenNMT
pip install rxnmapper
````

---

## 🧠 Usage

```bash
python main.py.py 
```
---

## ⚙️ Architecture

```mermaid
flowchart TD
    start(["__start__"])
    end(["__end__"])

    greet_and_route(["greet_and_route"])
    get_compound_info(["get_compound_info"])
    ask_user(["ask_user"])
    bio_reactions(["bio_reactions"])
    select_reactions(["select_reactions"])
    ec_number(["ec_number"])
    enzyme_lookup(["enzyme_lookup"])
    ask_for_standard_synthesis(["ask_for_standard_synthesis"])
    other_tools(["other_tools"])
    get_atom_mapping(["get_atom_mapping"])
    recommend_conditions(["recommend_conditions"])
    search_patents(["search_patents"])
    summarize_and_restart(["summarize_and_restart"])

    start --> greet_and_route
    greet_and_route --> get_compound_info
    greet_and_route --> ask_for_standard_synthesis
    greet_and_route --> other_tools
    greet_and_route --> end

    get_compound_info --> ask_user
    ask_user --> bio_reactions
    bio_reactions --> select_reactions
    select_reactions --> ec_number
    ec_number --> enzyme_lookup
    enzyme_lookup --> search_patents

    bio_reactions --> ask_for_standard_synthesis
    ask_user --> ask_for_standard_synthesis

    other_tools --> get_atom_mapping
    get_atom_mapping --> recommend_conditions
    recommend_conditions --> search_patents

    ask_for_standard_synthesis --> other_tools

    search_patents --> summarize_and_restart
    summarize_and_restart --> greet_and_route

```

---

## 🔭 Roadmap

* AlphaFold/Boltz enzyme validation
* UI (Gradio/Streamlit)
* Wet-lab protocol + synthesis cost estimation

---

## 📄 License

MIT

---

## 📬 Contact

[yourname@youremail.com](mailto:yourname@youremail.com)
[yourdomain.co](https://yourdomain.co)

```

---

Let me know if you want this split into actual project files (`run_agent.py`, `tools/`, etc.) or Dockerized.
```
