# MOSearchEngine

A search engine implementation for the Cranfield dataset using Vector Space Model (VSM), BM25, and Language Model (LM) ranking algorithms. This project evaluates the performance of these models using the `trec_eval` tool.

---

## Table of Contents
1. [Setup](#setup)
2. [Install Dependencies](#install-dependencies)
3. [Dataset](#dataset)
4. [Running the Project](#running-the-project)
5. [Evaluation with `trec_eval`](#evaluation-with-trec_eval)
6. [Project Structure](#project-structure)
7. [Requirements](#requirements)

---

## Setup

### Clone the Repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/your-username/MOSearchEngine.git
cd MOSearchEngine
```

## Install Dependencies

### 1. Install Python Dependencies
Install the required Python packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2. Install NLTK Resources
The project uses NLTK for text preprocessing. Download the required NLTK resources by running the following Python command:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 3. Install trec_eval
The project uses `trec_eval` for evaluating the search engine's performance. Follow these steps to install it:

#### On Linux/macOS:
Download the `trec_eval` tool from the official repository:

```bash
wget https://github.com/usnistgov/trec_eval/archive/refs/tags/v9.0.8.tar.gz
tar -xvzf v9.0.8.tar.gz
cd trec_eval-9.0.8
```

Compile and install:

```bash
make
sudo make install
```

Verify the installation:

```bash
trec_eval -h
```

#### On Windows:
- Download the precompiled binary for `trec_eval` from [here](https://github.com/usnistgov/trec_eval).
- Add the binary to your system's PATH.

---

## Dataset

### Cranfield Dataset
The project uses the Cranfield dataset, which consists of:

- `cran.all.1400.xml`: Document collection.
- `cran.qry.xml`: Query collection.
- `cranqrel.trec.txt`: Relevance judgments.

Place these files in the `datasets` folder:

```plaintext
MOSearchEngine/
└── datasets/
    ├── cran.all.1400.xml
    ├── cran.qry.xml
    └── cranqrel.trec.txt
```

---

## Running the Project

### 1. Run the Search Engine
Execute the `main.py` script to run the search engine and evaluate the models:

```bash
python main.py
```

### 2. Command-Line Arguments
You can customize the run using the following arguments:

```plaintext
--docs: Path to the documents XML file (default: datasets/cran.all.1400.xml).
--queries: Path to the queries XML file (default: datasets/cran.qry.xml).
--qrels: Path to the relevance judgments file (default: datasets/cranqrel.trec.txt).
--output: Directory to save evaluation results (default: results).
--no-cache: Disable caching of the inverted index (default: False).
--metrics: Metrics to evaluate (default: map, P.5, ndcg).
```

#### Example:

```bash
python main.py --docs datasets/cran.all.1400.xml --queries datasets/cran.qry.xml --qrels datasets/cranqrel.trec.txt --output custom_results --metrics map ndcg
```

---

## Evaluation with `trec_eval`

The project automatically runs `trec_eval` to evaluate the performance of the models. The results are saved in the `results` folder (or the folder specified by `--output`).

### Metrics
The following metrics are computed by default:

- **MAP**: Mean Average Precision.
- **P.5**: Precision at 5.
- **nDCG**: Normalized Discounted Cumulative Gain.

### View Results
After running the project, you can view the evaluation results in the `results` folder:

```plaintext
results/
├── vsm_results.txt
├── bm25_results.txt
└── lm_results.txt
```

---

## Project Structure

```plaintext
MOSearchEngine/
│
├── main.py
├── requirements.txt
├── README.md
├── datasets/
│   ├── cran.all.1400.xml
│   ├── cran.qry.xml
│   └── cranqrel.trec.txt
├── modules/
│   ├── __init__.py
│   ├── evaluation.py
│   ├── indexing.py
│   ├── parser.py
│   ├── preprocess.py
│   └── ranking.py
└── results/
    ├── vsm_results.txt
    ├── bm25_results.txt
    └── lm_results.txt
```

---

## Requirements

The project requires the following Python packages, which are listed in `requirements.txt`:

```plaintext
click==8.1.8
joblib==1.4.2
nltk==3.9.1
numpy==2.2.3
regex==2024.11.6
scipy==1.15.2
tqdm==4.67.1
```

---

## Troubleshooting

### 1. ModuleNotFoundError
If you encounter a `ModuleNotFoundError`, ensure that:

- The `modules` folder contains an `__init__.py` file.
- The root directory (`MOSearchEngine`) is in your `PYTHONPATH`. You can add it manually:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/MOSearchEngine
```

### 2. `trec_eval` Not Found
If `trec_eval` is not found, ensure that:

- It is installed and added to your system's `PATH`.
- You are running the command in a terminal where `trec_eval` is accessible.