## 🧬 DNA Sequence Classification for Forensic Origin Prediction

### 📖 Project Overview

This project focuses on **classifying synthetic DNA sequences** to predict their **biological origin** — whether they come from *Bacteria*, *Virus*, *Human*, or *Plant*.
It combines **classical machine learning** and **deep learning** to identify characteristic patterns in nucleotide sequences (A, T, C, G).

This project demonstrates how computational models can help in **forensic DNA analysis**, **biosecurity**, and **synthetic biology**.

---

### 🚀 Key Features

* Preprocessing of raw DNA sequences into numerical features
* Generation of **k-mer frequency** features (e.g., 3-mers like `ATG`, `CCT`)
* Classification using both:

  * 🧩 **Classical ML**: Random Forest, Logistic Regression, SVM
  * ⚡ **Deep Learning**: CNN / Bi-LSTM sequence models
* Visualizations:

  * Confusion matrices
  * ROC curves
  * Top discriminative 3-mers per class
* Auto-generated markdown report with visual results

---

### 🧠 Tech Stack

| Category      | Tools / Libraries                |
| ------------- | -------------------------------- |
| Language      | Python 3.10+                     |
| ML Frameworks | scikit-learn, TensorFlow / Keras |
| Visualization | Matplotlib, Seaborn              |
| Data Handling | Pandas, NumPy                    |
| Utility       | Joblib, JSON                     |

---

### 📁 Project Structure

```
DNA_Sequence_Classification/
│
├── DNA_sequence.ipynb              # Main Jupyter notebook (data processing + model training)
├── visualization_report.py         # Visualization & report generation script
├── synthetic_dna_dataset.csv       # Dataset used in this project
├── best_sequence_model.h5          # Trained CNN model
├── model_random_forest.pkl         # Trained Random Forest model
├── dna_tokenizer.json              # Tokenizer for DNA sequences
├── label_encoder.pkl               # Label encoder
│
├── confusion_ml.png                # Confusion matrix (Random Forest)
├── confusion_dl.png                # Confusion matrix (CNN)
├── roc_ml.png                      # ROC curve (Random Forest)
├── roc_dl.png                      # ROC curve (CNN)
├── top_kmers.png                   # Top 3-mer motifs visualization
├── report.md                       # Final Markdown report
│
└── README.md                       # This file
```

---

### ⚙️ How to Run

#### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/DNA_Sequence_Classification.git
cd DNA_Sequence_Classification
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

*(If `requirements.txt` doesn’t exist yet, generate it via:)*

```bash
pip freeze > requirements.txt
```

#### 3. Run the notebook

Open **`DNA_sequence.ipynb`** in Jupyter Notebook or Google Colab to preprocess data and train models.

#### 4. Generate visualizations & report

Once training is done and models are saved:

```bash
python visualization_report.py
```

This will create:

* Confusion matrix plots
* ROC curves
* Top 3-mer frequency plots
* A full markdown report (`report.md`)

---

### 📊 Example Outputs

| Visualization                         | Description               |
| ------------------------------------- | ------------------------- |
| ![Confusion Matrix](confusion_dl.png) | CNN Confusion Matrix      |
| ![ROC Curve](roc_dl.png)              | ROC Curve (CNN)           |
| ![Top k-mers](top_kmers.png)          | Discriminative DNA motifs |

---

### 🧩 Results Summary

| Model         | Accuracy | F1-Score | AUC    |
| ------------- | -------- | -------- | ------ |
| Random Forest | **94%**  | 0.94     | > 0.98 |
| CNN           | **93%**  | 0.93     | > 0.98 |

Both models perform excellently, showing strong ability to detect biological origins from sequence motifs.

---

### 🧬 Forensic Implications

* **Reliable origin tracing** for forensic or contamination samples
* **Motif detection** for distinguishing synthetic from natural DNA
* **High interpretability** — CNN learns “DNA fingerprints”

---

### 📈 Future Work

* Expand dataset with real-world forensic DNA samples
* Experiment with transformer-based DNA models (DNABERT, BioGPT)
* Integrate explainability tools like SHAP or Grad-CAM

---

### 👨‍💻 Author

**Rakibul Hasan**


