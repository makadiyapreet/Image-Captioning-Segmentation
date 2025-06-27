# 🖼️ Image Captioning and Segmentation

This project focuses on building a deep learning pipeline that performs:
- 🧾 Natural language **image captioning**
- 🎯 Object-level **image segmentation**

The entire project is developed from scratch using the **MS COCO 2017 dataset** and does **not use any pre-trained models**. It is built and trained on a **MacBook Air M2 (CPU)** with dataset stored on an **external HDD**.

---

## 📁 Project Structure

```
Image-Captioning-Segmentation/
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_EDA.ipynb              # Exploratory Data Analysis
│   ├── 02_Captioning_Model.ipynb # CNN + LSTM Captioning
│   ├── 03_Segmentation_Model.ipynb # Segmentation Model (U-Net/Mask R-CNN)
│   ├── 04_Integration.ipynb      # Combined Captioning + Segmentation
│   └── 05_Evaluation.ipynb       # Metrics and result visualization
│
├── src/                          # Source code
│   └── captioning/
│       ├── dataset.py            # Custom COCO Caption Dataset
│       └── (model, train, etc.)  # Upcoming modules
│
├── models/                       # Trained weights
├── outputs/                      # Predictions and results
├── data/                         # Path reference (external HDD used)
├── results/                      # EDA summaries and pickles
├── requirements.txt              # Required Python packages
├── .gitignore                    # Files to exclude
└── README.md                     # Project documentation
```

---

## ✅ Project Progress

### 🗂️ Task 01 - Exploratory Data Analysis (EDA)
- **Dataset:** MS COCO 2017
- **Focus Areas:**
  - Caption statistics (mean length, common words)
  - Vocabulary size vs frequency threshold
  - Word cloud of frequent words (excluding stopwords)
  - Pattern analysis (starting with "a", containing "in", "with", etc.)
  - Captions per image distribution
  - Top categories from segmentation annotations
  - Bounding box dimensions and area distributions
- **Results Saved To:**  
  `/Volumes/ExternalHD/ICP/results/eda_results.pkl`

📊 **EDA Status:** ✅ Completed  
📁 Notebook: `notebooks/01_EDA.ipynb`

---

## 🔧 Technologies Used

- Python 3.x
- PyTorch (no pre-trained models used)
- TorchVision
- NumPy, Pandas, Matplotlib, Seaborn
- NLTK, WordCloud
- Jupyter Notebooks

---

## 💽 Dataset Info

- **Source:** [MS COCO 2017](https://cocodataset.org/#download)
- **External Path:** `/Volumes/ExternalHD/ICP/data/MSCOCO/`
- **Contents:**
  - `train2017/`
  - `val2017/`
  - `annotations/` (captions & segmentation)

> ⚠️ Dataset is stored on an **external HDD** and not uploaded to GitHub.

---

## 📌 Setup Instructions

```bash
git clone https://github.com/makadiyapreet/Image-Captioning-Segmentation.git
cd Image-Captioning-Segmentation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📅 Upcoming Tasks

| Task | Description | Status |
|------|-------------|--------|
| 02   | Captioning Model (CNN + LSTM) | ⏳ In Progress |
| 03   | Segmentation Model (U-Net or Mask R-CNN) | 🔜 Not Started |
| 04   | Integration of Both Pipelines | 🔜 Not Started |
| 05   | Evaluation Metrics & Visualization | 🔜 Not Started |

---

## 👤 Author

**Preet Makadiya**  
🔗 [LinkedIn](https://www.linkedin.com/in/preet-makadiya-13102004-p)  
🔗 [GitHub](https://github.com/makadiyapreet)

---

## 🏁 License

This project is for educational and research purposes only.
