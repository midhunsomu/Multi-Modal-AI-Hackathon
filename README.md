#  Multi-Modal AI Hackathon
# Automated Event Analysis from Live-Stream Snippets


##  Project Overview

This project focuses on building a **Multi-Modal Deep Learning System** that analyzes **short video snippets from live-streams** to classify event types and detect anomalous or unusual activities.  
By integrating **audio**, **video**, and **textual data** (user comments), the system aims to achieve real-time contextual understanding — enabling use cases such as **content moderation**, **highlight generation**, and **safety alert detection**.

---

##  Motivation

Every second, thousands of live-streams generate massive volumes of unstructured data.  
Manual monitoring for moderation, highlight detection, or emergency alerts is impractical at scale.  

By leveraging **AI-driven fusion of image, sound, and text**, this project enables:  
- Real-time identification of key moments (e.g., a winning goal, sudden silence, crowd excitement).  
- Detection of anomalies such as security incidents, spam comments, or unexpected disruptions.  
- Smarter, automated content analytics and event understanding.  

---

##  Objective

Develop a **multi-modal AI pipeline** that can:  
1. Classify live-stream snippets into one of five event categories:  
   `Sporting_Event`, `Music_Concert`, `Tech_Conference`, `Gaming_Stream`, `Political_Rally`  
2. Detect **anomalous** snippets using unsupervised learning.  

The project explores **feature extraction**, **sensor fusion**, and **anomaly detection** using state-of-the-art deep learning techniques.

---

##  Dataset Description

The dataset contains **10,000 short (5–10 sec)** snippets from real live-streams.  
Each snippet includes:

| Data Type | Description |
|------------|--------------|
| **Image Data** | 10 frames extracted from the video |
| **Audio Data** | `.wav` file containing the snippet’s sound |
| **Text Data** | `.json` file with timestamped user comments |
| **Labels** | Event category + anomaly flag |

---

##  Problem Breakdown

### 1️⃣ Data Preprocessing & Feature Fusion 

**Modality-Specific Feature Extraction**
- **Image:** Use CNNs (e.g., ResNet, VGG) for frame-level embeddings.  
- **Audio:** Extract MFCCs or use pre-trained models like VGGish/PANNs.  
- **Text:** Clean and embed comments using BERT/DistilBERT.

**Feature Alignment & Fusion**
- Normalize feature dimensions.  
- Fuse embeddings using **concatenation**, **weighted averaging**, or **attention mechanisms**.

---

### 2️⃣ Multi-Modal Model Development 

**Classifier Design**
- Build a classifier (e.g., MLP or XGBoost) using fused embeddings.  
- Optimize with dropout, batch normalization, and learning rate tuning.  
- Evaluate using **accuracy, precision, recall, F1-score**, and **confusion matrix**.

**Modality Contribution Analysis**
- Compare multi-modal vs. single-modality baselines.  
- Identify strongest predictive modality and explain its impact.

---

### 3️⃣ Anomaly Detection 

**Autoencoder-Based Approach**
- Train an **autoencoder** on non-anomalous snippets.  
- Compute **reconstruction errors** as anomaly scores.  
- Visualize distributions and set thresholds for detection.

**Interpretation of Anomalies**
- Analyze top anomalous snippets and provide real-world context (e.g., technical glitch, crowd panic, spam flood).

---

##  Getting Started

###  Prerequisites
- Python 3.8+
- TensorFlow / PyTorch
- Jupyter Notebook
- Access to the official dataset (provided by organizers)

---

### Installation & Setup

```bash
# Clone the repository
git clone https://github.com/midhunsomu/Multi-Modal-AI-Hackathon.git
cd Multi-Modal-AI-Hackathon
```
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```


##  Team Members

The project was collaboratively developed by a talented and multidisciplinary team.

| Name |
|------|
| **Midhun S** |
| **Aadithya R** |
| **Rohith Prem** |
| **Srivatsan V** |
| **Devasanjay N** |


## Key Results & Insights

- **Multi-Modal Fusion Boost:** Combining image, audio, and text modalities improved F1-score by over **15%** compared to single-modality models.

- **Anomaly Detection:** The autoencoder achieved high sensitivity in identifying unusual event snippets.

- **Interpretability:** Audio features were most effective in distinguishing Music Concerts vs Political Rallies, while text features enhanced context understanding.



