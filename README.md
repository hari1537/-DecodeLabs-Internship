# DecodeLabs Internship 

Welcome to my comprehensive Artificial Intelligence engineering repository tracked during my **DecodeLabs Internship**. It documents my step-by-step progression from foundational rule-based scripts to supervised machine learning classifiers, preference-based recommendation engines, and deep learning computer vision networks.

Every module features a fully functional backend pipeline decoupled and served through a clean, modern **Streamlit web application interface**.

---

## 📂 Repository Dashboard & Project Matrix

| Module | Core Paradigm | Key Implemented Skills | Interface Mechanism | Status |
| **Project 1** | Rule-Based AI Engine | Control flows, Input normalization, Text handling | Live Text Input Matcher | **Completed** |
| **Project 2** | Supervised Data Classifier | Matrix array handling, Train-test split ratios, Mathematical metrics | Sliders & Prediction Metrics | **Completed** |
| **Project 3** | Content Recommendation Logic | Vector geometry dot-products, Metric similarity score matching | Real-Time Preference Cards | **Completed** |
| **Project 4** | Deep Learning Vision Engine | Multi-dimensional tensors, Pre-trained networks, Array transformation | Drag & Drop Image Uploader | **Completed** |

---

## 🚀 Deep-Dive Project Explanations

### 🤖 Project 1: Rule-Based AI Chatbot Logic Engine
* **Objective:** Build a clean input-process-output (IPO) chatbot engine responding to specific conditional parameters.
* **Core Skills:** Intent parsing, string normalization, edge-case routing constraints.

####  Implementation Details
This script implements a strict **Deterministic White Box System**. To protect against unexpected runtime crashes or invalid query inputs, raw text data passes through a string cleaning layer using the `.lower().strip()` format. Once clean, a branching structure (`if-elif-else`) checks for specific patterns (e.g., greetings, help prompts, status updates) and maps them cleanly to their assigned outputs.

---

###  Project 2: Data Classification Using AI
* **Objective:** Ingest, structure, evaluate, and train a supervised machine learning algorithm using a classic multi-feature dataset.
* **Core Skills:** Matrix slicing, train-test splitting models, validation metrics.

####  Implementation Details
This module moves away from static rules into **Probabilistic Machine Learning** using the classic *Iris Flower Dataset*. 
1. **Data Handling:** Numerical physical attributes are formatted as a feature matrix ($X$), and targeted classifications are mapped to an integer label vector ($y$).
2. **Train-Test Splitting:** The data is split using a strict validation strategy ($80\%$ for model pattern training, $20\%$ isolated for testing validation).
3. **Model Training:** A `DecisionTreeClassifier` is trained to discover data splitting thresholds automatically. It displays model accuracy on a dashboard alongside interactive sliders for testing species classification changes live.

---

###  Project 3: AI Vector-Space Recommendation Engine
* **Objective:** Develop a content-matching logic routine that pairs item catalogs with dynamic user affinity preferences.
* **Core Skills:** Similarity algorithms, score normalization, feature array weights.

####  Implementation Details
This project builds a responsive **Content-Based Recommender System**. Rather than performing basic string matches, it calculates similarity inside a multi-attribute vector space. The user sets preference levels ($0-5$) across multiple topics using sliders, which generates a custom personal profile vector. The system calculates a dot product score against the pre-tagged movie/TV database, normalizes the values into clear **Match Confidence Percentages**, and surfaces the top three content cards instantly.

---

###  Project 4: Deep Learning Image Recognition Engine
* **Objective:** Deploy a state-of-the-art deep convolutional neural network to classify objects from raw user image uploads.
* **Core Skills:** Neural network inferences, multi-dimensional tensor formatting, image data pipelines.

####  Implementation Details
Project 4 implements **Computer Vision Basics** using **PyTorch**. It initializes **MobileNetV2**, an industry-standard deep learning convolutional network pre-trained on millions of real-world images from the ImageNet database.
1. **The Vision Pipeline:** Raw uploaded images are caught by the interface, converted to clean RGB color channels, and center-cropped to a strict $224 \times 224$ matrix layout.
2. **Mathematical Normalization:** The pixel values are transformed into normalized multi-dimensional tensors.
3. **Inference Execution:** Tensors run through the model layers, and a softmax activation computes prediction probabilities, highlighting the primary target class with its matching accuracy percentage.

---

##  Environmental Installation & Local Launch

To spin up and interact with these module dashboards across your personal machine environment, verify that you have Python installed and run this setup command in your terminal:

```bash
pip install streamlit pandas scikit-learn pillow torchvision

To run any of the interactive web dashboards, execute them with Streamlit's internal module path runner:
# Run Project 1
python -m streamlit run chatbot.py

# Run Project 2
python -m streamlit run classification_project.py

# Run Project 3
python -m streamlit run recommendation_project.py

# Run Project 4
python -m streamlit run vision_project.py


Complete update of README for all 4 projects







