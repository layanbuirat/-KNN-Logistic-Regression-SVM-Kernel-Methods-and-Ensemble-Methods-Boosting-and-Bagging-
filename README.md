# Machine Learning Classification on Iris Dataset  
**ENCS5341 - Assignment #3**  
**Electrical and Computer Engineering Department**

---

## 📌 Project Overview  
This project implements and compares multiple machine learning classification algorithms on the **Iris Dataset** to predict the species of iris flowers based on four morphological features. The models evaluated include:

- **K-Nearest Neighbors (KNN)**  
- **Logistic Regression (with L1 & L2 Regularization)**  
- **Support Vector Machines (SVM)**  
- **Ensemble Methods (AdaBoost and Random Forest)**  

The goal is to analyze model performance, compare results, and determine the most effective approach for this well-known classification task.

---

## 📂 Dataset  
**Iris Dataset** – 150 samples, 3 species, 4 features:  
- Sepal Length (cm)  
- Sepal Width (cm)  
- Petal Length (cm)  
- Petal Width (cm)  

Species:  
- Iris-setosa  
- Iris-versicolor  
- Iris-virginica  

Source: [Kaggle - Iris Dataset](https://www.kaggle.com/datasets/uciml/iris)

---

## 🧠 Models & Methods  

### 1. **K-Nearest Neighbors (KNN)**  
- Tested with Euclidean, Manhattan, and Cosine distance metrics.  
- Optimal **K = 6** determined via 5-fold cross-validation.  
- **Best accuracy: 98%** (Euclidean & Manhattan).  

### 2. **Logistic Regression**  
- Trained with **L1 (Lasso)** and **L2 (Ridge)** regularization.  
- **L1 performed better** (93.33% accuracy) vs L2 (90% accuracy).  

### 3. **Support Vector Machines (SVM)**  
- Kernels tested: Linear, Polynomial, and RBF.  
- **RBF kernel achieved the highest accuracy: 98.33%**.  

### 4. **Ensemble Methods**  
- **AdaBoost** (with Decision Trees): 93% accuracy.  
- **Random Forest** (Bagging): 98% accuracy.  
- Random Forest outperformed AdaBoost in overall classification.  

---

## 📊 Results Summary  

| Model                 | Accuracy  | Best Configuration          |
|-----------------------|-----------|-----------------------------|
| KNN                   | 98%       | K=6, Euclidean/Manhattan    |
| Logistic Regression   | 93.33%    | L1 Regularization           |
| SVM                   | 98.33%    | RBF Kernel                  |
| AdaBoost              | 93%       | 50 estimators, Decision Tree|
| Random Forest         | 98%       | 100 estimators              |

---

## 🏆 Key Findings  
- **KNN** and **SVM (RBF)** achieved the highest accuracies (~98%).  
- **Distance-based metrics** (Euclidean/Manhattan) outperformed Cosine in KNN.  
- **Random Forest** outperformed AdaBoost in ensemble methods.  
- **Logistic Regression** was less effective than KNN and SVM for this dataset.  
- **Ensemble methods** generally provided better generalization and robustness compared to individual models.

---

## 👥 Group Contribution  

| Member          | Tasks                                                                 |
|-----------------|-----------------------------------------------------------------------|
| **Rana Musa**   | KNN, Ensemble Methods (AdaBoost & Random Forest), Report sections     |
| **Leyan Burait**| Logistic Regression, SVM, Report sections                             |

---

## 🛠️ Technologies Used  
- Python  
- Scikit-learn  
- Pandas, NumPy  
- Matplotlib (for visualization)  
- Jupyter Notebook (optional)

---

## 📈 Conclusion  
The **Iris Dataset** is well-suited for comparing classification algorithms. In this study, **KNN with Euclidean distance** and **SVM with RBF kernel** delivered the best performance, both achieving above 98% accuracy. Ensemble methods, especially **Random Forest**, also performed excellently, demonstrating the value of model aggregation for improved stability and accuracy.

This project serves as a practical exploration of fundamental ML algorithms and provides clear insights into model selection for simple yet classic classification problems.

---
**Course:** ENCS5341 - Machine Learning and Data Science  
**Instructor:** Dr. Ismail Khater  
**Section:** #3  
**Submitted by:** Rana Musa (1210007), Leyan Burait (1211439)
