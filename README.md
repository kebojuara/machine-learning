# Machine Learning Projects

Five Jupyter notebooks from the **Machine Learning** course at **Institut Teknologi Sepuluh Nopember (ITS)**. They cover clustering, classification, text processing, and image recognition using Python.

## Projects

### [Customer Segmentation](Customer_Segmentation.ipynb)

Groups customers using the [Customer Personality Analysis dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data) from Kaggle. The notebook processes demographic, purchase, and campaign data, standardizes the features, and compares K-Means with four agglomerative clustering methods. PCA is used to visualize the clusters.

For four clusters, **K-Means scored 0.1068** on the silhouette metric; single, complete, and average linkage each scored approximately **0.7202**. The notebook compares cluster separation, though the scores alone do not establish whether the groups are useful customer segments.

### [Classification Algorithms](Classification_Algorithms.ipynb)

Four small classification experiments with calculations shown in Python:

- **k-Nearest Neighbors:** predicts a T-shirt size from height and weight using Manhattan distance.
- **Categorical Naive Bayes:** calculates class probabilities from categorical inputs.
- **Mixed-Feature Naive Bayes:** combines categorical probabilities with Gaussian likelihoods.
- **Decision Tree:** calculates entropy and information gain for a hiking decision.

The datasets are defined in the notebook, making it possible to inspect each calculation directly.

### [Dry Bean Variety Classification](Dry_Bean_Variety_Classification.ipynb)

Classifies seven bean varieties using the [UCI Dry Bean dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset), with **13,611 samples** and **16 numerical features**. The notebook compares k-NN, Gaussian Naive Bayes, Decision Tree, and SVM using a stratified train-test split, classification reports, and confusion matrices.

**SVM performed best** in the recorded test run, with **92.21% accuracy** and **0.9343 macro F1-score**.

### [SMS Spam Detection](SMS_Spam_Detection.ipynb)

Classifies messages as spam or ham using the [SMS Spam Collection dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). Messages are converted into TF-IDF features, then classified with Multinomial Naive Bayes, Linear SVM, and Decision Tree.

**Linear SVM reached 98.39% test accuracy** and a **0.94 spam F1-score**. It identified 132 of 149 spam messages in the test set. The notebook also predicts labels for new example messages.

### [Potato Leaf Disease Classification](Potato_Leaf_Disease_Classification.ipynb)

Classifies potato leaves as **Early Blight**, **Late Blight**, or **Healthy** using **1,721 images** selected from the [PlantVillage dataset](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage). A CNN trained on 224 × 224 images is compared with SVM and Random Forest models, including versions with balanced class weights.

The CNN achieved **93.90% validation accuracy** on 344 images. Its Healthy-class recall was **0.43**, so the notebook also examines per-class results, confusion matrices, and predictions on uploaded images. Training plots and model export are included.

## Running the Notebooks

1. Open a notebook in [Google Colab](https://colab.research.google.com/) and run the cells in order.
2. Install any packages requested by the notebook. The experiments use **Pandas**, **NumPy**, **scikit-learn**, **Matplotlib**, and, where needed, **TensorFlow/Keras**, **Seaborn**, `kagglehub`, or `ucimlrepo`.
3. For **Customer Segmentation**, download `marketing_campaign.csv` from the linked Kaggle dataset and upload it when prompted. The **Classification Algorithms** notebook uses data defined in its cells. The remaining notebooks fetch their datasets through their code, with a local-file fallback in Dry Bean Variety Classification.
4. **Potato Leaf Disease Classification** downloads images and trains a CNN. Its image-prediction cells prompt you to upload test images. A GPU runtime can shorten training time.

Saved outputs are included for reference. Results may vary when notebooks are rerun with different environments or dataset versions.

## Author

[William Hans Chandra](https://williamchandra.web.id/)
