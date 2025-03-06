# MindMetrics: Mental Health Analytics of Tweets
*This work is part of an IEEE research paper submission and is intended solely for academic research purposes. Use or redistribution without proper permission is prohibited.*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-blue)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-ScikitLearn-orange)
![NLP](https://img.shields.io/badge/NLP-Spacy-yellow)

This repository introduces a cutting-edge, modular framework for classifying tweets into mental health categories using advanced Natural Language Processing (NLP) and Machine Learning (ML) techniques. It serves as both an educational resource and a robust baseline for computational mental health research.


## Key Highlights

- **Diverse Model Architectures:**  
  Leverage a range of modern techniques including transformer models (BERT, DistilBERT, XLNet, GPT-2), zero-shot classification, topic modeling (LDA, NMF), and a rule-based classifier. This variety supports in-depth comparative analysis between contemporary and conventional methods.

- **Comprehensive End-to-End Pipeline:**  
  The framework spans from raw data ingestion and rigorous preprocessing to model evaluation and result visualization, offering a complete walkthrough of an ML workflow.

- **Reproducible & Extensible Research:**  
  With clearly defined modules and scripts, researchers can easily replicate results, benchmark performance, and extend the pipeline for custom experiments.

- **Advanced Technical Capabilities:**  
  Incorporates techniques such as GPU acceleration, mixed precision training, and extensive metric evaluations (ROC, Precision-Recall curves, classification reports) to offer deeper insights into model behavior.

- **Real-World Application:**  
  Provides practical strategies for handling noisy social media data, including sophisticated preprocessing, feature engineering, and robust evaluation methodologies.


## Repository Structure

### Data Preparation

- **Raw Data:**  
  The primary dataset (`mental.csv`) consists of unprocessed tweets.

- **Preprocessing Scripts:**  
  Clean and normalize text data by removing noise (e.g., mentions, URLs, digits, non-alphabetic characters), followed by tokenization, lemmatization, and stopword filtering.

### Modeling and Classification

- **Transformer-Based Models:**  
  Fine-tune and deploy BERT, DistilBERT, XLNet, and GPT-2 for high-performance text classification.

- **Zero-Shot Classification:**  
  Utilize transfer learning to predict labels without requiring explicit training on every category.

- **Topic Modeling:**  
  Implement Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF) to reveal latent structures and cluster similar content.

- **Rule-Based Approach:**  
  Employ a simple yet effective classifier based on token occurrence for initial categorization.

### Evaluation and Analysis

- **Performance Metrics:**  
  Measure model effectiveness using accuracy, precision, recall, and F1 scores.

- **Visualization Tools:**  
  Generate ROC and Precision-Recall curves, histograms, word clouds, and n-gram analyses to visually interpret data insights.

- **Comparative Analysis:**  
  Conduct detailed, side-by-side comparisons of model outputs with comprehensive classification reports.


## Technical Requirements

- **Programming Language:**  
  Python (3.7+)

- **Development Environment:**  
  Google Colab or any Jupyter Notebook setup with GPU support.

- **Core Libraries:**  
  - **Data Handling:** `pandas`, `numpy`
  - **Visualization:** `matplotlib`, `seaborn`, `WordCloud`
  - **NLP & ML:** `nltk`, `scikit-learn`, `transformers`, `tensorflow`, `torch`

- **API Integrations:**  
  Support for API-driven classification via OpenAI and Google Generative AI.



## Conclusion

This repository is a valuable asset for researchers, data scientists, and practitioners eager to explore and apply advanced NLP techniques to mental health analytics. By engaging with the codebase, you will enhance your understanding of state-of-the-art classification methods, model benchmarking, and the complexities of processing real-world textual data.

## Contributing

Contributions to this repo are welcome. Please open an issue or submit a pull request if you have suggestions for:
- New feature extraction and data preprocessing techniques.
- Additional machine learning or deep learning models.
- Enhanced visualization and interpretability tools.

Before using or modifying the project, please note that it is part of an IEEE research paper submission and should not be used without explicit permission.

## License

This project is licensed under the [MIT License](LICENSE).


## Contact

For questions, collaborations, or further details, please reach out:
- **Name:** Shib Kumar  
- **Email:** [shibkumarsaraf05@gmail.com](mailto:shibkumarsaraf05@gmail.com)  
- **GitHub:** [@shib1111111](https://github.com/shib1111111)
