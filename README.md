# Fake News Detection System

## Project Description

The Fake News Detection System is a machine learning-based solution designed to identify and classify news articles as real or fake. This system helps combat misinformation by analyzing textual content using Natural Language Processing (NLP) and Deep Learning techniques. It provides an efficient way to verify news credibility, ensuring a more reliable information ecosystem.

## Features

✅ Advanced Machine Learning Models – Uses deep learning models like CNN, LSTM, and Transformers.

✅ Real-time News Classification – Quickly determines whether an article is fake or real.

✅ Natural Language Processing (NLP) – Includes text tokenization, stemming, stop-word removal, and TF-IDF vectorization.

✅ Scalability – Supports large datasets for training and prediction.

✅ User-Friendly Interface – Can be integrated into a web or mobile application.

## Tech Stack

🔹 Programming Language: Python 3.7+

🔹 Machine Learning Frameworks: TensorFlow, PyTorch, Scikit-learn

🔹 NLP Libraries: NLTK, SpaCy, Gensim

🔹 Database: MySQL, SQLite (for storing datasets

🔹 Web Framework (Optional): Flask / FastAPI for API 

🔹 Version Control: Git & GitHub

## System Architecture

1️⃣ Data Collection - Collects real and fake news datasets  

2️⃣ Data Preprocessing - Cleans text (removes stop words, stemming, tokenization)  

3️⃣ Feature Extraction - Uses TF-IDF and word embeddings  

4️⃣ Model Training - Trains using CNN/LSTM/BERT models  

5️⃣ Fake News Classification - Predicts if news is real or fake  

6️⃣ Evaluation - Analyzes accuracy, precision, recall, and F1-score  

## System Architecture Diagram:

 Installation & Setup
```
# Clone the repository
git clone https://github.com/your-username/fake-news-detection.git  
```
```
# Navigate to project directory
cd fake-news-detection  
```
```
# Install dependencies
pip install -r requirements.txt  
```
```
# Run the model
python train_model.py  
```
```
# Test with a sample news article
python predict.py --input "Sample news article text here"
```

 ## Model Performance
The model achieved 92.8% accuracy on test data, with a balanced precision-recall score, making it highly effective for fake news classification.

### Performance Metrics:

🔹 Accuracy: 92.8%

🔹 Precision: 91.0%

🔹 Recall: 89.5%

🔹 F1-Score: 90.2%

### Output Examples

![image](https://github.com/user-attachments/assets/e9f72dfd-920e-413b-8a7c-b329d29cecdb)

![image](https://github.com/user-attachments/assets/5d964727-793e-40aa-ba5d-39f4887a6485)

![image](https://github.com/user-attachments/assets/426cd62f-4dad-4600-ae64-412f7d717d04)

## Future Improvements

🔹 Integration with social media platforms for real-time fake news detection.

🔹 Enhancement using Transformer-based models (BERT, GPT-4).

🔹 Deployment as a browser extension or chatbot.

🔹 Support for multiple languages using NLP translation models.

## References
1.Iftikhar, M., & Ali, A. (2023). Fake news detection using machine learning. In Proceedings of the 3rd International Conference on Artificial Intelligence (ICAI). IEEE.

2.Shu, K., Wang, S., & Liu, H. (2018). FakeNewsTracker: A tool for fake news collection, detection, and visualization. In Proceedings of the 27th ACM International Conference on Information and Knowledge Management (pp. 1627-1630). ACM.

3.Kaliyar, R. K., Goswami, A., Narang, P., & Sinha, S. (2021). FakeBERT: Fake news detection in social media with a BERT-based deep learning approach. Multimedia Tools and Applications 

4.Raza, S., & Ding, C. (2021). Fake news detection based on news content and social contexts: A transformer-based approach. International Journal of Data Science and Analytics, 12(3) 

5.Rubin, V. L., Chen, Y., & Conroy, N. J. (2015). Automatic deception detection: Methods for finding fake news. Proceedings of the Association for Information Science and Technology

6. Gupta, N.S. & Rout, S.K. (2024). Enhancing Fake News Detection using Hybrid ML Models. EAI Endorsed Transactions on AI.
   
7.Shu, K., Wang, S., & Liu, H. (2018). FakeNewsTracker: A tool for fake news detection. ACM Knowledge Management Conference.

8.Kaliyar, R.K., et al. (2021). FakeBERT: Fake news detection with BERT. Multimedia Tools & Applications Journal.
