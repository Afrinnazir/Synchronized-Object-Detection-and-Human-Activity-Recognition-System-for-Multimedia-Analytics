# Synchronized-Object-Detection-and-Human-Activity-Recognition-System-for-Multimedia-Analytics


## 📝 Project Description

This project integrates the power of **Object Detection (OD)** and **Human Activity Recognition (HAR)** with an added touch of **Natural Language Processing (NLP)** to analyze multimedia data effectively. By leveraging cutting-edge models like **YOLOv8** and **LSTM**, the system provides a seamless fusion of object and activity recognition with real-time descriptive insights. 🌐

### 🌟 Key Applications:
- 🎥 **Surveillance**: Enhanced scene analysis and real-time monitoring.
- 🏥 **Healthcare**: Automated patient activity tracking.
- 🏭 **Industrial Automation**: Streamlined operational efficiency.

---

## ✨ Features

✅ **Real-Time Object Detection**: Accurately identifies and localizes objects using YOLOv8.  
✅ **Activity Recognition**: Learns complex temporal patterns with an LSTM model for recognizing human actions.  
✅ **Seamless Integration**: Merges OD and HAR results to provide comprehensive scene understanding.  
✅ **NLP-Driven Insights**: Generates detailed, human-readable descriptions of detected scenes.  
✅ **Performance Metrics**: Achieves high accuracy with detailed precision, recall, and F1-score analyses.  
✅ **Modular Testing**: Test individual modules (OD, HAR) or the full integration with ease.  

---

## 🛠️ Tech Stack

- **Models**: YOLOv8 🦾, LSTM 🧠  
- **Programming Language**: Python 🐍  
- **Libraries & Frameworks**:  
  - Machine Learning: `PyTorch`, `TensorFlow`  
  - Data Manipulation: `NumPy`, `Pandas`  
  - Visualization: `Matplotlib`, `Seaborn`  
  - NLP: `NLTK`  
  - Computer Vision: `OpenCV`

---

## 📦 Modules Implemented

1. 🎯 **Object Detection (`od.py`)**: Detects and classifies objects in multimedia data.  
2. 🕺 **Human Activity Recognition (`har.py`)**: Recognizes complex human activities from sequential frames.  
3. 🔗 **Integration with NLP (`Integration_NLP.py`)**: Combines OD and HAR with natural language outputs.  
4. 📊 **Evaluation Metrics**: Analyzes performance with confusion matrices, precision, recall, and F1 scores.

---

## ⚙️ Setup Instructions

1. **Download Data & Models**:  
   Get the required datasets and trained models from [this link](https://drive.google.com/drive/folders/1kbJnAsMOfXh3wHCJhk0ufgQkhKW24cpR?usp=sharing).  
 

2. **Clone the Repository**:  
   ```bash
   git clone https://github.com/Afrinnazir/Synchronized-Object-Detection-and-Human-Activity-Recognition-System-for-Multimedia-Analytics.git
   
3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
---
## 🚀 Running the Code

## 🟢 To test Object Detection only:  
    ```bash
    python od.py
 
## 🟢 To test Human Activity Recognition only:  
    ```bash
    python har.py

 ## 🟢 To run the integrated system:
     ```bash
     python Integration_NLP.py

---

## 🌈 Deployment

Currently designed for local execution. Future plans include deploying in real-world scenarios like:  
- 🔒 **Security Surveillance**  
- 🏥 **Healthcare Monitoring**  
- ⚙️ **Industrial Automation**  

---

## 🌟 Future Enhancements

- 🎯 **Enhanced Synchronization**: Better real-time fusion of OD and HAR results.  
- 💡 **Context-Aware NLP**: Use spatial relationships for improved descriptions.  
- 🌐 **Cloud-Based Deployment**: Scale the system for broader applications.  
- 🧪 **Field Testing**: Test in real-world scenarios for refinement.


---

## 📸 Output Screenshots

### Object Detection Results  
**Detection 1:**  
![Object Detection Output 1](screenshots/object_detection1.jpg)  

**Detection 2:**  
![Object Detection Output 2](screenshots/object_detection2.jpg)  

### Human Activity Recognition Results  
**Activity 1:**  
![Human Activity Recognition Output 1](screenshots/human_activity_recognition1.jpg)  

**Activity 2:**  
![Human Activity Recognition Output 2](screenshots/human_activity_recognition2.jpg)  

### Integrated System Results  
**Integration 1:**  
![Integrated System Output 1](screenshots/integrated_output1.jpg)  

**Integration 2:**  
![Integrated System Output 2](screenshots/integrated_output2.jpg)  




