# 🎓 Student Grade Predictor

This project builds a machine learning pipeline to predict whether students will pass or fail based on the [UCI Student Performance dataset](https://archive.ics.uci.edu/ml/datasets/student+performance).

## Features

- Data preprocessing with Pandas
- Decision Tree and Logistic Regression models
- Confusion matrix visualizations

## How to Run It

### 1. Clone the repo
```bash
git clone https://github.com/behzadjanjua/student-grade-predictor.git
cd student-grade-predictor
```

### 2. Set up the environment
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Add the dataset
Download [`student-mat.csv`](https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip)  
→ unzip → place inside the `data/` folder.

### 4. Run the app
```bash
python main.py
```

## 📝 License
MIT License
