# Personalized Student Recommendations

This project aims to develop a Python-based solution to analyze quiz performance and provide students with personalized recommendations to improve their preparation. By analyzing past performance data, the system generates tailored insights, providing valuabe inputs to the student.

---

## Preprocessing

The Performance data from the previous quizzes underwent preprocessing to extract relevant features.

---

## Exploratory Data analysis

Using the extracted features from the historical data a comprehensive analysis including Performance Trends Over Time, by topic, diffculty and other relavant metrics.

---

## Machine learning model

Using the historical data a Random Forest classifier was built to predict the student's performance level. Additionally, the features that were deemed important by the model, were used to generate insights and propose actionable steps for the user to improve upon.

---

## Execution

Ensure python is installed. Install the dependencies via

```python
pip install pandas numpy matplotlib scikit-learn joblib seaborn json re fuzzywuzzy jupyter
```
Open Jupyter notebook to explore the codebase

```python
python -m notebook
```

To run the model for a particular case,
 ```python
 python test.py
 ```

 ## Screenshots

 
![time_analysis](https://github.com/user-attachments/assets/0a114460-17f3-4350-aa49-d7203623de20)
![topic_analysis](https://github.com/user-attachments/assets/780eb60e-49e8-4ee7-93cc-597f404df584)
![difficulty_level_analysis](https://github.com/user-attachments/assets/2d09e8f8-e3ca-498c-b75d-f9ce3fad8305)
<img width="1042" alt="insights" src="https://github.com/user-attachments/assets/56594122-5e15-4074-bc64-fa0239fa6b69" />


