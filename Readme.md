# Personalized Student Recommendations

This project aims to develop a Python-based solution to analyze quiz performance and provide students with personalized recommendations to improve their preparation. By analyzing past performance data, the system generates tailored insights, providing valuabe inputs to the student.

--

## Preprocessing

The Performance data from the previous quizzes underwent preprocessing to extract relevant features.

--

## Exploratory Data analysis

Using the extracted features from the historical data a comprehensive analysis including Performance Trends Over Time, by topic, diffculty and other relavant metrics.

--

## Machine learning model

Using the historical data a Random Forest classifier was built to predict the student's performance level. Additionally, the features that were deemed important by the model, were used to generate insights and propose actionable steps for the user to improve upon.

--

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

 


