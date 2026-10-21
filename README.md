# Predicting Student Dropout

Can we tell, early on, which university students are at risk of dropping out? This project trains and compares several machine-learning models on real student records to answer that, and then looks at *which* factors matter most.

Built for COSC 4337 (Data Science) at the University of Houston by **Hoang Minh Bui** and **Caden L. Carrier-Rieger**.

## The data

[Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) from the UCI Machine Learning Repository: **4,424 students** with **36 features** each, collected from a higher-education institution in Portugal.

The features fall into a few groups:

- **Background:** age at enrollment, marital status, nationality, parents' education and occupation
- **Admission:** application mode, previous qualification and grade, admission grade
- **Finances:** scholarship holder, debtor, tuition fees up to date
- **Academic progress:** courses enrolled, evaluated, and approved, and average grade, for the 1st and 2nd semesters
- **Economy at the time:** unemployment rate, inflation rate, GDP

Each student is labeled **Dropout**, **Enrolled**, or **Graduate**. A copy is included as `data.csv` (semicolon-separated).

## What we did

We framed the problem two ways:

1. **Dropout vs. not (binary).** Enrolled students were grouped with graduates, since they haven't dropped out. This is what the written report covers.
2. **All three outcomes (multi-class).** Dropout vs. Enrolled vs. Graduate. This is what the notebook covers.

Features were standardized with `StandardScaler`, and the data was split 80/20 into training and test sets.

### Results — binary, from the report

| Model | Accuracy | F1 |
|-------|---------:|---:|
| Support Vector Machine (SVC) | 86% | 85% |
| MLP classifier (scikit-learn, 100-50-25 hidden layers) | 83% | 81% |
| **Custom neural network** (64-32-16, ReLU, 30% dropout) | **88%** | |

The first version of the custom network scored about 90% on training data but fell apart on validation data, which is a clear sign of overfitting. Adding dropout layers after the first two hidden layers brought training accuracy down slightly but made validation results line up with training, so the model actually generalized. That's why it was picked as the final model.

### Results — three classes, from the notebook (test set of 885 students)

| Model | Accuracy |
|-------|---------:|
| Random Forest | 76.2% |
| Logistic Regression | 75.3% |
| Neural network (16-16, softmax) | 74.4% |

Accuracy is lower here, and the confusion matrices show why: the models are good at spotting Dropouts and Graduates, but **Enrolled** students are hard to tell apart from either group (recall around 30–42%). That makes sense, because "still enrolled" is a snapshot of someone whose outcome isn't decided yet.

## What drives dropout

Using permutation importance on the final model, a handful of features stood out far above the rest:

1. **Courses approved in the 2nd semester** — by far the strongest signal
2. **Courses approved in the 1st semester**
3. Course grades and enrollment counts in each semester

In plain terms: how many classes a student actually passes in their first year says more about whether they'll finish than their background or admission scores do. Most other features had very small, similar importance.

## Files

| File | What it is |
|------|-----------|
| `minh_project.ipynb` | Main notebook: loading the data, exploring it, and training Random Forest, Logistic Regression, and a neural network for the three-class problem, with feature-importance charts |
| `minh_project.html` | The same notebook exported to HTML, so you can read the results without running anything |
| `COSC 4337 Report.pdf` | Full written report, including the SVM, MLP, and custom neural network comparison |
| `DATA SCIENCE PROJECT2.pptx` | Presentation slides |
| `Project Code` | Early data-prep script for the binary version of the problem |
| `data.csv` | The dataset |

## Running the notebook

```bash
git clone https://github.com/Minhbui100/Predict-Students-Dropout.git
cd Predict-Students-Dropout
pip install pandas numpy scikit-learn matplotlib seaborn ucimlrepo tensorflow jupyter
jupyter notebook minh_project.ipynb
```

The notebook downloads the dataset with `ucimlrepo`, so it needs an internet connection. The local `data.csv` is there if you'd rather load it yourself:

```python
df = pd.read_csv("data.csv", delimiter=";")
```

The neural network cell needs TensorFlow/Keras. Everything else only needs scikit-learn.
