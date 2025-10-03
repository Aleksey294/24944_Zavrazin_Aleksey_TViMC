import pandas as pd
import numpy as np
import random
import math
from collections import defaultdict

# Загружаем датасет
df = pd.read_csv("emails.csv")

# Последняя колонка = метка (0 - не спам, 1 - спам)
X = df.drop(columns=["Prediction", "Email No."])  # признаки
y = df["Prediction"].values  # целевая переменная

# Разделяем на train/test (80/20)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def train_naive_bayes(X, y):
    n_samples, n_features = X.shape
    spam_count = np.sum(y == 1)
    ham_count = np.sum(y == 0)
    
    # априорные вероятности
    p_spam = spam_count / n_samples
    p_ham = ham_count / n_samples

    # Подсчёт слов в спаме и не спаме
    word_counts_spam = np.sum(X[y == 1], axis=0) + 1  # сглаживание Лапласа
    word_counts_ham = np.sum(X[y == 0], axis=0) + 1

    total_spam_words = np.sum(word_counts_spam)
    total_ham_words = np.sum(word_counts_ham)

    # вероятности слов
    p_w_spam = word_counts_spam / total_spam_words
    p_w_ham = word_counts_ham / total_ham_words

    return {
        "p_spam": p_spam,
        "p_ham": p_ham,
        "p_w_spam": np.array(p_w_spam),
        "p_w_ham": np.array(p_w_ham)
    }

def predict_naive_bayes(model, X):
    log_prob_spam = np.log(model["p_spam"]) + np.dot(X, np.log(model["p_w_spam"]))
    log_prob_ham = np.log(model["p_ham"]) + np.dot(X, np.log(model["p_w_ham"]))
    return (log_prob_spam > log_prob_ham).astype(int)

# Обучение
model = train_naive_bayes(X_train.values, y_train)

# Предсказание
y_pred = predict_naive_bayes(model, X_test.values)

# Метрики
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

print("Точность:", acc)
print("Чувствительность (спам):", recall)
print("Специфичность (не спам):", specificity)
