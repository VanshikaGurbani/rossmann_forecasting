# src/explainability.py

import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

def run_shap_explanation(model, X_train, X_test):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)
    shap.summary_plot(shap_values, X_test, show=True)

def plot_partial_dependence_rf(model, X, top_features):
    fig, ax = plt.subplots(figsize=(12,8))
    PartialDependenceDisplay.from_estimator(model, X, features=top_features, ax=ax)
    plt.suptitle('Partial Dependence Plots')
    plt.show()
