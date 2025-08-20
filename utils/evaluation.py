from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import streamlit as st

def evaluate_model(model, X_test, y_test, model_type='xgb'):
    if model_type == 'tabnet':
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
    elif model_type == 'tfmlp':
        y_proba = model.predict(X_test).flatten()
        y_pred = (y_proba > 0.5).astype(int)
    else:
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    return accuracy, roc_auc

def plot_metrics(results):
    import matplotlib.pyplot as plt
    import streamlit as st

    models = list(results.keys())
    accuracies = [results[m][0] for m in models]
    rocs = [results[m][1] for m in models]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#4F8BF9', '#FFB547', '#54BD94']

    # Accuracy Bar Chart
    ax[0].bar(models, accuracies, color=colors[:len(models)])
    ax[0].set_title(" Accuracy", fontsize=14, pad=18)
    ax[0].set_ylim(0, 1)
    for i, v in enumerate(accuracies):
        ax[0].text(i, v + 0.02, f"{v:.3f}", ha='center', color=colors[i], fontsize=12)

    # ROC AUC Bar Chart
    ax[1].bar(models, rocs, color=colors[:len(models)])
    ax[1].set_title(" ROC AUC", fontsize=14, pad=18)
    ax[1].set_ylim(0, 1)
    for i, v in enumerate(rocs):
        ax[1].text(i, v + 0.02, f"{v:.3f}", ha='center', color=colors[i], fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)

