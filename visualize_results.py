import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from trainer import evaluate
from model import BinaryClassifier


def validation_loss(results):
    plt.figure(figsize=(12, 6))
    for loss_name, result in results.items():
        plt.plot(result["train_losses"], label=f"{loss_name} (Train)")
        plt.plot(result["val_losses"], linestyle='--', label=f"{loss_name} (Validation)")

    plt.title("Convergence Speed: Training and Validation Loss Over Epochs", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.5)
    plt.show()


def validation_accuracy(results):
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for loss_name, result in results.items():
            ax.plot(result["metrics"][metric], label=f"{loss_name}")
        ax.set_title(f"{metric.capitalize()} Over Epochs", fontsize=12)
        ax.set_xlabel("Epochs", fontsize=10)
        ax.set_ylabel(metric.capitalize(), fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.5)

    if len(metrics) % 2 != 0:
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.show()


def validation_accuracy_heatmap(metrics, results):
    final_metrics = {loss_name: {metric: result["metrics"][metric][-1] for metric in metrics} for loss_name, result in results.items()}
    metrics_df = pd.DataFrame(final_metrics).T

    plt.figure(figsize=(10, 6))
    sns.heatmap(metrics_df, annot=True, cmap="coolwarm", fmt=".3f", cbar=True, linewidths=0.5)
    plt.title("Final Metrics Heatmap Across Loss Functions", fontsize=14)
    plt.ylabel("Loss Function", fontsize=12)
    plt.xlabel("Metrics", fontsize=12)
    plt.show()



def get_robust_outlier(test_loader, results, loss_functions, device="cpu"):
    robust_metrics = {}
    outlier_metrics = {}

    for loss_name, result in results.items():
        model = BinaryClassifier().to(device)
        model.load_state_dict(result["model_state_dict"]) 

        noisy_metrics = evaluate(model, test_loader, loss_functions[loss_name], device=device)  

        outlier_metrics[loss_name] = noisy_metrics[1]  
        robust_metrics[loss_name] = {"Loss Value": noisy_metrics[0]}  

    robust_df = pd.DataFrame(robust_metrics).T
    outlier_df = pd.DataFrame(outlier_metrics).T

    return robust_df, outlier_df
    

def get_noisy_robust_outlier(noisy_test_loaders, results, loss_functions, device="cpu"):
    robust_metrics = {}
    outlier_metrics = {}

    for loss_name, result in results.items():
        model = BinaryClassifier().to(device)
        model.load_state_dict(result["model_state_dict"]) 

        for noisy_type, noisy_loader in noisy_test_loaders.items():
            noisy_metrics = evaluate(model, noisy_loader, loss_functions[loss_name], device=device)  

            outlier_metrics[f"{loss_name} with {noisy_type} noisy"] = noisy_metrics[1]  
            robust_metrics[f"{loss_name} with {noisy_type} noisy"] = {"Loss Value": noisy_metrics[0]}  

    robust_df = pd.DataFrame(robust_metrics).T
    outlier_df = pd.DataFrame(outlier_metrics).T

    return robust_df, outlier_df


def validation_loss_robust(robust_df, noise=""):
    plt.figure(figsize=(14, 7))
    robust_df.plot(kind='bar', figsize=(14, 7), colormap='viridis', alpha=0.85, width=0.8)
    if noise:
        plt.title("Loss Robustness on Noisy Data", fontsize=16, weight='bold')
    else:
        plt.title("Loss Robustness on Test Data", fontsize=16, weight='bold')

    plt.ylabel("Loss Values", fontsize=14)
    if noise:
        plt.xlabel("Loss Function and Noise Type", fontsize=14)
    else:
        plt.xlabel("Loss Function Type", fontsize=14)

    plt.xticks(rotation=30, ha='right', fontsize=10)  
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.6, linestyle='--')  
    if noise:
        plt.legend(fontsize=12, title="Noise Types", title_fontsize=13, loc='upper right')  
    else:
        plt.legend(fontsize=12, title="Loss Types", title_fontsize=13, loc='upper right')  

    plt.tight_layout() 
    plt.show()

def validation_accuracy_outlier(outlier_df):
    plt.figure(figsize=(14, 7))
    outlier_df.plot(kind='bar', figsize=(14, 7), colormap='autumn', alpha=0.85, width=0.8)
    plt.title("Model Interpretability: Performance with Outliers", fontsize=16, weight='bold')
    plt.ylabel("Metric Values", fontsize=14)
    plt.xlabel("Loss Function and Outlier Type", fontsize=14)
    plt.xticks(rotation=30, ha='right', fontsize=10) 
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.6, linestyle='--')  
    plt.legend(fontsize=12, title="Outlier Types", title_fontsize=13, )  # loc='upper right'
    plt.tight_layout() 
    plt.show()