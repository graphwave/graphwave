"""Training and evaluation script for GraphWave."""

import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from config import get_args, get_mapping
from datasets import TrafficDataset
from model import NetworkTrafficContextual, NetworkTrafficModel, NetworkTrafficTemporal
from utils import get_contents_in_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, loader, optimizer, criterion, num_classes):
    """Run one training epoch. Returns (average loss, accuracy)."""
    model.train()
    total_loss = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        if (data.y < 0).any() or (data.y >= num_classes).any():
            raise ValueError(
                f"Label out of valid range [0, {num_classes - 1}]: "
                f"min={data.y.min()}, max={data.y.max()}")
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()

        del data, out, pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_loss / len(loader), correct / len(loader.dataset)


def evaluate(model, loader, criterion):
    """Compute loss and accuracy. Returns (loss, accuracy, preds, labels)."""
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)

            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()

            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(data.y.cpu().tolist())

    return total_loss / len(loader), correct / len(loader.dataset), all_preds, all_labels


def evaluate2(model, loader, criterion):
    """Detailed evaluation: predictions, probabilities, per-class ROC/AUC,
    and a record of every misclassified sample."""
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    all_probs = []

    total_errors = 0
    false_positive = 0      # true label 0 -> predicted non-0
    false_negative = 0      # true label non-0 -> predicted 0
    malicious_confusion = 0 # true non-0 -> predicted non-0 (wrong class)
    error_cases = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)

            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()

            # Collect misclassified samples (five-tuple + true/predicted labels)
            for i in range(len(data.y)):
                if pred[i] != data.y[i]:
                    ft = data.five_tuple[i]
                    num_nodes = data.graph_nodes[i].item()
                    true_l = data.y[i].item()
                    pred_l = pred[i].item()

                    total_errors += 1
                    if true_l == 0 and pred_l != 0:
                        false_positive += 1
                    elif true_l != 0 and pred_l == 0:
                        false_negative += 1
                    else:
                        malicious_confusion += 1

                    error_cases.append({
                        "five_tuple": ft,
                        "true_label": true_l,
                        "pred_label": pred_l,
                        "context_all_five_tuples": data.context_five_tuples[i],
                        "num_nodes": num_nodes
                    })

            # Use tolist() instead of .numpy() for compatibility with
            # torch 1.13 + numpy 2.x environments
            probs = np.array(torch.softmax(out, dim=1).cpu().tolist())
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(data.y.cpu().tolist())
            all_probs.extend(probs)

    print(f'Total misclassified samples: {total_errors}')
    print(f'False positives (benign -> malicious): {false_positive}')
    print(f'False negatives (malicious -> benign): {false_negative}')
    print(f'Malicious-to-malicious confusion: {malicious_confusion}')

    # Per-class ROC curve and AUC (one-vs-rest)
    n_classes = out.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(all_labels) == i,
                                      [prob[i] for prob in all_probs])
        roc_auc[i] = auc(fpr[i], tpr[i])
    macro_auc = np.mean(list(roc_auc.values()))

    return {
        'loss': total_loss / len(loader),
        'accuracy': correct / len(loader.dataset),
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'macro_auc': macro_auc,
        'error_cases': error_cases
    }


def save_results(results, filename="evaluation_results.json"):
    results_serializable = {
        "loss": float(results["loss"]),
        "accuracy": float(results["accuracy"]),
        "predictions": [int(x) for x in results["predictions"]],
        "labels": [int(x) for x in results["labels"]],
        "probabilities": [list(map(float, prob)) for prob in results["probabilities"]],
        "fpr": {str(k): v.tolist() for k, v in results["fpr"].items()},
        "tpr": {str(k): v.tolist() for k, v in results["tpr"].items()},
        "roc_auc": {str(k): float(v) for k, v in results["roc_auc"].items()},
        "macro_auc": float(results["macro_auc"]),
        "error_cases": results["error_cases"]
    }

    with open(filename, "w") as f:
        json.dump(results_serializable, f, indent=4)

    # Save misclassified samples separately
    error_filename = filename.replace(".json", "_error_cases.json")
    with open(error_filename, "w") as f:
        json.dump(results["error_cases"], f, indent=4)


def tSNE(test_loader, model, filename, downsample_ratio=1.0):
    """Project the contextual / temporal / fused features of the test set to
    2-D with t-SNE and render a 3-panel scatter plot.

    Only supported for the full fusion model (which returns intermediate
    features when ``visualize=True``).
    """
    all_contextual_features = []
    all_temporal_features = []
    all_fused_features = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            _, contextual_features, temporal_features, fused_features = model(data, visualize=True)
            all_contextual_features.extend(contextual_features.cpu().tolist())
            all_temporal_features.extend(temporal_features.cpu().tolist())
            all_fused_features.extend(fused_features.cpu().tolist())
            all_labels.extend(data.y.cpu().tolist())

    all_contextual_features = np.array(all_contextual_features)
    all_temporal_features = np.array(all_temporal_features)
    all_fused_features = np.array(all_fused_features)
    all_labels = np.array(all_labels)

    if downsample_ratio < 1.0:
        num_samples = int(len(all_labels) * downsample_ratio)
        indices = np.random.choice(len(all_labels), num_samples, replace=False)
        all_contextual_features = all_contextual_features[indices]
        all_temporal_features = all_temporal_features[indices]
        all_fused_features = all_fused_features[indices]
        all_labels = all_labels[indices]

    tsne = TSNE(n_components=2, random_state=42, perplexity=40, learning_rate=350, n_iter=1000)
    contextual_features_2d = tsne.fit_transform(all_contextual_features)
    temporal_features_2d = tsne.fit_transform(all_temporal_features)
    fused_features_2d = tsne.fit_transform(all_fused_features)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 24

    num_classes = len(np.unique(all_labels))
    color_palette = plt.get_cmap('tab20').colors[:num_classes]
    cmap = ListedColormap(color_palette)

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    def plot_square_subplot(ax, x, y, title, labels, cmap, num_classes):
        """Scatter plot with a colorbar and equal x/y ranges."""
        scatter = ax.scatter(x, y, c=labels, cmap=cmap, s=30, edgecolor='k', alpha=0.8)
        ax.set_title(title)
        cbar = fig.colorbar(scatter, ax=ax, ticks=range(num_classes),
                            orientation="vertical", pad=0.01)
        cbar.set_label('Class Labels')
        cbar.set_ticks(range(num_classes))
        cbar.set_ticklabels([f'Label {i}' for i in range(num_classes)])

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        # Keep the subplot square by equalizing the x and y ranges
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        range_max = max(x_max - x_min, y_max - y_min)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        ax.set_xlim(x_center - range_max / 2, x_center + range_max / 2)
        ax.set_ylim(y_center - range_max / 2, y_center + range_max / 2)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('')
        ax.set_ylabel('')

    plot_square_subplot(axes[0], contextual_features_2d[:, 0], contextual_features_2d[:, 1],
                        'Contextual Features (GAT)', all_labels, cmap, num_classes)
    plot_square_subplot(axes[1], temporal_features_2d[:, 0], temporal_features_2d[:, 1],
                        'Temporal Features', all_labels, cmap, num_classes)
    plot_square_subplot(axes[2], fused_features_2d[:, 0], fused_features_2d[:, 1],
                        'Fusion Features', all_labels, cmap, num_classes)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = get_args()
    mapping, num_classes = get_mapping(args.dataset)

    # Output directories (model checkpoints / figures / evaluation results)
    # are created next to the data directory
    base_path = Path(args.data_path).parent.parent
    model_path = base_path / 'model' / args.dataset
    fig_path = base_path / 'fig' / args.dataset
    evaluate_result_path = base_path / 'evaluation_result' / args.dataset
    for path in (model_path, fig_path, evaluate_result_path):
        path.mkdir(parents=True, exist_ok=True)

    # Load graphs and split into train / val / test
    starttime = time.time()
    data_files = get_contents_in_dir(args.data_path, ['.'], ['contextual.npy'])
    datasets = [TrafficDataset(file, args.dataset) for file in data_files]
    dataset = torch.utils.data.ConcatDataset(datasets)
    print(f'Graph construction time: {(time.time() - starttime) / len(dataset):.4f} s / sample')

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Model selection: contextual-only / temporal-only ablations, or the full
    # cross-attention fusion model (default)
    if args.contextual == 'yes':
        model = NetworkTrafficContextual(num_classes=num_classes).to(device)
        result_filename = 'contextual'
    elif args.temporal == 'yes':
        model = NetworkTrafficTemporal(num_classes=num_classes).to(device)
        result_filename = 'temporal'
    else:
        model = NetworkTrafficModel(num_classes=num_classes).to(device)
        result_filename = 'fusion'

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    if args.tsne == 'yes':
        tSNE(test_loader, model, os.path.join(fig_path, "t-SNE_before.png"))

    # Training loop with early stopping
    best_val_acc = 0
    patience = 30
    counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    m_path = os.path.join(model_path, args.dataset + '_' + result_filename + '_model.pth')

    if os.path.exists(m_path):
        print("Model checkpoint already exists, skip training.")
    else:
        for epoch in range(50):
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, num_classes)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0
                torch.save(model.state_dict(), m_path)
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

            print(f'Epoch {epoch:02d}')
            print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            print(f'Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}\n')

    # Final test on the best checkpoint
    model.load_state_dict(torch.load(m_path))
    results = evaluate2(model, test_loader, criterion)
    test_loss = results['loss']
    test_acc = results['accuracy']
    preds = results['predictions']
    labels = results['labels']
    json_path = os.path.join(evaluate_result_path, args.dataset + '_' + result_filename + '.json')
    save_results(results, filename=json_path)

    print('\nFinal Test Results:')
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')

    # Classification report and confusion matrix
    print('\nClassification Report:')
    class_indices = sorted(mapping.values())
    class_names = [k for k, v in sorted(mapping.items(), key=lambda item: item[1])]

    report = classification_report(
        labels,
        preds,
        labels=class_indices,
        target_names=class_names,
        zero_division=0,
        digits=4
    )
    print(report)

    report_path = os.path.join(evaluate_result_path, args.dataset + '_' + result_filename + '.txt')
    with open(report_path, 'w') as file:
        file.write(report)
    print(f"Classification report saved to {report_path}")

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(fig_path, args.dataset + '_' + result_filename + '.png'))

    if args.tsne == 'yes':
        tSNE(test_loader, model, os.path.join(fig_path, "t-SNE_after.png"))


if __name__ == "__main__":
    main()
