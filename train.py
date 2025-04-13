import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from datasets2 import TrafficDataset
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader
from model2 import *
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from config import get_mapping, get_args
import seaborn as sns
import numpy as np
from utils import *
import argparse
import json
from build_graph import *
import time
from sklearn.manifold import TSNE

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.memory_allocated())
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        if (data.y < 0).any() or (data.y >= num_classes).any():
            print(f"Illegal label value! min={data.y.min()}, max={data.y.max()}")
            raise ValueError("The tag value is out of the valid range. [0, num_classes-1]")
        optimizer.zero_grad()
        out = model(data, visualize=False) 
        loss = criterion(out, data.y)
        
        loss.backward()
        torch.cuda.empty_cache()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()

        del data, out, pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_loss/len(loader), correct/len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data, visualize=False)
            loss = criterion(out, data.y)
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    
    return total_loss/len(loader), correct/len(loader.dataset), all_preds, all_labels

def evaluate2(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data, visualize=False)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            probs = torch.softmax(out, dim=1).cpu().numpy()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_probs.extend(probs)
    
    n_classes = out.shape[1] 
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(all_labels) == i, [prob[i] for prob in all_probs])
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
        'macro_auc': macro_auc
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
        "macro_auc": float(results["macro_auc"])
    }
    
    with open(filename, "w") as f:
        json.dump(results_serializable, f, indent=4)

def main():
    args = get_args()
    mapping, num_classes = get_mapping(args.dataset)
    
    data_path = args.data_path
    starttime = time.time()
    data_path = get_contents_in_dir(data_path, ['.'], ['contextual.npy'])
    datasets = [TrafficDataset(file) for file in data_path]
    dataset = torch.utils.data.ConcatDataset(datasets) 
    endtime = time.time()
    print(f'time for graph construction is: {(endtime-starttime)/len(dataset)}')
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
    
    result_filename = 'fusion'
    if(args.contextual == 'yes'):
        model = NetworkTrafficContextual(num_classes=num_classes).to(device)
        result_filename = 'contextual'
    elif(args.temporal == 'yes'):
        
        model = NetworkTrafficTemporal(num_classes=num_classes).to(device)
        result_filename = 'temporal'
    else:
        model = NetworkTrafficModel(num_classes=num_classes).to(device)
        result_filename = 'fusion'
    #
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    patience = 30
    counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(100):
        if(os.path.exists(args.dataset+'_'+result_filename+'_model.pth')):
            print("model has already exists!")
            break
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), args.dataset+'_'+result_filename+'_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        print(f'Epoch {epoch:02d}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        print(f'Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}\n')
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend()
    plt.savefig(args.dataset+'_'+result_filename+'_curve.png')
    
    model.load_state_dict(torch.load(args.dataset+'_'+result_filename+'_model.pth'))
    
    results = evaluate2(model, test_loader, criterion)
    test_loss = results['loss']
    test_acc = results['accuracy']
    preds = results['predictions']
    labels = results['labels']
    json_path = args.dataset+'_'+result_filename+'.json'
    save_results(results, filename=json_path)



    print(f'\nFinal Test Results:')
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')
    
    print('\nClassification Report:')
    if isinstance(preds, torch.Tensor):
        preds_np = preds.detach().cpu().numpy()
    else:
        preds_np = preds

    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = labels

    class_indices = sorted(mapping.values())
    class_names = [k for k, v in sorted(mapping.items(), key=lambda item: item[1])]

    report = classification_report(
        labels_np,
        preds_np,
        labels=class_indices,
        target_names=class_names,
        zero_division=0,
        digits=4
    )
    print(report)

    file_path = args.dataset+'_'+result_filename+'.txt'
    with open(file_path, 'w') as file:
        file.write(report)

    print(f"The report was successfully written to {file_path}")


    
    # 绘制混淆矩阵
    cm = confusion_matrix(labels_np, preds_np)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(args.dataset+'_'+result_filename+'.png')

if __name__ == "__main__":
    main()