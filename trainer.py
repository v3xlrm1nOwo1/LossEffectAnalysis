import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC


def evaluate(model, data_loader, loss_function, device="cpu"):
    model.eval()
    total_loss, preds, targets = 0, [], []

    # Initialize metric objects
    accuracy = Accuracy(task="binary").to(device=device)
    precision = Precision(task="binary").to(device=device)
    recall = Recall(task="binary").to(device=device)
    f1 = F1Score(task="binary").to(device)
    auroc = AUROC(task="binary").to(device)

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device=device), labels.to(device=device).float()
            outputs = model(inputs).squeeze()
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            preds.append(outputs)
            targets.append(labels) 

    preds = torch.cat(tensors=preds)
    targets = torch.cat(tensors=targets)

    preds_binary = (preds > 0.5).int()

    metrics = {
        "accuracy": accuracy(preds_binary, targets.int()).item(),
        "precision": precision(preds_binary, targets.int()).item(),
        "recall": recall(preds_binary, targets.int()).item(),
        "f1": f1(preds_binary, targets.int()).item(),
        "auc": auroc(preds, targets.int()).item(),
    }

    mean_loss = total_loss / len(data_loader)
    return mean_loss, metrics



def train_evaluate(model, train_loader, val_loader, loss_function, optimizer, epochs=30, device="cpu"):
    train_losses, val_losses = [], []
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        val_loss, val_metrics = evaluate(model, val_loader, loss_function, device)
        val_losses.append(val_loss)

        for key, value in val_metrics.items():
            metrics[key].append(value)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {val_metrics['accuracy']:.4f}")

    return train_losses, val_losses, metrics
