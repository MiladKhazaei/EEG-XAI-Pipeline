# Import Libraries
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# evaluation Mode
model.eval() 
# Use defaultdict to store both hard predictions AND raw probabilities per signal
signal_preds = defaultdict(list)
signal_probs = defaultdict(list)
signal_labels = {}
# This tells PyTorch not to calculate gradients
with torch.no_grad():
    for X_test_batch, y_test_batch, y_test_idx in test_loader:
        X_test_batch = X_test_batch.to(device)
        y_test_batch = y_test_batch.to(device)
        test_outputs = model(X_test_batch)
        # Get raw probabilities using Softmax
        probabilities = F.softmax(test_outputs, dim=1)
        # Probability of Seizure (class = 1)
        prob_class_1 = probabilities[:,1].cpu().numpy()
        # Get hard predictions 
        _, test_predicted = torch.max(test_outputs, 1)
        predicted = test_predicted.cpu().numpy()
        # y_test_batch = y_test_batch.cpu().numpy()
        y_test_idx =y_test_idx.numpy()
        # Map windows back to their parent signals
        for i in range(len(y_test_idx)):
            sig_id = y_test_idx[i]
            signal_preds[sig_id].append(predicted[i])
            # Store probability
            signal_probs[sig_id].append(prob_class_1[i])
            if(sig_id not in signal_labels):
                signal_labels[sig_id]= y_test_batch[i].item() # cpu().numpy()
                
# Aggregate Window-Level data to Signal-Level
all_signal_labels=[]
all_signal_predicted = []
all_signal_probabilities = []
for signal_id in signal_labels.keys():
    # Hard Voting (Majority Vote) for Confusion Matrix
    majority_pred = np.argmax(np.bincount(signal_preds[signal_id]))
    # Soft Voting (Mean probability) for ROC Curve
    mean_prob = np.mean(signal_probs[signal_id])

    all_signal_labels.append(signal_labels[signal_id])
    all_signal_predicted.append(majority_pred)
    all_signal_probabilities.append(mean_prob)

all_signal_labels = np.array(all_signal_labels)
all_signal_predicted = np.array(all_signal_predicted)
all_signal_probabilities = np.array(all_signal_probabilities)

# You can simply, save it to prevent re-calculate.
# np.save('all_signal_labels_UCI.npy',all_signal_labels)

# Confusion Matrix components (signal-level)
tn, fp, fn, tp = confusion_matrix(all_signal_labels, all_signal_predicted, labels=[0, 1]).ravel()
epsilon = 1e-8
accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
precision = tp / (tp + fp + epsilon)
recall = tp / (tp + fn + epsilon)
specificity = tn / (tn + fp + epsilon)
f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

print(f"""
================ Signal-Level Evaluation Report ================
Total Signals: {len(all_signal_labels)}
\n--- Confusion Matrix ---
    TP (Seizure correct): {tp} | TN (PNES correct): {tn} | FP (PNES as Seizure): {fp} | FN (Seizure as PNES): {fn}
\n--- Key Performance Metrics ---
    Accuracy: {accuracy:.4f}
    Recall (Sensitivity): {recall:.4f}
    Specificity: {specificity:.4f}
    Precision: {precision:.4f}
    F1-Score: {f1_score:.4f}
===============================================================
      """)

# ROC Curve (Using Soft Voting)
fpr, tpr, _ = roc_curve(all_signal_labels, all_signal_probabilities)
roc_auc = auc(fpr, tpr)

# Create 1 * 2 Panel
fig, axs = plt.subplots(1,2,figsize=(14, 6))
# Panel A: Confusion Matrix
cm = confusion_matrix(all_signal_labels, all_signal_predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['PNES','Seizure'])
disp.plot(ax=axs[0], cmap='Blues', values_format='d')
axs[0].set_title('(a) Confusion Matrix (Signal-Level Hard Voting)')
# Panel B: ROC Curve
axs[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
axs[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axs[1].set_xlim([0.0, 1.0])
axs[1].set_ylim([0.0, 1.05])
axs[1].set_xlabel('False Positive Rate (1 - Specificity)')
axs[1].set_ylabel('True Positive Rate (Sensitivity)')
axs[1].set_title('(b) ROC (Signal-Level Soft Voting)')
axs[1].legend(loc="lower right")

plt.suptitle("Model Performance on Dataset (ES vs. PNES)", fontsize=16)
plt.tight_layout()
plt.savefig('Figure_7_Performance.png', dpi=600, bbox_inches='tight')
plt.grid(True)
plt.show()