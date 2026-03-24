# Import libraries
import torch
import torch.nn.functional as F
import numpy as np
import shap
import pandas as pd

# Put model in evaluation mode
# Don't forget to use:
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
model.eval()

# Channel names (21 EEG + 1 EKG)
ch_names = [
    'Fp1-F7', 'F7-T1', 'T1-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T2',
    'T2-T4', 'T4-T6', 'T6-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz', 'Pz-Oz',
    'EKG-bipolar'
]

global_idx = 0  
loop = 0        
# Shap Optimization
print("Extracting background features...")
background_features, _, _ = next(iter(train_loader))
# Slice to 30 to speed up execution by 4x without losing accuracy
small_background = background_features[:30].to(device)
explainer = shap.GradientExplainer(model, background_features.to(device))

metadata_registry = []
total_samples = len(test_loader.dataset)
print(f"Starting Fast Scanner Loop for {total_samples} samples...")

# Pass 1: The Fast Scanner Loop (No Plotting just compute)
for batch_features, batch_labels, _ in test_loader:
    for i in range(batch_features.size(0)):
        single_feature = batch_features[i:i + 1].to(device)
        true_label = batch_labels[i].item()

        sample_idx = global_idx
        global_idx += 1
        
        # 1. Get Confidence
        with torch.no_grad():
            outputs_no_grads = model(single_feature)
            probabilities  = F.softmax(outputs_no_grads, dim=1)
            predicted_label = outputs_no_grads.argmax().item()
            confidence = probabilities[0, predicted_label].item() 

        # 2. Get SHAP Values mathematically
        shap_values_4d = explainer.shap_values(single_feature) 
        shap_2d = shap_values_4d[0, :, :, true_label]
        importance_per_channel = shap_2d.mean(axis=1) 

        # 3. Calculate Focality Metrics
        channel_focus_variance = np.var(importance_per_channel)
        max_shap_intensity = np.max(shap_2d)
        dominant_channel_idx = np.argmax(importance_per_channel)
        dominant_channel_name = ch_names[dominant_channel_idx]

        # 4. Record the metadata securely
        metadata_registry.append({
            'sample_idx': sample_idx,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'is_correct': true_label == predicted_label,
            'confidence': confidence,
            'channel_focus_variance': channel_focus_variance,
            'peak_intensity': max_shap_intensity,
            'dominant_channel': dominant_channel_name 
        })
        if sample_idx % 20 == 0:
            # Print progress every 20 samples so you know it hasn't frozen
            print(f"Processed {sample_idx} / {total_samples} samples ==> True: {true_label}, Pred: {predicted_label}")
            
    loop += 1 

# Save to CSV
df_metadata = pd.DataFrame(metadata_registry)
df_metadata.to_csv('Compute_Result/data_driven_window_metadata.csv', index=False)
print(f"\nFinished processing {global_idx} samples safely. Metadata saved!")

