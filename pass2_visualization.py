import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
import shap
import pandas as pd

# -------------------------------------------------
# 1. Initialization
# -------------------------------------------------
# Let's standardize your output folder to 'Journal_Visualizations'
base_out_dir = 'Visualizations'

os.makedirs(f'{base_out_dir}/manual_gradcam_heatmaps', exist_ok=True)
os.makedirs(f'{base_out_dir}/manual_gradcam_overlays', exist_ok=True)
os.makedirs(f'{base_out_dir}/shap_analysis_results', exist_ok=True)

model.eval() 

ch_names = [
    'Fp1-F7', 'F7-T1', 'T1-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T2',
    'T2-T4', 'T4-T6', 'T6-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz', 'Pz-Oz',
    'EKG-bipolar'
]
eeg_ch_names = ch_names[:]

target_layer = model.conv2 

# -------------------------------------------------
# 2. Grad-CAM Function
# -------------------------------------------------
def generate_and_save_gradcam(single_feature, class_to_explain, class_name,
                              true_label, predicted_label, sample_idx, loop, mode):
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    outputs = model(single_feature) 
    score = outputs[:, class_to_explain] 

    model.zero_grad(set_to_none=True) 
    score.backward() 

    pooled_gradients = torch.mean(gradients[0], dim=[2], keepdim=True) 
    feature_maps = activations[0] 
    weighted_feature_maps = (pooled_gradients * feature_maps).squeeze(0) 
    heatmap_2d = torch.relu(weighted_feature_maps).detach().cpu().numpy()
    
    forward_handle.remove()
    backward_handle.remove()

    # First plot
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(heatmap_2d, cmap='hot', aspect='auto')
    ax.set_title(f'Grad-CAM ({mode.upper()}) for class {class_name}\n(True: {true_label}, Pred: {predicted_label})')
    ax.set_xlabel('Time Points (Processed)')
    ax.set_ylabel(f'Processed Channels')
    plt.colorbar(im, ax=ax, label='Weighted Activation')
    plt.savefig(f'{base_out_dir}/manual_gradcam_heatmaps/sample_{sample_idx}_{mode}_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Second Plot
    fig, ax = plt.subplots(figsize=(20, 6))
    fig.suptitle(f'Overlays ({mode.upper()}) for class {class_name}\n(True: {true_label}, Pred: {predicted_label})', fontsize=20)

    heatmap_1d = heatmap_2d.mean(axis=0)
    resized_heatmap = zoom(heatmap_1d, 1024 / heatmap_1d.shape[0]) 

    separation_factor = 0.5
    y_tick_positions = []

    for ch_idx in range(22): 
        y_position = ch_idx * separation_factor
        y_tick_positions.append(y_position)
        eeg_wave = single_feature[0, ch_idx, :].cpu().numpy()
        eeg_wave_norm = (eeg_wave - np.mean(eeg_wave)) / (np.std(eeg_wave) + 1e-6)
        eeg_wave_scaled = eeg_wave_norm * (separation_factor / 2.5)

        time_axis = np.arange(eeg_wave.shape[0])
        ax.plot(time_axis, y_position + eeg_wave_scaled, color='blue', linewidth=0.8)

        ax.imshow(resized_heatmap[np.newaxis, :], cmap='hot', aspect='auto',
                  extent=[0, 1024, y_position - separation_factor / 2, y_position + separation_factor / 2],
                  alpha=0.6)

    ax.set_xlabel('Time Points', fontsize=12)
    ax.set_ylabel('Channels', fontsize=12)
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(eeg_ch_names)
    ax.invert_yaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{base_out_dir}/manual_gradcam_overlays/sample_{sample_idx}_{mode}_stacked_overlays.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


# -------------------------------------------------
# 3. Data-Driven ID Selection
# -------------------------------------------------
# NOTE: Ensure the path below matches where Pass 1 saved the CSV!
print("Reading metadata to find Golden IDs...")
data = pd.read_csv('Compute_Result/data_driven_window_metadata.csv') 

temporal_channels = ['T1-T3','T2-T4', 'T3-T5', 'T4-T6', 'F7-T1', 'F8-T2']
frontal_channels = ['Fp1-F7', 'Fp2-F8', 'Fp1-F3', 'Fp2-F4']

df_correct = data[data['is_correct'] == True]

df_seizure = df_correct[df_correct['true_label'] == 1]
df_seizure_temporal = df_seizure[df_seizure['dominant_channel'].isin(temporal_channels)]
best_seizure = df_seizure_temporal.sort_values(by=['confidence','channel_focus_variance'], ascending=[False, False])

df_pnes = df_correct[df_correct['true_label'] == 0]
df_pnes_frontal = df_pnes[df_pnes['dominant_channel'].isin(frontal_channels)]
best_pnes = df_pnes_frontal.sort_values(by=['confidence','channel_focus_variance'], ascending=[False, True])

# .head(3) isolates the top 3 IDs for each class (Total = 6 images)
top_seizure_ids = best_seizure['sample_idx'].head(3).tolist()
top_pnes_ids = best_pnes['sample_idx'].head(3).tolist()

final_target_ids = top_seizure_ids + top_pnes_ids
print(f"Target IDs locked: {final_target_ids}")


# -------------------------------------------------
# 4. The Renderer Loop
# -------------------------------------------------
print("Initializing Fast SHAP Explainer...")
background_features, _, _ = next(iter(train_loader))
# SHAP Speed Optimization by select 30 ones
small_background = background_features[:30].to(device) 
explainer = shap.GradientExplainer(model, small_background)

global_idx = 0
loop = 0
found_count = 0

print("Starting evaluation scan...")
for batch_features, batch_labels, _ in test_loader:
    
    if found_count >= len(final_target_ids):
        print("All target figures generated. Exiting loop.")
        break

    for i in range(batch_features.size(0)):
        sample_idx = global_idx
        
        # ONLY DO HEAVY WORK IF THIS IS A TARGET ID
        if sample_idx in final_target_ids:
            print(f"\nFound Target ID {sample_idx}! Generating Figures...")
            
            single_feature = batch_features[i:i + 1].to(device)
            true_label = batch_labels[i].item()

            with torch.no_grad():
                outputs_no_grads = model(single_feature)
                predicted_label = outputs_no_grads.argmax().item()

            # Generate Grad-CAMs for both True and Predicted classes respectively
            generate_and_save_gradcam(single_feature, true_label, "True", true_label, predicted_label, sample_idx, loop, mode='true') # True class
            generate_and_save_gradcam(single_feature, predicted_label, "Predicted", true_label, predicted_label, sample_idx, loop, mode='pred') # Predicted class

            # Generate SHAP
            shap_values_4d = explainer.shap_values(single_feature) 
            class_name = "PNES" if true_label == 0 else "Seizure"
            shap_2d = shap_values_4d[0, :, :, true_label]

            importance_over_time = shap_2d.mean(axis=0) 
            importance_per_channel = shap_2d.mean(axis=1) 

            # Plot 1: SHAP Heatmap 
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(15, 10))
            im = ax_heatmap.imshow(shap_2d, cmap='bwr', aspect='auto')
            ax_heatmap.set_title(f"Spatio-Temporal Importance for {class_name}\n(True: {true_label}, Pred: {predicted_label})")
            ax_heatmap.set_ylabel("EEG Channels")
            ax_heatmap.set_xlabel("Time Points")
            ax_heatmap.set_yticks(np.arange(len(eeg_ch_names)))
            ax_heatmap.set_yticklabels(eeg_ch_names, fontsize=8)
            fig_heatmap.colorbar(im, ax_heatmap, label="SHAP Value")
            plt.tight_layout()
            plt.savefig(f"{base_out_dir}/shap_analysis_results/sample_{sample_idx}_{class_name}_heatmap.png", dpi=300)
            plt.close(fig_heatmap)

            # Plot 2: Time Importance
            fig_temporal, ax_temporal = plt.subplots(figsize=(15, 5))
            ax_temporal.plot(importance_over_time)
            ax_temporal.axhline(0, color='black', linestyle='--', linewidth=0.8)
            ax_temporal.set_title(f"Aggregate Importance Over Time for {class_name}")
            ax_temporal.set_xlabel("Time Points")
            ax_temporal.set_ylabel("Mean SHAP Value")
            plt.tight_layout()
            plt.savefig(f"{base_out_dir}/shap_analysis_results/sample_{sample_idx}_{class_name}_temporal.png", dpi=300)
            plt.close(fig_temporal)

            # Plot 3: Channel Importance
            fig_channel, ax_channel = plt.subplots(figsize=(15, 8))
            ax_channel.bar(eeg_ch_names, importance_per_channel)
            ax_channel.axhline(0, color='black', linestyle='--', linewidth=0.8)
            ax_channel.set_title(f"Aggregate Importance Per Channel for {class_name}")
            ax_channel.set_ylabel("Mean SHAP Value")
            ax_channel.set_xlabel("Channels Name")
            ax_channel.tick_params(axis='x', rotation=90)
            plt.tight_layout()
            plt.savefig(f"{base_out_dir}/shap_analysis_results/sample_{sample_idx}_{class_name}_channel.png", dpi=300)
            plt.close(fig_channel)
            
            found_count += 1
            
        global_idx += 1
    loop += 1

print("\nVisualization successfully finished! Check the 'Visualizations' folder.")
