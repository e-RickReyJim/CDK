"""
Utility Functions for CKD Prediction
====================================
Helper functions for eGFR calculation, stage classification, and visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from io import BytesIO
import base64

matplotlib.use('Agg')  # Non-interactive backend


def calculate_egfr(serum_creatinine, age):
    """
    Calculate eGFR using MDRD equation.
    
    Parameters:
    -----------
    serum_creatinine : float
        Serum creatinine level in mg/dL
    age : float
        Patient age in years
    
    Returns:
    --------
    float or None
        Estimated GFR value or None if invalid inputs
    """
    if serum_creatinine is None or age is None or serum_creatinine <= 0 or age <= 0:
        return None
    return 175 * (serum_creatinine ** -1.154) * (age ** -0.203)


def get_stage_from_egfr(egfr):
    """
    Classify CKD stage based on eGFR value.
    
    Parameters:
    -----------
    egfr : float
        Estimated GFR value
    
    Returns:
    --------
    int or None
        CKD stage (1-5) or None if invalid input
    """
    if egfr is None or pd.isna(egfr):
        return None
    if egfr >= 90:
        return 1
    elif egfr >= 60:
        return 2
    elif egfr >= 30:
        return 3
    elif egfr >= 15:
        return 4
    else:
        return 5


def get_stage_info(stage):
    """Get detailed information about a CKD stage"""
    stage_info = {
        1: {"name": "Stage 1", "severity": "Normal/High", "color": "#28a745", "description": "Kidney damage with normal function"},
        2: {"name": "Stage 2", "severity": "Mild", "color": "#90ee90", "description": "Mild reduction in kidney function"},
        3: {"name": "Stage 3", "severity": "Moderate", "color": "#ffc107", "description": "Moderate reduction in kidney function"},
        4: {"name": "Stage 4", "severity": "Severe", "color": "#ff8c00", "description": "Severe reduction in kidney function"},
        5: {"name": "Stage 5", "severity": "Kidney Failure", "color": "#dc3545", "description": "Kidney failure - dialysis may be needed"}
    }
    return stage_info.get(stage, {"name": "Unknown", "severity": "Unknown", "color": "#6c757d", "description": "Unknown stage"})


def create_stage_probability_plot(probabilities):
    """
    Create a bar chart showing probabilities for all 5 CKD stages.
    
    Parameters:
    -----------
    probabilities : dict
        Dictionary with stage numbers as keys and probabilities as values
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stages = list(range(1, 6))
    probs = [probabilities.get(i, 0) for i in stages]
    colors = ['#28a745', '#90ee90', '#ffc107', '#ff8c00', '#dc3545']
    
    bars = ax.bar(stages, probs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.1%}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('CKD Stage', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax.set_title('Predicted Stage Probability Distribution', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(stages)
    ax.set_xticklabels([f'Stage {s}' for s in stages])
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig


def create_egfr_gauge(egfr, predicted_stage):
    """
    Create a horizontal gauge showing eGFR position across stage boundaries.
    
    Parameters:
    -----------
    egfr : float
        Estimated GFR value
    predicted_stage : int
        Predicted CKD stage (1-5)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # Stage boundaries and colors
    boundaries = [0, 15, 30, 60, 90, 120]
    colors = ['#dc3545', '#ff8c00', '#ffc107', '#90ee90', '#28a745']
    stage_labels = ['Stage 5\n(<15)', 'Stage 4\n(15-29)', 'Stage 3\n(30-59)', 
                    'Stage 2\n(60-89)', 'Stage 1\n(≥90)']
    
    # Draw colored segments
    for i in range(len(colors)):
        ax.barh(0, boundaries[i+1] - boundaries[i], left=boundaries[i], 
                height=0.5, color=colors[i], alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add stage labels
    for i, label in enumerate(stage_labels):
        mid_point = (boundaries[i] + boundaries[i+1]) / 2
        ax.text(mid_point, 0, label, ha='center', va='center', 
                fontweight='bold', fontsize=10, color='black')
    
    # Plot patient's eGFR position
    egfr_display = min(egfr, 120)  # Cap display at 120
    ax.plot(egfr_display, 0, 'v', markersize=20, color='blue', 
            markeredgecolor='black', markeredgewidth=2, zorder=5)
    ax.text(egfr_display, 0.35, f'Your eGFR\n{egfr:.1f}', 
            ha='center', va='bottom', fontweight='bold', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue', linewidth=2))
    
    ax.set_xlim(0, 120)
    ax.set_ylim(-0.4, 0.6)
    ax.set_xlabel('eGFR (mL/min/1.73m²)', fontsize=12, fontweight='bold')
    ax.set_title('Kidney Function Assessment - eGFR Gauge', fontsize=14, fontweight='bold', pad=20)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    return fig


def create_model_agreement_plot(model_predictions):
    """
    Create a heatmap showing which models predicted which stage.
    
    Parameters:
    -----------
    model_predictions : dict
        Dictionary with model names as keys and predicted stages as values
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(model_predictions.keys())
    stages = list(range(1, 6))
    
    # Create matrix: 1 if model predicted that stage, 0 otherwise
    matrix = np.zeros((len(models), len(stages)))
    for i, model in enumerate(models):
        stage = model_predictions[model]
        if stage in stages:
            matrix[i, stage - 1] = 1
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(stages)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([f'Stage {s}' for s in stages])
    ax.set_yticklabels(models)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(stages)):
            text = ax.text(j, i, '✓' if matrix[i, j] == 1 else '',
                          ha="center", va="center", color="black", 
                          fontsize=16, fontweight='bold')
    
    ax.set_title('Model Predictions Agreement', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Stage', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Prediction', rotation=270, labelpad=20, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_info_plot():
    """
    Create a static informational plot explaining CKD stages.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Title
    fig.suptitle('Chronic Kidney Disease (CKD) Stages Reference', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Stage information
    stages = [
        {'stage': 1, 'egfr': '≥ 90', 'severity': 'Normal/High', 'color': '#28a745',
         'description': 'Kidney damage with normal or high GFR'},
        {'stage': 2, 'egfr': '60-89', 'severity': 'Mild', 'color': '#90ee90',
         'description': 'Mild reduction in kidney function'},
        {'stage': 3, 'egfr': '30-59', 'severity': 'Moderate', 'color': '#ffc107',
         'description': 'Moderate reduction in kidney function'},
        {'stage': 4, 'egfr': '15-29', 'severity': 'Severe', 'color': '#ff8c00',
         'description': 'Severe reduction in kidney function'},
        {'stage': 5, 'egfr': '< 15', 'severity': 'Kidney Failure', 'color': '#dc3545',
         'description': 'Kidney failure - dialysis or transplant needed'}
    ]
    
    y_start = 0.85
    y_step = 0.15
    
    for i, stage_data in enumerate(stages):
        y_pos = y_start - (i * y_step)
        
        # Stage box
        rect = plt.Rectangle((0.05, y_pos - 0.06), 0.9, 0.12, 
                            facecolor=stage_data['color'], 
                            edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        
        # Text content
        stage_text = f"Stage {stage_data['stage']}"
        egfr_text = f"eGFR: {stage_data['egfr']} mL/min/1.73m²"
        severity_text = f"{stage_data['severity']}"
        desc_text = stage_data['description']
        
        ax.text(0.08, y_pos + 0.03, stage_text, fontsize=14, fontweight='bold', va='center')
        ax.text(0.25, y_pos + 0.03, egfr_text, fontsize=11, va='center')
        ax.text(0.50, y_pos + 0.03, severity_text, fontsize=11, fontweight='bold', va='center')
        ax.text(0.08, y_pos - 0.02, desc_text, fontsize=9, va='center', style='italic')
    
    # Footer
    ax.text(0.5, 0.02, 
            'eGFR calculated using MDRD equation: 175 × (SerumCreatinine)^-1.154 × (Age)^-0.203',
            ha='center', fontsize=9, style='italic', color='gray')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig
