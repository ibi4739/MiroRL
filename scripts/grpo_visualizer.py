# Copyright 2025 MiroMind Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python3
"""
GRPO Rollout Visualizer

A beautiful interactive visualizer for GRPO (Generalized Preference Optimization) rollouts.
Supports hierarchical navigation: Projects → Epochs → Input Groups → Individual Rollouts
"""

import json
import os
import sys
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import defaultdict
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="GRPO Rollout Visualizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Global font size increases */
    .main .block-container {
        font-size: 2.2rem;
    }
    
    /* Streamlit elements font size increases */
    .stMarkdown, .stText, .stCaption {
        font-size: 2.1rem !important;
    }
    
    .stMetric {
        font-size: 2.2rem !important;
    }
    
    .stMetric > div {
        font-size: 2.2rem !important;
    }
    
    .stMetric [data-testid="metric-label"] {
        font-size: 2.1rem !important;
    }
    
    .stMetric [data-testid="metric-value"] {
        font-size: 2rem !important;
    }
    
    .stButton > button {
        font-size: 2rem !important;
        padding: 1rem 2rem !important;
        height: auto !important;
        min-height: 3rem !important;
    }
    
    .stSelectbox > div > div {
        font-size: 2rem !important;
        min-height: 3rem !important;
    }
    
    .stTextInput > div > div > input {
        font-size: 2rem !important;
        line-height: 1.5 !important;
        padding: 1rem !important;
        min-height: 3rem !important;
    }
    
    .stTextArea > div > div > textarea {
        font-size: 2rem !important;
        line-height: 1.6 !important;
        font-family: 'Courier New', monospace !important;
        padding: 1rem !important;
    }
    
    .stExpander > div > div > div {
        font-size: 2rem !important;
        padding: 1rem !important;
    }
    
    .stExpander > div > div > div > div {
        padding: 0.5rem !important;
    }
    
    .main-header {
        font-size: 4.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: bold;
    }
    
    .quick-access-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        border: 2px solid #dee2e6;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .quick-access-title {
        font-size: 2.5rem;
        color: #495057;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    .quick-access-description {
        font-size: 1.8rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
        font-style: italic;
    }
    
    /* Suggestion buttons styling */
    .suggestion-button {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%) !important;
        color: #424242 !important;
        border: 1px solid #90caf9 !important;
        font-size: 1.6rem !important;
        padding: 0.8rem 1.5rem !important;
        border-radius: 12px !important;
        margin: 0.3rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    .suggestion-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        background: linear-gradient(135deg, #bbdefb 0%, #e1bee7 100%) !important;
    }
    
    /* History buttons styling */
    .history-button {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%) !important;
        color: #bf360c !important;
        border: 1px solid #ffb74d !important;
        font-size: 1.6rem !important;
        padding: 0.8rem 1.5rem !important;
        border-radius: 12px !important;
        margin: 0.3rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    .history-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        background: linear-gradient(135deg, #ffcc02 0%, #ffb300 100%) !important;
    }
    
    /* Match buttons styling */
    .match-button {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%) !important;
        color: #1b5e20 !important;
        border: 1px solid #4caf50 !important;
        font-size: 1.6rem !important;
        padding: 0.8rem 1.5rem !important;
        border-radius: 12px !important;
        margin: 0.3rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    .match-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        background: linear-gradient(135deg, #a5d6a7 0%, #81c784 100%) !important;
    }
    
    /* Help text styling */
    .help-text {
        background: #f8f9fa;
        color: #6c757d;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
        font-size: 1.6rem;
    }
    
    .project-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        color: white;
        text-align: center;
        cursor: pointer;
        transition: transform 0.3s ease;
        font-size: 2.3rem;
    }
    
    .project-card:hover {
        transform: translateY(-8px);
    }
    
    .epoch-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        text-align: center;
        font-size: 2.2rem;
    }
    
    .group-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        font-size: 2.2rem;
    }
    
    .rollout-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        font-size: 2.2rem;
    }
    
    .better-rollout {
        border-left: 6px solid #28a745;
    }
    
    .worse-rollout {
        border-left: 6px solid #dc3545;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        font-size: 2.2rem;
    }
    
    .navigation-breadcrumb {
        background: #e9ecef;
        padding: 1rem 1.5rem;
        border-radius: 25px;
        margin-bottom: 1.5rem;
        font-size: 2.1rem;
    }
    
    /* Increase heading sizes */
    h1 {
        font-size: 2.5rem !important;
    }
    
    h2 {
        font-size: 2rem !important;
    }
    
    h3 {
        font-size: 2.7rem !important;
    }
    
    h4 {
        font-size: 2.4rem !important;
    }
    
    /* Sidebar font sizes */
    .css-1d391kg {
        font-size: 2.1rem !important;
        padding: 1rem !important;
    }
    
    .sidebar .sidebar-content {
        font-size: 2.1rem !important;
        padding: 1rem !important;
    }
    
    /* Adjust sidebar width for larger fonts */
    .css-1d391kg {
        width: 400px !important;
    }
    
    /* Adjust column spacing */
    .css-1r6slb0 {
        gap: 2rem !important;
    }
    
    /* Increase spacing between elements */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Adjust warning and info boxes */
    .stAlert {
        font-size: 1.8rem !important;
        padding: 1.5rem !important;
    }
    
    /* Adjust success messages */
    .stSuccess {
        font-size: 1.8rem !important;
        padding: 1.5rem !important;
    }
    
    /* Adjust error messages */
    .stError {
        font-size: 1.8rem !important;
        padding: 1.5rem !important;
    }
    
    /* Conversation display styles */
    .conversation-display {
        font-family: 'Courier New', monospace;
        font-size: 1.8rem;
        line-height: 1.6;
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        max-height: 2000px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    .user-message {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
    }
    
    .system-message {
        background-color: #fff3e0;
        color: #ef6c00;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
    }
    
    .role-label {
        font-weight: bold;
        text-transform: uppercase;
        font-size: 0.9em;
        margin-bottom: 0.5rem;
        opacity: 0.8;
    }
    
    /* Force large font sizes for st.code() components */
    .stCode {
        font-size: 2.5rem !important;
        line-height: 1.6 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stCode > div {
        font-size: 2.5rem !important;
        line-height: 1.6 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stCode pre {
        font-size: 2.5rem !important;
        line-height: 1.6 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stCode code {
        font-size: 2.5rem !important;
        line-height: 1.6 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    /* Target the specific code block container */
    [data-testid="stCode"] {
        font-size: 2.5rem !important;
        line-height: 1.6 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    [data-testid="stCode"] > div {
        font-size: 2.5rem !important;
        line-height: 1.6 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    [data-testid="stCode"] pre {
        font-size: 2.5rem !important;
        line-height: 1.6 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    [data-testid="stCode"] code {
        font-size: 2.5rem !important;
        line-height: 1.6 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    /* Force large font sizes for st.write() components */
    .stWrite {
        font-size: 2.5rem !important;
        line-height: 1.6 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stWrite > div {
        font-size: 2.5rem !important;
        line-height: 1.6 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stWrite p {
        font-size: 2.5rem !important;
        line-height: 1.6 !important;
        font-family: 'Courier New', monospace !important;
        white-space: pre-wrap !important;
    }
    
    /* Target markdown container used by st.write() */
    [data-testid="stMarkdownContainer"] {
        font-size: 2.5rem !important;
        line-height: 1.6 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    [data-testid="stMarkdownContainer"] p {
        font-size: 2.5rem !important;
        line-height: 1.6 !important;
        font-family: 'Courier New', monospace !important;
        white-space: pre-wrap !important;
    }
    
    [data-testid="stMarkdownContainer"] div {
        font-size: 2.5rem !important;
        line-height: 1.6 !important;
        font-family: 'Courier New', monospace !important;
        white-space: pre-wrap !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Utility function for sorting epochs
def sort_epoch_key(epoch_name: str):
    """Extract numeric part for proper epoch sorting."""
    # Try to extract number from epoch name
    match = re.search(r'(\d+)', epoch_name)
    if match:
        return (int(match.group(1)), epoch_name)
    else:
        return (float('inf'), epoch_name)  # Non-numeric epochs go to end

# Efficient caching functions using streamlit's native caching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_project_list(work_folder: str) -> List[str]:
    """Get list of project folders."""
    work_path = Path(work_folder)
    if not work_path.exists():
        return []
    return [f.name for f in work_path.iterdir() if f.is_dir()]

@st.cache_data(ttl=3600)
def get_epoch_list(work_folder: str, project_name: str) -> List[str]:
    """Get list of epochs for a project."""
    project_path = Path(work_folder) / project_name
    if not project_path.exists():
        return []
    return [f.stem for f in project_path.glob("*.jsonl")]

@st.cache_data(ttl=3600)
def get_project_stats(work_folder: str, project_name: str) -> Dict:
    """Get lightweight project statistics only."""
    project_path = Path(work_folder) / project_name
    if not project_path.exists():
        return {}
    
    epochs = []
    total_rollouts = 0
    total_groups = 0
    
    all_scores = []
    all_accuracies = []
    all_tool_format_rewards = []
    all_combined_rewards = []
    
    for jsonl_file in project_path.glob("*.jsonl"):
        epoch_stats = get_epoch_stats(work_folder, project_name, jsonl_file.stem)
        epochs.append(jsonl_file.stem)
        total_rollouts += epoch_stats.get('total_rollouts', 0)
        total_groups += epoch_stats.get('total_groups', 0)
        
        # Collect all scores for project-level averages
        all_scores.append(epoch_stats.get('average_score', 0))
        all_accuracies.append(epoch_stats.get('average_accuracy', 0))
        all_tool_format_rewards.append(epoch_stats.get('average_tool_format_reward', 0))
        all_combined_rewards.append(epoch_stats.get('average_combined_reward', 0))
    
    return {
        'epochs': epochs,
        'total_rollouts': total_rollouts,
        'total_groups': total_groups,
        'average_score': float(np.mean(all_scores)) if all_scores else 0.0,
        'average_accuracy': float(np.mean(all_accuracies)) if all_accuracies else 0.0,
        'average_tool_format_reward': float(np.mean(all_tool_format_rewards)) if all_tool_format_rewards else 0.0,
        'average_combined_reward': float(np.mean(all_combined_rewards)) if all_combined_rewards else 0.0
    }

@st.cache_data(ttl=3600)
def get_epoch_raw_data(work_folder: str, project_name: str, epoch_name: str) -> Dict:
    """Get raw epoch data grouped by input - shared by both get_epoch_stats and get_epoch_groups."""
    jsonl_file = Path(work_folder) / project_name / f"{epoch_name}.jsonl"
    if not jsonl_file.exists():
        return {}
    
    # Process only statistics, not full rollout data
    grouped_stats = defaultdict(list)
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rollout = json.loads(line)
                    input_text = rollout.get('input', '')
                    output_text = rollout.get('output', '')
                    if "turn_count" not in rollout.keys():
                        assistant_count = output_text.count("\nassistant\n")
                        # Split text by \nassistant\n and \nuser\n to get assistant responses
                        parts = output_text.split("\nassistant\n")
                        
                        # Get content before first user prompt
                        pre_user = parts[0].split("\nuser\n")[0] if parts else ""
                        
                        # Get assistant responses that are between assistant and user tags
                        assistant_responses = []
                        if pre_user:
                            assistant_responses.append(pre_user)
                            
                        for i in range(1, len(parts)):
                            if "\nuser\n" in parts[i]:
                                assistant_responses.append(parts[i].split("\nuser\n")[0])
                            else:
                                assistant_responses.append(parts[i])
                        assistant_count = assistant_count + 1
                    else:
                        assistant_count = rollout.get('turn_count', 0)
                    
                    output_tokens = len(output_text) // 4

                    # Store only essential metrics, not full rollout
                    grouped_stats[input_text].append({
                        'score': rollout.get('score', 0),
                        'accuracy_reward': rollout.get('accuracy_reward', 0)/0.9,
                        'tool_format_reward': rollout.get('tool_format_reward', 0),
                        'combined_reward': rollout.get('combined_reward', 0),
                        'turn_count': assistant_count,
                        'output_tokens': output_tokens,
                        'average_responses_tokens': sum(len(response)//4 for response in assistant_responses) / len(assistant_responses) if assistant_responses else 0
                    })
    except Exception as e:
        st.error(f"Error reading {jsonl_file}: {e}")
        return {}
    
    return grouped_stats

@st.cache_data(ttl=3600)
def get_epoch_stats(work_folder: str, project_name: str, epoch_name: str) -> Dict:
    """Get lightweight epoch statistics only - now reuses get_epoch_raw_data."""
    grouped_stats = get_epoch_raw_data(work_folder, project_name, epoch_name)
    
    if not grouped_stats:
        return {}
    
    # Calculate statistics
    total_rollouts = sum(len(rollouts) for rollouts in grouped_stats.values())
    total_groups = len(grouped_stats)
    
    all_scores = []
    all_accuracies = []
    all_tool_format_rewards = []
    all_combined_rewards = []
    all_turn_counts = []
    all_output_tokens = []
    all_average_responses_tokens = []
    for rollouts in grouped_stats.values():
        for rollout in rollouts:
            all_scores.append(rollout['score'])
            all_accuracies.append(rollout['accuracy_reward'])
            all_tool_format_rewards.append(rollout['tool_format_reward'])
            all_combined_rewards.append(rollout['combined_reward'])
            all_turn_counts.append(rollout['turn_count'])
            all_output_tokens.append(rollout['output_tokens'])
            all_average_responses_tokens.append(rollout['average_responses_tokens'])
    return {
        'total_rollouts': total_rollouts,
        'total_groups': total_groups,
        'average_score': float(np.mean(all_scores)) if all_scores else 0.0,
        'average_accuracy': float(np.mean(all_accuracies)) if all_accuracies else 0.0,
        'average_tool_format_reward': float(np.mean(all_tool_format_rewards)) if all_tool_format_rewards else 0.0,
        'average_combined_reward': float(np.mean(all_combined_rewards)) if all_combined_rewards else 0.0,
        'average_turn_count': float(np.mean(all_turn_counts)) if all_turn_counts else 0.0,
        'average_output_tokens': float(np.mean(all_output_tokens)) if all_output_tokens else 0.0,
        'average_responses_tokens_per_turn': float(np.mean(all_average_responses_tokens)) if all_average_responses_tokens else 0.0
    }

@st.cache_data(ttl=3600)
def get_epoch_groups(work_folder: str, project_name: str, epoch_name: str) -> Dict:
    """Get group information for an epoch - now reuses get_epoch_raw_data."""
    grouped_rollouts = get_epoch_raw_data(work_folder, project_name, epoch_name)
    
    if not grouped_rollouts:
        return {}
    
    processed_groups = {}
    for input_text, rollouts in grouped_rollouts.items():
        scores = [r['score'] for r in rollouts]
        accuracies = [r['accuracy_reward'] for r in rollouts]
        tool_format_rewards = [r['tool_format_reward'] for r in rollouts]
        combined_rewards = [r['combined_reward'] for r in rollouts]
        turn_counts = [r['turn_count'] for r in rollouts]
        output_tokens = [r['output_tokens'] for r in rollouts]
        responses_tokens = [r['average_responses_tokens'] for r in rollouts]
        
        # Extract user prompt
        user_prompt = input_text.split('\nuser\n', 1)[1].strip() if '\nuser\n' in input_text else input_text[:200]
        
        processed_groups[input_text] = {
            'user_prompt': user_prompt,
            'count': len(rollouts),
            'average_score': float(np.mean(scores)) if scores else 0.0,
            'average_accuracy': float(np.mean(accuracies)) if accuracies else 0.0,
            'average_tool_format_reward': float(np.mean(tool_format_rewards)) if tool_format_rewards else 0.0,
            'average_combined_reward': float(np.mean(combined_rewards)) if combined_rewards else 0.0,
            'average_turn_count': float(np.mean(turn_counts)) if turn_counts else 0.0,
            'average_output_tokens': float(np.mean(output_tokens)) if output_tokens else 0.0,
            'score_std': float(np.std(scores)) if scores else 0.0,
            'accuracy_std': float(np.std(accuracies)) if accuracies else 0.0,
            'tool_format_reward_std': float(np.std(tool_format_rewards)) if tool_format_rewards else 0.0,
            'combined_reward_std': float(np.std(combined_rewards)) if combined_rewards else 0.0,
            'turn_count_std': float(np.std(turn_counts)) if turn_counts else 0.0,
            'output_tokens_std': float(np.std(output_tokens)) if output_tokens else 0.0,
            'average_responses_tokens_per_turn': float(np.mean(responses_tokens)) if responses_tokens else 0.0
        }
    
    return processed_groups

def get_group_rollouts(work_folder: str, project_name: str, epoch_name: str, group_input: str) -> List[Dict]:
    """Get full rollout data for a specific group - loaded on demand."""
    jsonl_file = Path(work_folder) / project_name / f"{epoch_name}.jsonl"
    if not jsonl_file.exists():
        return []
    
    rollouts = []
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rollout = json.loads(line)
                    if rollout.get('input', '') == group_input:
                        rollouts.append(rollout)
    except Exception as e:
        st.error(f"Error reading {jsonl_file}: {e}")
    
    return rollouts

class GRPOVisualizer:
    def __init__(self, work_folder: str):
        self.work_folder = Path(work_folder)
    
    def display_conversation(self, text: str, key_prefix: str, is_ground_truth: bool = False):
        """Display conversation text with color-coded user/assistant messages using Streamlit components."""
        if not text:
            st.info("No content available")
            return
        
        # Split by role markers
        if is_ground_truth:
            current_text = text
        else:
            current_text = "\nassistant\n" + text
        
        # Find all role markers and their positions
        pattern = r'\n(user|assistant|system)\n'
        matches = list(re.finditer(pattern, current_text, re.IGNORECASE))
        
        if not matches:
            # No role markers found, display as system message
            if not is_ground_truth:
                st.warning("Content (no role markers found)")
            st.text_area("", value=text, height=300, key=f"{key_prefix}_no_roles", disabled=True)
            return
        
        # Process each segment
        for i, match in enumerate(matches):
            # Get the role and find the content for this role
            role = match.group(1).lower()
            content_start = match.end()
            
            # Find the end of this role's content (next role marker or end of text)
            if i + 1 < len(matches):
                content_end = matches[i + 1].start()
            else:
                content_end = len(current_text)
            
            content = current_text[content_start:content_end].strip()
            
            if content:
                # Determine display style based on role
                if role == 'user':
                    st.markdown("---")
                    st.markdown("### 👤 USER")
                    st.write(content)
                        
                elif role == 'assistant':
                    st.markdown("---")
                    st.markdown("### 🤖 ASSISTANT")
                    st.write(content)
                        
                else:  # system
                    st.markdown("### ⚙️ SYSTEM") 
                    st.write(content)
    

    
    def render_projects_view(self):
        """Render the main projects view."""
        st.markdown('<div class="main-header">🎯 GRPO Rollout Visualizer</div>', unsafe_allow_html=True)
        
        # Quick project navigation section
        st.markdown("""
        <div class="quick-access-section">
            <div class="quick-access-title">🚀 Quick Project Access</div>
            <div class="quick-access-description">Enter a project name to jump directly to it (useful when you have many projects)</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add help info
        with st.expander("ℹ️ How to use Quick Access", expanded=False):
            st.markdown("""
            <div class="help-text">
                <strong>💡 Tips:</strong><br>
                • Type any part of a project name to see matching suggestions<br>
                • Press Enter after typing to navigate directly<br>
                • Use the 'Go' button to navigate to the entered project<br>
                • Recent projects will appear as 🕒 buttons for quick access<br>
                • Project suggestions show as 📂 buttons below<br>
                • Clear the input field with the 🧹 Clear button
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            quick_project = st.text_input(
                "Project Name:",
                placeholder="Enter project name to jump directly...",
                key="quick_project_input",
                help="Type the exact project name and press Enter or click 'Go' to navigate directly to that project",
                label_visibility="collapsed"
            )
        
        with col2:
            # Add some spacing to align with the input field
            st.markdown("<br>", unsafe_allow_html=True)
            go_button = st.button("🎯 Go", key="quick_project_go", use_container_width=True)
        
        with col3:
            # Add some spacing to align with the input field
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🧹 Clear", key="clear_quick_project", use_container_width=True):
                st.session_state.quick_project_input = ""
                st.rerun()
            
        # Handle Go button click
        if go_button:
            if quick_project.strip():
                # Check if project exists
                project_path = Path(self.work_folder) / quick_project.strip()
                if project_path.exists() and project_path.is_dir():
                    # Check if it has any .jsonl files (valid project)
                    if list(project_path.glob("*.jsonl")):
                        # Add to project history
                        if 'project_history' not in st.session_state:
                            st.session_state.project_history = []
                        
                        project_name = quick_project.strip()
                        if project_name in st.session_state.project_history:
                            st.session_state.project_history.remove(project_name)
                        st.session_state.project_history.insert(0, project_name)
                        st.session_state.project_history = st.session_state.project_history[:5]  # Keep only 5 recent
                        
                        st.session_state.selected_project = project_name
                        st.session_state.selected_epoch = None
                        st.session_state.selected_group = None
                        st.success(f"🎉 Navigating to project: {project_name}")
                        st.rerun()
                    else:
                        st.error(f"❌ Project '{quick_project.strip()}' exists but contains no .jsonl files!")
                else:
                    st.error(f"❌ Project '{quick_project.strip()}' not found!")
            else:
                st.warning("⚠️ Please enter a project name!")
        
        # Auto-complete suggestions
        if quick_project and len(quick_project.strip()) > 0:
            try:
                all_projects = get_project_list(str(self.work_folder))
                matching_projects = [p for p in all_projects if quick_project.strip().lower() in p.lower()]
                
                if matching_projects and quick_project.strip().lower() not in [p.lower() for p in matching_projects]:
                    st.markdown("**🔍 Matching projects:**")
                    match_cols = st.columns(min(len(matching_projects), 3))
                    for idx, project in enumerate(matching_projects[:6]):  # Show max 6 matches
                        with match_cols[idx % 3]:
                            if st.button(f"🔍 {project}", key=f"match_{project}", use_container_width=True):
                                # Add to project history
                                if 'project_history' not in st.session_state:
                                    st.session_state.project_history = []
                                
                                if project in st.session_state.project_history:
                                    st.session_state.project_history.remove(project)
                                st.session_state.project_history.insert(0, project)
                                st.session_state.project_history = st.session_state.project_history[:5]
                                
                                st.session_state.selected_project = project
                                st.session_state.selected_epoch = None
                                st.session_state.selected_group = None
                                st.rerun()
            except Exception as e:
                # If there's an error getting matches, silently continue
                pass
        
        # Also handle Enter key press
        if quick_project and quick_project.strip():
            if st.session_state.get('last_quick_project') != quick_project.strip():
                st.session_state.last_quick_project = quick_project.strip()
                # Auto-navigate when Enter is pressed (detected by input change)
                project_path = Path(self.work_folder) / quick_project.strip()
                if project_path.exists() and project_path.is_dir() and list(project_path.glob("*.jsonl")):
                    # Add to project history
                    if 'project_history' not in st.session_state:
                        st.session_state.project_history = []
                    
                    project_name = quick_project.strip()
                    if project_name in st.session_state.project_history:
                        st.session_state.project_history.remove(project_name)
                    st.session_state.project_history.insert(0, project_name)
                    st.session_state.project_history = st.session_state.project_history[:5]
                    
                    st.session_state.selected_project = project_name
                    st.session_state.selected_epoch = None
                    st.session_state.selected_group = None
                    st.rerun()
        
        # Show project history
        if 'project_history' in st.session_state and st.session_state.project_history:
            st.markdown("**🕒 Recent Projects:**")
            history_cols = st.columns(min(len(st.session_state.project_history), 3))
            for idx, project in enumerate(st.session_state.project_history):
                with history_cols[idx % 3]:
                    if st.button(f"🕒 {project}", key=f"history_{project}", use_container_width=True):
                        st.session_state.selected_project = project
                        st.session_state.selected_epoch = None
                        st.session_state.selected_group = None
                        st.rerun()
        
        # Show project suggestions
        try:
            # Get a sample of projects for quick access (first 6 projects)
            sample_projects = get_project_list(str(self.work_folder))[:6]
            if sample_projects:
                st.markdown("**📝 Quick suggestions:**")
                suggestion_cols = st.columns(min(len(sample_projects), 3))
                for idx, project in enumerate(sample_projects):
                    with suggestion_cols[idx % 3]:
                        if st.button(f"📂 {project}", key=f"suggestion_{project}", use_container_width=True):
                            # Add to project history
                            if 'project_history' not in st.session_state:
                                st.session_state.project_history = []
                            if project in st.session_state.project_history:
                                st.session_state.project_history.remove(project)
                            st.session_state.project_history.insert(0, project)
                            st.session_state.project_history = st.session_state.project_history[:5]
                            
                            st.session_state.selected_project = project
                            st.session_state.selected_epoch = None
                            st.session_state.selected_group = None
                            st.rerun()
        except Exception as e:
            # If there's an error getting project suggestions, silently continue
            pass
        
        st.markdown("---")
        
        # Regular project loading section
        st.markdown("### 📁 All Available Projects")
        
        projects = get_project_list(str(self.work_folder))
        
        if not projects:
            st.error("No projects found in the work folder!")
            return
        
        # Show project statistics
        st.markdown(f"""
        <div class="metric-card">
            <strong>📊 Project Statistics:</strong> Found {len(projects)} projects in total
        </div>
        """, unsafe_allow_html=True)
        
        if len(projects) > 10:
            st.markdown("*Loading all projects... (this may take a moment since you have many projects)*")
        
        # Create columns for project cards
        cols = st.columns(min(2, len(projects)))
        
        for idx, project_name in enumerate(projects):
            with cols[idx % 2]:
                # Get project statistics
                project_stats = get_project_stats(str(self.work_folder), project_name)
                
                if st.button(f"📊 {project_name}", key=f"project_{project_name}", use_container_width=True):
                    # Add to project history
                    if 'project_history' not in st.session_state:
                        st.session_state.project_history = []
                    
                    if project_name in st.session_state.project_history:
                        st.session_state.project_history.remove(project_name)
                    st.session_state.project_history.insert(0, project_name)
                    st.session_state.project_history = st.session_state.project_history[:5]
                    
                    st.session_state.selected_project = project_name
                    st.session_state.selected_epoch = None
                    st.session_state.selected_group = None
                    st.rerun()
                
                # Show comprehensive project stats
                st.markdown(f"""
                <div class="metric-card">
                    <strong>📊 Basic Stats:</strong><br>
                    Epochs: {len(project_stats.get('epochs', []))} | Rollouts: {project_stats.get('total_rollouts', 0)} | Samples: {project_stats.get('total_groups', 0)}<br><br>
                    <strong>🎯 Performance Metrics:</strong><br>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px; margin-top: 10px;">
                        <div>Score: <strong>{project_stats.get('average_score', 0):.3f}</strong></div>
                        <div>Accuracy: <strong>{project_stats.get('average_accuracy', 0):.3f}</strong></div>
                        <div>Tool Format: <strong>{project_stats.get('average_tool_format_reward', 0):.3f}</strong></div>
                        <div>Reward: <strong>{project_stats.get('average_reward', 0):.3f}</strong></div>
                        <div colspan="2">Combined: <strong>{project_stats.get('average_combined_reward', 0):.3f}</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add performance trend visualization
                epoch_names = project_stats.get('epochs', [])
                if epoch_names:
                    epoch_scores = []
                    epoch_accuracies = []
                    
                    for epoch_name in epoch_names:
                        epoch_stats = get_epoch_stats(str(self.work_folder), project_name, epoch_name)
                        epoch_scores.append(epoch_stats.get('average_score', 0))
                        epoch_accuracies.append(epoch_stats.get('average_accuracy', 0))
                    
                    # Create mini chart for project overview
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(len(epoch_names))),
                        y=epoch_scores,
                        mode='lines+markers',
                        name='Score',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=4)
                    ))
                    fig.add_trace(go.Scatter(
                        x=list(range(len(epoch_names))),
                        y=epoch_accuracies,
                        mode='lines+markers',
                        name='Accuracy',
                        line=dict(color='#ff7f0e', width=2),
                        marker=dict(size=4)
                    ))
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(l=30, r=30, t=50, b=30),
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            font=dict(size=16)
                        ),
                        title=f"Trend Overview",
                        title_font_size=20,
                        font=dict(size=16),
                        xaxis_title="Epoch",
                        yaxis_title="Score",
                        xaxis=dict(title_font_size=16, tickfont_size=14),
                        yaxis=dict(title_font_size=16, tickfont_size=14)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_epochs_view(self, project_name: str):
        """Render the epochs view for a selected project."""
        epochs = get_epoch_list(str(self.work_folder), project_name)
        
        # Breadcrumb navigation
        st.markdown(f"""
        <div class="navigation-breadcrumb">
            🏠 <a href="#" onclick="location.reload()">Projects</a> > 📊 {project_name}
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("← Back to Projects", key="back_to_projects"):
            st.session_state.selected_project = None
            st.rerun()
        
        st.markdown(f"### 📈 Epochs for {project_name}")
        
        if not epochs:
            st.warning("No epochs found for this project!")
            return
        
        # Create epochs overview chart - sort epochs first for proper line connections
        sorted_epochs = sorted(epochs, key=sort_epoch_key)
        
        epochs_df = []
        for epoch_name in sorted_epochs:
            epoch_stats = get_epoch_stats(str(self.work_folder), project_name, epoch_name)
            epochs_df.append({
                'Epoch': epoch_name,
                'Average Score': epoch_stats.get('average_score', 0),
                'Average Accuracy': epoch_stats.get('average_accuracy', 0),
                'Average Tool Format Reward': epoch_stats.get('average_tool_format_reward', 0),
                'Average Combined Reward': epoch_stats.get('average_combined_reward', 0),
                'Total Rollouts': epoch_stats.get('total_rollouts', 0),
                'Total Samples': epoch_stats.get('total_groups', 0),
                'Average Turn Count': epoch_stats.get('average_turn_count', 0),
                'Average Output Tokens': epoch_stats.get('average_output_tokens', 0),
                'Average Response Tokens per Turn': epoch_stats.get('average_responses_tokens_per_turn', 0)
            })
        
        epochs_df = pd.DataFrame(epochs_df)
        
        if not epochs_df.empty:
            # Create subplots for metrics
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Score & Accuracy', 'Tool Format Reward', 'Combined Reward', 'All Rewards Comparison', 'Turn Count', 'Average Response Tokens per Turn'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            

            fig.add_trace(
                go.Scatter(x=epochs_df['Epoch'], y=epochs_df['Average Accuracy'], 
                          name='Accuracy', line=dict(color='#ff7f0e')),
                row=1, col=1
            )
            
            # Tool Format and Reward
            fig.add_trace(
                go.Scatter(x=epochs_df['Epoch'], y=epochs_df['Average Tool Format Reward'], 
                          name='Tool Format Reward', line=dict(color='#2ca02c')),
                row=1, col=2
            )

            # Combined Reward
            fig.add_trace(
                go.Scatter(x=epochs_df['Epoch'], y=epochs_df['Average Combined Reward'], 
                          name='Combined Reward', line=dict(color='#9467bd')),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=epochs_df['Epoch'], y=epochs_df['Average Turn Count'], 
                          name='Turn Count', line=dict(color='#d62728')),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=epochs_df['Epoch'], y=epochs_df['Average Response Tokens per Turn'], 
                          name='Response Tokens per Turn', line=dict(color='#d62728')),
                row=3, col=2
            )
            
            # All Rewards Comparison
            fig.add_trace(
                go.Scatter(x=epochs_df['Epoch'], y=epochs_df['Average Tool Format Reward'], 
                          name='Tool Format Reward', line=dict(color='#2ca02c', dash='dot')),
                row=2, col=2
            )

            fig.add_trace(
                go.Scatter(x=epochs_df['Epoch'], y=epochs_df['Average Accuracy'], 
                          name='Accuracy', line=dict(color='#ff7f0e', dash='dot')),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=epochs_df['Epoch'], y=epochs_df['Average Combined Reward'], 
                          name='Combined', line=dict(color='#9467bd', dash='dot')),
                row=2, col=2
            )
            
            fig.update_layout(
                height=1000, 
                showlegend=False, 
                title_text="Comprehensive Epoch Metrics Overview",
                title_font_size=28,
                font=dict(size=20),
                xaxis=dict(tickfont=dict(size=18)),
                yaxis=dict(tickfont=dict(size=18))
            )
            
            # Update all subplot titles and axis labels font sizes
            fig.update_annotations(font_size=22)
            fig.update_xaxes(title_font_size=20, tickfont_size=18)
            fig.update_yaxes(title_font_size=20, tickfont_size=18)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Epochs list
        st.markdown("### 📝 Epoch Details")
        
        # Sort epochs properly (numerically when possible)  
        sorted_epochs = sorted(epochs, key=sort_epoch_key)
        
        for epoch_name in sorted_epochs:
            stats = get_epoch_stats(str(self.work_folder), project_name, epoch_name)
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                if st.button(f"📊 {epoch_name}", key=f"epoch_{epoch_name}", use_container_width=True):
                    st.session_state.selected_epoch = epoch_name
                    st.session_state.selected_group = None
                    st.rerun()
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <strong>📊 Metrics:</strong> Score: {stats.get('average_score', 0):.3f} | Accuracy: {stats.get('average_accuracy', 0):.3f} | Tool Format: {stats.get('average_tool_format_reward', 0):.3f} |  Combined: {stats.get('average_combined_reward', 0):.3f}
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <strong>📈 Counts:</strong> Valid Samples: {stats.get('total_groups', 0)} | Rollouts: {stats.get('total_rollouts', 0)}
                </div>
                """, unsafe_allow_html=True)
    
    def render_groups_view(self, project_name: str, epoch_name: str):
        """Render the groups view for a selected epoch."""
        groups = get_epoch_groups(str(self.work_folder), project_name, epoch_name)
        
        # Breadcrumb navigation
        st.markdown(f"""
        <div class="navigation-breadcrumb">
            🏠 Projects > 📊 {project_name} > 📈 {epoch_name}
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("← Back to Epochs", key="back_to_epochs"):
            st.session_state.selected_epoch = None
            st.rerun()
        
        st.markdown(f"### 📝 Samples for epoch {epoch_name}")
        
        if not groups:
            st.warning("No groups found for this epoch!")
            return
        
        # Sort groups by average accuracy (descending)
        sorted_groups = sorted(groups.items(), key=lambda x: x[1]['average_accuracy'], reverse=True)
        
        # Groups overview chart
        group_names = [f"Sample {i+1}" for i in range(len(sorted_groups))]
        group_scores = [group_data['average_score'] for _, group_data in sorted_groups]
        group_accuracies = [group_data['average_accuracy'] for _, group_data in sorted_groups]
        group_turn_counts = [group_data['average_turn_count'] for _, group_data in sorted_groups]
        group_average_responses_tokens = [group_data['average_responses_tokens_per_turn'] for _, group_data in sorted_groups]
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Score by Sample', 'Average Accuracy by Sample', 'Average Turn Count by Sample', 'Average Response Tokens per turn '),
            vertical_spacing=0.15,  # Add vertical spacing between rows
            horizontal_spacing=0.1  # Add horizontal spacing between columns
        )
        
        fig.add_trace(
            go.Bar(x=group_names, y=group_scores, name='Score', marker_color='#1f77b4'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=group_names, y=group_accuracies, name='Accuracy', marker_color='#ff7f0e'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=group_names, y=group_turn_counts, name='Turn Count', marker_color='#2ca02c'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=group_names, y=group_average_responses_tokens, name='Response Tokens per Turn', marker_color='#9467bd'),
            row=2, col=2
        )
        fig.update_layout(
            height=1000,  # Increase height to accommodate 4 charts with larger fonts
            showlegend=False, 
            title_text="Sample Performance Overview",
            title_x=0.5,  # Center the title
            title_font_size=28,
            font=dict(size=20),
            margin=dict(t=120, b=80, l=80, r=80)  # Add larger margins for better spacing
        )
        
        # Update all subplot titles and axis labels font sizes
        fig.update_annotations(font_size=22)
        fig.update_xaxes(title_font_size=20, tickfont_size=18)
        fig.update_yaxes(title_font_size=20, tickfont_size=18)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display groups as direct buttons
        for idx, (input_text, group_data) in enumerate(sorted_groups):
            if st.button(f"📝 Sample {idx + 1} - {group_data['count']} rollouts (Accuracy: {group_data['average_accuracy']:.3f}) (Turn Count: {group_data['average_turn_count']:.3f}) (Response Tokens per Turn: {group_data['average_responses_tokens_per_turn']:.3f})", key=f"group_button_{idx}", use_container_width=True):
                st.session_state.selected_group = input_text
                st.session_state.selected_sample_index = idx
                st.rerun()
    
    def render_rollouts_view(self, project_name: str, epoch_name: str, group_input: str):
        """Render the rollouts view for a selected group."""
        rollouts = get_group_rollouts(str(self.work_folder), project_name, epoch_name, group_input)
        
        # Breadcrumb navigation
        st.markdown(f"""
        <div class="navigation-breadcrumb">
            🏠 Projects > 📊 {project_name} > 📈 {epoch_name} > 📝 Sample {st.session_state.selected_sample_index + 1}
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("← Back to Samples", key="back_to_groups"):
            st.session_state.selected_group = None
            st.rerun()
        
        st.markdown(f"### 🎯 Rollouts ({len(rollouts)} total)")
        
        if not rollouts:
            st.warning("No rollouts found for this group!")
            return
        
        # Calculate median accuracy for splitting

        median_accuracy = 0.5
        
        # Split rollouts into worse and better
        worse_rollouts = [r for r in rollouts if r.get('accuracy_reward', 0) <= 0.5]
        better_rollouts = [r for r in rollouts if r.get('accuracy_reward', 0) > 0.5]
        
        # Display user prompt
        st.markdown("**User Prompt:**")
        user_prompt = group_input.split('\nuser\n', 1)[1].strip() if '\nuser\n' in group_input else group_input[:200]
        st.text_area("", value=user_prompt, height=200, key="group_prompt", disabled=True)
        
        # Extract and display answer distribution from \boxed{} patterns
        st.markdown("---")
        st.markdown("### 📊 Answer Distribution")
        
        # Extract all answers from \boxed{} patterns in rollout inputs
        answer_counts = defaultdict(int)
        answer_accuracy = defaultdict(float)
        boxed_pattern = r'\\boxed{([^}]+)}'
        
        for rollout in rollouts:
            output_text = rollout.get('output', '')
            matches = re.findall(boxed_pattern, output_text)
            if matches:  # Only process if we found at least one match
                answer = matches[-1].strip()
                answer_counts[answer] += 1
                answer_accuracy[answer] += rollout.get('accuracy_reward',0) / 0.9
        for answer, accuracy in answer_accuracy.items():
            answer_accuracy[answer] = accuracy/answer_counts[answer]
        
        if answer_counts:
            # Sort answers by frequency (descending)
            sorted_answers = sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Create horizontal bar chart
            answers = [item[0] for item in sorted_answers]
            counts = [item[1] for item in sorted_answers]


            # Create markdown table
            table_md = "| Answer | Count | Accuracy |\n|--------|-------|----------|\n"
            for answer, count in zip(answers, counts):
                table_md += f"| {answer} | {count} | {answer_accuracy[answer]:.3f} |\n"
            st.markdown(table_md)


            # Display summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Unique Answers", len(answer_counts))
            with col2:
                st.metric("Most Common Answer", sorted_answers[0][0] if sorted_answers else "N/A")
            with col3:
                st.metric("Ground Truth", rollouts[0].get('ground_truth', 'N/A'))
            with col4:
                st.metric("Max Frequency", sorted_answers[0][1] if sorted_answers else 0)

        else:
            st.info("No answers found in \\boxed{} patterns in the rollout inputs.")
        
        # Two-column layout for worse/better rollouts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📉 Lower Accuracy Rollouts")
            st.caption(f"False Rollouts ({len(worse_rollouts)} rollouts)")
            
            for idx, rollout in enumerate(sorted(worse_rollouts, key=lambda x: x.get('accuracy_reward', 0))):
                self.render_rollout_card(rollout, idx, "worse")
        
        with col2:
            st.markdown("### 📈 Higher Accuracy Rollouts")
            st.caption(f"True Rollouts ({len(better_rollouts)} rollouts)")
            
            for idx, rollout in enumerate(sorted(better_rollouts, key=lambda x: x.get('accuracy_reward', 0), reverse=True)):
                self.render_rollout_card(rollout, idx, "better")
    
    def render_rollout_card(self, rollout: Dict, idx: int, category: str):
        """Render a single rollout card."""
        score = rollout.get('score', 0)
        accuracy = rollout.get('accuracy_reward', 0)
        combined_reward = rollout.get('combined_reward', 0)
        tool_format = rollout.get('tool_format_reward', 0)
        card_class = "better-rollout" if category == "better" else "worse-rollout"
        
        with st.container():
            st.markdown(f"""
            <div class="rollout-card {card_class}">
                <strong>Rollout {idx + 1}</strong>
                <br><strong>Score:</strong> {score:.3f}
                <br><strong>Accuracy:</strong> {accuracy:.3f}
                <br><strong>Tool Format:</strong> {tool_format:.3f}
                <br><strong>Combined:</strong> {combined_reward:.3f}
            </div>
            """, unsafe_allow_html=True)
            
            # Show output in an expander
            with st.expander(f"View Output", expanded=False):
                output = rollout.get('output', 'No output available')
                
                # Display formatted conversation
                st.markdown("**Conversation Output:**")
                self.display_conversation(output, f"output_{category}_{idx}")
                
                # Show ground truth if available
                if 'ground_truth' in rollout:
                    st.markdown("---")
                    st.markdown("### **Ground Truth:**")
                    ground_truth = rollout['ground_truth']
                    self.display_conversation(ground_truth, f"gt_{category}_{idx}", is_ground_truth=True)

def main():
    """Main application function."""
    load_custom_css()
    
    # Initialize session state
    if 'selected_project' not in st.session_state:
        st.session_state.selected_project = None
    if 'selected_epoch' not in st.session_state:
        st.session_state.selected_epoch = None
    if 'selected_group' not in st.session_state:
        st.session_state.selected_group = None
    
    # Sidebar for work folder selection
    with st.sidebar:
        st.markdown("## 📁 Configuration")
        
        # Work folder input
        work_folder = st.text_input(
            "Work Folder Path:",
            value=st.session_state.get('work_folder', './rollout_data'),
            help="Path to the folder containing project subdirectories"
        )
        
        # Cache control buttons
        if st.button("🔄 Clear Cache", help="Clear Streamlit's native cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")
            st.rerun()
        
        # Store work folder in session state
        st.session_state.work_folder = work_folder
        
        # Cache status
        st.markdown("## 💾 Cache Status")
        st.markdown("Using Streamlit's native caching system for optimal performance.")
        
        # Navigation info
        st.markdown("## 📍 Navigation")
        st.markdown(f"""
        **Current Path:**
        - Project: {st.session_state.selected_project or 'None'}
        - Epoch: {st.session_state.selected_epoch or 'None'}
        - Sample: {'Selected' if st.session_state.selected_group else 'None'}
        """)
        
        # Project history in sidebar
        if 'project_history' in st.session_state and st.session_state.project_history:
            st.markdown("## 🕒 Recent Projects")
            for project in st.session_state.project_history:
                if st.button(f"🚀 {project}", key=f"sidebar_history_{project}", use_container_width=True):
                    st.session_state.selected_project = project
                    st.session_state.selected_epoch = None
                    st.session_state.selected_group = None
                    st.rerun()
            
            # Clear history button
            if st.button("🗑️ Clear History", key="clear_history", use_container_width=True):
                st.session_state.project_history = []
                st.rerun()
    
    # Main content area
    try:
        visualizer = GRPOVisualizer(work_folder)
        
        # Navigation logic
        if st.session_state.selected_project is None:
            visualizer.render_projects_view()
        elif st.session_state.selected_epoch is None:
            visualizer.render_epochs_view(st.session_state.selected_project)
        elif st.session_state.selected_group is None:
            visualizer.render_groups_view(st.session_state.selected_project, st.session_state.selected_epoch)
        else:
            visualizer.render_rollouts_view(
                st.session_state.selected_project, 
                st.session_state.selected_epoch, 
                st.session_state.selected_group
            )
    
    except Exception as e:
        st.error(f"Error initializing visualizer: {e}")
        st.info("Please check that the work folder path is correct and contains valid project data.")

if __name__ == "__main__":
    main() 