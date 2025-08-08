# 🎯 GRPO Rollout Visualizer

A beautiful and interactive visualizer for GRPO (Generalized Preference Optimization) rollouts. This tool provides a hierarchical navigation system to explore your training data across projects, epochs, and individual rollouts with stunning visualizations.

## ✨ Features

### 🏗️ Hierarchical Navigation
- **Projects View**: Browse different training projects
- **Epochs View**: Explore training epochs with performance metrics
- **Sample View**: Analyze input groups and their statistics
- **Rollouts View**: Compare individual rollouts side-by-side

### 📊 Rich Visualizations
- **Interactive Charts**: Plotly-powered charts for metrics visualization
- **Performance Trends**: Track improvements across epochs
- **Comparison Views**: Split rollouts by performance (better vs worse)
- **Statistical Insights**: Average scores, accuracy metrics, and standard deviations

### 🎨 Beautiful UI
- **Modern Design**: Gradient cards and smooth animations
- **Responsive Layout**: Works on different screen sizes
- **Intuitive Navigation**: Breadcrumb navigation and clear visual hierarchy
- **Color-coded Performance**: Green for better, red for worse rollouts

## 📦 Installation

### Install Dependencies
```bash
pip install -r visualizer_requirements.txt
```

### Dependencies
- **Streamlit**: Web app framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Plotly**: Interactive visualizations

## 🚀 Usage

### Directory Structure
Your work folder should be organized as follows:
```
work_folder/
├── project1/
│   ├── epoch1.jsonl
│   ├── epoch2.jsonl
│   └── epoch3.jsonl
├── project2/
│   ├── epoch1.jsonl
│   └── epoch2.jsonl
└── project3/
    └── epoch1.jsonl
```

### JSONL File Format
Each line in the JSONL file should contain a rollout dictionary with the following structure:
```json
{
  "input": "Text input with \\nuser\\n marker",
  "output": "Model output/response",
  "ground_truth": "Expected answer (optional)",
  "score": 0.85,
  "accuracy_reward": 0.9,
  "tool_format_reward": 0.1,
  "combined_reward": 0.87,
  "step": 0
}
```

### Running the Visualizer

1. **Start the application**:
   ```bash
   streamlit run grpo_visualizer.py
   ```

2. **Configure the work folder**:
   - Enter the path to your work folder in the sidebar
   - Click "🔄 Refresh Data" to load the data

3. **Navigate through your data**:
   - **Projects**: Click on a project to explore its epochs
   - **Epochs**: View epoch performance and click to see input groups
   - **Sample**: Examine grouped inputs and their statistics
   - **Rollouts**: Compare individual rollouts split by performance

## 🎮 Interface Guide

### 📁 Projects View
- **Project Cards**: Shows project statistics (epochs, rollouts, average score)
- **Quick Access**: Click any project to dive into its epochs
- **Overview**: Get a high-level view of all your training projects

### 📈 Epochs View
- **Performance Charts**: Interactive plots showing score and accuracy trends
- **Epoch Details**: Click on any epoch to explore its input groups
- **Metrics Dashboard**: Comprehensive overview of training progress

### 📝 Sample View
- **Input Grouping**: All rollouts with the same input are grouped together
- **User Prompts**: Displays the text after `\nuser\n` in the input
- **Sample Statistics**: Average scores, accuracy, and standard deviations
- **Performance Ranking**: Sample sorted by accuracy (best first)

### 🎯 Rollouts View
- **Side-by-Side Comparison**: Lower accuracy rollouts on the left, higher accuracy on the right
- **Detailed Metrics**: Score, accuracy, reward, and combined reward for each rollout
- **Output Inspection**: Expandable views to examine model outputs
- **Ground Truth**: Compare outputs with expected answers (when available)

## 🔧 Customization

### Styling
The visualizer uses custom CSS for beautiful styling. You can modify the `load_custom_css()` function to adjust:
- Color schemes
- Card designs
- Layout spacing
- Font sizes

### Data Processing
The `GRPOVisualizer` class can be extended to:
- Support different data formats
- Add custom metrics
- Implement additional visualizations
- Handle special data preprocessing

## 📊 Key Metrics

### Score Metrics
- **Score**: Overall performance score
- **Accuracy Reward**: Accuracy-based reward
- **Reward**: Base reward value
- **Combined Reward**: Composite reward metric

### Statistics
- **Average**: Mean values across rollouts
- **Standard Deviation**: Measure of variance
- **Sample Count**: Number of rollouts per input group
- **Performance Split**: Median-based classification

## 🎨 UI Components

### Visual Elements
- **Gradient Cards**: Beautiful project and epoch cards
- **Interactive Charts**: Plotly-powered visualizations
- **Responsive Grid**: Adaptive layout system
- **Color Coding**: Performance-based visual cues

### Navigation
- **Breadcrumb Navigation**: Always know where you are
- **Back Buttons**: Easy navigation between levels
- **Session State**: Maintains your position during exploration
- **Sidebar Controls**: Configuration and navigation info

## 🔍 Data Insights

### Performance Analysis
- **Trend Tracking**: Monitor improvements across epochs
- **Comparative Analysis**: Better vs worse rollout comparison
- **Statistical Overview**: Mean, standard deviation, and distributions
- **Input Grouping**: Understand performance per input type

### Quality Assessment
- **Accuracy Metrics**: Detailed accuracy breakdowns
- **Score Distributions**: Understand performance variations
- **Output Inspection**: Manual quality assessment
- **Ground Truth Comparison**: Validate model outputs

## 🛠️ Technical Details

### Architecture
- **Modular Design**: Clean separation of concerns
- **Session Management**: Streamlit session state for navigation
- **Efficient Data Loading**: Optimized JSONL parsing
- **Responsive UI**: Modern web interface

### Performance
- **Lazy Loading**: Data loaded on demand
- **Caching**: Efficient data retrieval
- **Memory Management**: Optimized for large datasets
- **Interactive Updates**: Real-time UI updates

## 🤝 Contributing

Feel free to contribute improvements:
- **Bug Reports**: Submit issues with detailed descriptions
- **Feature Requests**: Suggest new visualization features
- **Code Improvements**: Submit pull requests with enhancements
- **Documentation**: Help improve this README

## 📝 License

This project follows the same license as the main VERL project.

---

**Happy Visualizing!** 🎉 Explore your GRPO rollouts with style and gain insights into your model's training progress. 