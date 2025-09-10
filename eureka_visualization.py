import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(__file__).parent
DEFAULT_RESULTS_DIR = BASE_DIR / "EurekaRewards" 
DEFAULT_TENSORBOARD_DIR = BASE_DIR / "EurekaRewards" / "tensorboard"

st.set_page_config(
    page_title="Eureka RL Results Visualization",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'results_dir' not in st.session_state:
    st.session_state.results_dir = DEFAULT_RESULTS_DIR
if 'selected_files' not in st.session_state:
    st.session_state.selected_files = []

class EurekaDataLoader:
    """Handles loading and parsing of Eureka RL results"""
    
    @staticmethod
    def load_json_results(filepath: Path) -> Dict:
        """Load a single JSON result file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def get_available_files(directory: Path) -> List[Path]:
        """Get JSON files from specific subdirectories"""
        files = []
        
        # Check for conversations folder
        conversations_dir = directory / "conversations"
        if conversations_dir.exists():
            files.extend(sorted(conversations_dir.glob("*.json")))
        
        # Check for results folder
        results_dir = directory / "results"
        if results_dir.exists():
            files.extend(sorted(results_dir.glob("*.json")))
        
        # Check for rewards folder
        rewards_dir = directory / "rewards"
        if rewards_dir.exists():
            files.extend(sorted(rewards_dir.glob("*.json")))
        
        return files
    
    @staticmethod
    def get_subdirectories(directory: Path) -> List[Path]:
        """Get all subdirectories in the given directory"""
        if not directory.exists():
            return []
        return sorted([d for d in directory.iterdir() if d.is_dir()])
    
    @staticmethod
    def parse_filename(filename: str) -> Dict[str, str]:
        """Parse iteration, sample, and attempt from filename"""
        parts = filename.replace(".json", "").split("_")
        info = {"filename": filename}
        
        for i, part in enumerate(parts):
            if part.startswith("iter") and i + 1 < len(parts):
                info["iteration"] = int(parts[i][4:])
            elif part.startswith("sample") and i + 1 < len(parts):
                info["sample"] = int(parts[i][6:])
            elif part.startswith("attempt"):
                info["attempt"] = int(part[7:])
        
        return info
    
    @staticmethod
    def extract_checkpoint_data(data: Dict) -> pd.DataFrame:
        """Extract checkpoint data into a DataFrame"""
        records = []
        
        for result in data.get("results", []):
            if result.get("type") == "checkpoint":
                record = {
                    "step_number": result.get("step_number", 0),
                    "checkpoint_number": result.get("checkpoint_number", 0),
                    "success_rate": result.get("success_rate", 0),
                    "total_episodes": result.get("total_episodes", 0),
                    "training_time": result.get("training_time", 0),
                    "episode_reward_mean": result.get("episode_reward_stats", {}).get("mean", 0),
                    "episode_reward_std": result.get("episode_reward_stats", {}).get("std", 0),
                    "episode_length_mean": result.get("episode_length_stats", {}).get("mean", 0),
                    "episode_length_std": result.get("episode_length_stats", {}).get("std", 0),
                }
                
                # Extract reward components
                for component in result.get("reward_components", []):
                    component_name = f"reward_{component['name']}"
                    record[f"{component_name}_mean"] = component["stats"]["mean"]
                    record[f"{component_name}_std"] = component["stats"]["std"]
                
                # Extract observations
                for obs in result.get("observations", []):
                    obs_name = f"obs_{obs['name']}"
                    record[f"{obs_name}_mean"] = obs["stats"]["mean"]
                    record[f"{obs_name}_std"] = obs["stats"]["std"]
                
                records.append(record)
        
        return pd.DataFrame(records)
    
    @staticmethod
    def get_compilation_errors(data: Dict) -> List[Dict]:
        """Extract compilation errors from results"""
        errors = []
        for result in data.get("results", []):
            if result.get("type") == "compilation_error":
                errors.append({
                    "error_message": result.get("error_message", "Unknown error"),
                    "reward_path": result.get("reward_path", "Unknown path")
                })
        return errors

def create_training_progress_plot(df: pd.DataFrame, title: str = "Training Progress") -> go.Figure:
    """Create a plot showing training progress over time"""
    fig = go.Figure()
    
    # Success rate
    fig.add_trace(go.Scatter(
        x=df["step_number"],
        y=df["success_rate"],
        mode='lines+markers',
        name='Success Rate',
        yaxis='y',
        line=dict(color='green', width=2)
    ))
    
    # Episode reward mean
    fig.add_trace(go.Scatter(
        x=df["step_number"],
        y=df["episode_reward_mean"],
        mode='lines+markers',
        name='Episode Reward (Mean)',
        yaxis='y2',
        line=dict(color='blue', width=2)
    ))
    
    # Add std band for rewards
    reward_upper = df["episode_reward_mean"] + df["episode_reward_std"]
    reward_lower = df["episode_reward_mean"] - df["episode_reward_std"]
    
    fig.add_trace(go.Scatter(
        x=df["step_number"].tolist() + df["step_number"].tolist()[::-1],
        y=reward_upper.tolist() + reward_lower.tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,100,255,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Step Number",
        yaxis=dict(
            title="Success Rate",
            titlefont=dict(color="green"),
            tickfont=dict(color="green"),
            range=[0, 1.05]
        ),
        yaxis2=dict(
            title="Episode Reward",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_reward_components_plot(df: pd.DataFrame) -> go.Figure:
    """Create a plot showing individual reward components"""
    reward_cols = [col for col in df.columns if col.startswith("reward_") and col.endswith("_mean")]
    
    fig = go.Figure()
    
    for col in reward_cols:
        component_name = col.replace("reward_", "").replace("_mean", "")
        fig.add_trace(go.Scatter(
            x=df["step_number"],
            y=df[col],
            mode='lines+markers',
            name=component_name
        ))
    
    fig.update_layout(
        title="Reward Components Over Time",
        xaxis_title="Step Number",
        yaxis_title="Reward Value",
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_comparison_plot(data_dict: Dict[str, pd.DataFrame], metric: str = "success_rate") -> go.Figure:
    """Create a comparison plot for multiple runs"""
    fig = go.Figure()
    
    for name, df in data_dict.items():
        if metric in df.columns:
            fig.add_trace(go.Scatter(
                x=df["step_number"],
                y=df[metric],
                mode='lines+markers',
                name=name
            ))
    
    fig.update_layout(
        title=f"{metric.replace('_', ' ').title()} Comparison",
        xaxis_title="Step Number",
        yaxis_title=metric.replace('_', ' ').title(),
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_heatmap(files: List[Path]) -> go.Figure:
    """Create a heatmap of final success rates across iterations and samples"""
    data_matrix = {}
    
    for filepath in files:
        info = EurekaDataLoader.parse_filename(filepath.name)
        if "iteration" in info and "sample" in info:
            try:
                data = EurekaDataLoader.load_json_results(filepath)
                df = EurekaDataLoader.extract_checkpoint_data(data)
                if not df.empty:
                    final_success_rate = df.iloc[-1]["success_rate"]
                    key = (info["iteration"], info["sample"])
                    if "attempt" in info:
                        # Keep the best attempt
                        if key not in data_matrix or final_success_rate > data_matrix[key]:
                            data_matrix[key] = final_success_rate
                    else:
                        data_matrix[key] = final_success_rate
            except:
                continue
    
    if not data_matrix:
        return go.Figure()
    
    # Convert to matrix format
    iterations = sorted(set(k[0] for k in data_matrix.keys()))
    samples = sorted(set(k[1] for k in data_matrix.keys()))
    
    z = []
    for iter_num in iterations:
        row = []
        for sample_num in samples:
            row.append(data_matrix.get((iter_num, sample_num), 0))
        z.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"Sample {s}" for s in samples],
        y=[f"Iteration {i}" for i in iterations],
        colorscale='Viridis',
        text=[[f"{val:.2f}" for val in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Final Success Rate")
    ))
    
    fig.update_layout(
        title="Final Success Rates Heatmap",
        xaxis_title="Sample",
        yaxis_title="Iteration",
        height=500
    )
    
    return fig

def main():
    st.title("🤖 Eureka RL Results Visualization")
    st.markdown("Interactive dashboard for exploring Eureka reinforcement learning results")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Selection")
        
        # Enhanced Folder Navigation
        current_dir = st.session_state.results_dir
        
        # Direct path input for folder navigation
        with st.expander("📝 Enter Path Directly", expanded=False):
            direct_path = st.text_input(
                "Enter full path:",
                value=str(current_dir),
                help="Enter the full path to the directory containing your JSON files"
            )
            if st.button("Go to Path"):
                try:
                    new_path = Path(direct_path)
                    if new_path.exists() and new_path.is_dir():
                        st.session_state.results_dir = new_path
                        st.rerun()
                    else:
                        st.error("Invalid directory path")
                except:
                    st.error("Invalid path format")
        
        # Option 2: Breadcrumb navigation
        st.markdown("**📍 Current Location:**")
        path_parts = current_dir.parts
        breadcrumb_html = ""
        for i, part in enumerate(path_parts):
            if i > 0:
                breadcrumb_html += " / "
            if i == len(path_parts) - 1:
                breadcrumb_html += f"<b>{part}</b>"
            else:
                breadcrumb_html += part
        st.markdown(f"<small>{breadcrumb_html}</small>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Enhanced directory browser with tree view
        st.markdown("**📁 Browse Folders:**")
        
        # Show parent directory option
        parent_dir = current_dir.parent
        if parent_dir != current_dir:
            if st.button(f"⬆️ .. (Parent Directory)", use_container_width=True):
                st.session_state.results_dir = parent_dir
                st.rerun()
        
        # Get and display subdirectories with file counts
        subdirs = EurekaDataLoader.get_subdirectories(current_dir)
        if subdirs:
            for subdir in subdirs:
                # Count JSON files in each subdirectory
                json_count = len(list(subdir.glob("**/*.json")))
                folder_icon = "📁" if json_count == 0 else "📂"
                button_label = f"{folder_icon} {subdir.name} ({json_count} files)"
                
                if st.button(button_label, key=f"dir_{subdir}", use_container_width=True):
                    st.session_state.results_dir = subdir
                    st.rerun()
        else:
            st.info("No subdirectories found")
        
        st.markdown("---")
        
        # Get available files
        files = EurekaDataLoader.get_available_files(current_dir)
        
        if not files:
            st.warning(f"No JSON files found in {current_dir.name}")
        else:
            st.markdown("**📄 Available Files:**")
            
            # Select all / Deselect all buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Select All"):
                    st.session_state.selected_files = files.copy()
                    st.rerun()
            with col2:
                if st.button("❌ Deselect All"):
                    st.session_state.selected_files = []
                    st.rerun()
            
            st.markdown("---")
            
            # Organize files by folder
            conversations_files = [f for f in files if "conversations" in str(f.relative_to(current_dir))]
            results_files = [f for f in files if "results" in str(f.relative_to(current_dir))]
            rewards_files = [f for f in files if "rewards" in str(f.relative_to(current_dir))]
            
            # Display conversation files
            if conversations_files:
                st.markdown("**💬 Conversations**")
                for file in conversations_files:
                    is_selected = file in st.session_state.selected_files
                    display_name = f"💬 {file.name}"
                    
                    if st.checkbox(display_name, value=is_selected, key=f"file_{file}"):
                        if file not in st.session_state.selected_files:
                            st.session_state.selected_files.append(file)
                            st.rerun()
                    else:
                        if file in st.session_state.selected_files:
                            st.session_state.selected_files.remove(file)
                            st.rerun()
            
            # Display results files
            if results_files:
                st.markdown("**📊 Results**")
                for file in results_files:
                    is_selected = file in st.session_state.selected_files
                    display_name = f"📊 {file.name}"
                    
                    if st.checkbox(display_name, value=is_selected, key=f"file_{file}"):
                        if file not in st.session_state.selected_files:
                            st.session_state.selected_files.append(file)
                            st.rerun()
                    else:
                        if file in st.session_state.selected_files:
                            st.session_state.selected_files.remove(file)
                            st.rerun()
            
            # Display rewards files
            if rewards_files:
                st.markdown("**🎯 Rewards**")
                for file in rewards_files:
                    is_selected = file in st.session_state.selected_files
                    display_name = f"🎯 {file.name}"
                    
                    if st.checkbox(display_name, value=is_selected, key=f"file_{file}"):
                        if file not in st.session_state.selected_files:
                            st.session_state.selected_files.append(file)
                            st.rerun()
                    else:
                        if file in st.session_state.selected_files:
                            st.session_state.selected_files.remove(file)
                            st.rerun()
            
            # Show selected count
            if st.session_state.selected_files:
                st.markdown("---")
                st.success(f"📊 {len(st.session_state.selected_files)} file(s) selected")
            else:
                st.info("No files selected")
        
        st.markdown("---")
        
        # Visualization options
        st.header("📊 Visualization Options")
        show_training_progress = st.checkbox("Training Progress", value=True)
        show_reward_components = st.checkbox("Reward Components", value=True)
        show_comparison = st.checkbox("Multi-Run Comparison", value=len(st.session_state.selected_files) > 1)
        show_heatmap = st.checkbox("Success Rate Heatmap", value=True)
        show_statistics = st.checkbox("Statistical Summary", value=True)
        
        if show_comparison and len(st.session_state.selected_files) > 1:
            comparison_metric = st.selectbox(
                "Comparison Metric:",
                ["success_rate", "episode_reward_mean", "episode_length_mean"]
            )
    
    # Main content
    if not st.session_state.selected_files:
        st.info("👈 Please select one or more result files from the sidebar")
        return
    
    selected_files = st.session_state.selected_files
    
    # Load data
    data_dict = {}
    all_dfs = {}
    errors_dict = {}
    conversation_data = {}
    
    for filepath in selected_files:
        try:
            # Check if it's a conversation file
            if "conversations" in str(filepath):
                data = EurekaDataLoader.load_json_results(filepath)
                conv_label = filepath.stem
                conversation_data[conv_label] = data
            else:
                # Existing results file handling
                data = EurekaDataLoader.load_json_results(filepath)
                df = EurekaDataLoader.extract_checkpoint_data(data)
                errors = EurekaDataLoader.get_compilation_errors(data)
                
                file_info = EurekaDataLoader.parse_filename(filepath.name)
                label = f"Iter{file_info.get('iteration', '?')}_Sample{file_info.get('sample', '?')}"
                if "attempt" in file_info:
                    label += f"_Attempt{file_info['attempt']}"
                
                if not df.empty:
                    data_dict[filepath] = data
                    all_dfs[label] = df
                
                if errors:
                    errors_dict[label] = errors
                    
        except Exception as e:
            st.error(f"Error loading {filepath.name}: {str(e)}")
    
    # Display visualizations
    if all_dfs:
        # Training Progress
        if show_training_progress:
            st.header("📈 Training Progress")
            cols = st.columns(min(2, len(all_dfs)))
            for idx, (label, df) in enumerate(all_dfs.items()):
                with cols[idx % len(cols)]:
                    fig = create_training_progress_plot(df, title=label)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Reward Components
        if show_reward_components:
            st.header("🎯 Reward Components")
            for label, df in all_dfs.items():
                with st.expander(f"{label} - Reward Components"):
                    fig = create_reward_components_plot(df)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Comparison Plot
        if show_comparison and len(all_dfs) > 1:
            st.header("🔄 Multi-Run Comparison")
            fig = create_comparison_plot(all_dfs, comparison_metric)
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap
        if show_heatmap:
            st.header("🗺️ Success Rate Heatmap")
            # Get result files specifically for heatmap
            results_dir = st.session_state.results_dir / "results"
            if results_dir.exists():
                heatmap_files = sorted(results_dir.glob("*.json"))
            else:
                heatmap_files = []
            fig = create_heatmap(heatmap_files)
            if fig.data:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data for heatmap visualization")
        
        # Statistical Summary
        if show_statistics:
            st.header("📊 Statistical Summary")
            
            summary_data = []
            for label, df in all_dfs.items():
                if not df.empty:
                    final_row = df.iloc[-1]
                    summary_data.append({
                        "Run": label,
                        "Final Success Rate": f"{final_row['success_rate']:.3f}",
                        "Final Episode Reward": f"{final_row['episode_reward_mean']:.3f} ± {final_row['episode_reward_std']:.3f}",
                        "Total Episodes": int(final_row['total_episodes']),
                        "Training Time (s)": f"{final_row['training_time']:.1f}",
                        "Steps": len(df)
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
        
        # Raw Data Export
        with st.expander("💾 Export Data"):
            export_format = st.selectbox("Export Format:", ["CSV", "JSON"])
            
            if st.button("Generate Export"):
                if export_format == "CSV":
                    # Combine all dataframes
                    combined_df = pd.concat(
                        [df.assign(run=label) for label, df in all_dfs.items()],
                        ignore_index=True
                    )
                    csv = combined_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="eureka_results_export.csv",
                        mime="text/csv"
                    )
                else:  # JSON
                    export_data = {
                        label: df.to_dict(orient='records') 
                        for label, df in all_dfs.items()
                    }
                    json_str = json.dumps(export_data, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name="eureka_results_export.json",
                        mime="application/json"
                    )
    
    # Display conversation data if any
    if conversation_data:
        st.header("💬 Conversation Analysis")
        
        for conv_label, conv_data in conversation_data.items():
            with st.expander(f"{conv_label}"):
                # Display conversation metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Iteration", conv_data.get("iteration", "N/A"))
                with col2:
                    st.metric("Run ID", conv_data.get("run_id", "N/A"))
                with col3:
                    st.metric("Type", conv_data.get("conversation_type", "N/A"))
                
                # Display model metadata if available
                if "metadata" in conv_data:
                    meta = conv_data["metadata"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model", meta.get("model", "N/A"))
                    with col2:
                        st.metric("Total Tokens", meta.get("total_tokens", "N/A"))
                    with col3:
                        st.metric("Latency (ms)", f"{meta.get('latency_ms', 0):.0f}")
                
                # Display messages
                if "messages" in conv_data:
                    st.markdown("**Messages:**")
                    
                    # Master toggle for all messages in this conversation
                    expand_all = st.toggle("Expand all messages", key=f"expand_all_{conv_label}")
                    
                    for i, msg in enumerate(conv_data["messages"]):
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        msg_key = f"{conv_label}_{role}_{i}"
                        
                        if len(content) > 200:
                            preview = content[:200] + "..."
                            # Individual toggle, but overridden by master toggle
                            individual_expanded = st.toggle(f"Show full {role} message", key=f"toggle_{msg_key}", value=expand_all)
                            
                            display_content = content if (expand_all or individual_expanded) else preview
                            if role == "system":
                                st.info(f"**System:** {display_content}")
                            elif role == "user":
                                st.warning(f"**User:** {display_content}")
                            elif role == "assistant":
                                st.success(f"**Assistant:** {display_content}")
                        else:
                            if role == "system":
                                st.info(f"**System:** {content}")
                            elif role == "user":
                                st.warning(f"**User:** {content}")
                            elif role == "assistant":
                                st.success(f"**Assistant:** {content}")
                
                # Display generated content
                if "generated_content" in conv_data and conv_data["generated_content"]:
                    st.markdown("**Generated Reward Function:**")
                    content = conv_data["generated_content"]
                    code_key = f"{conv_label}_code"
                    
                    if len(content) > 500:
                        preview = content[:500] + "..."
                        is_expanded = st.toggle("Show full code", key=f"toggle_{code_key}")
                        
                        display_content = content if is_expanded else preview
                        st.code(display_content, language="csharp")
                    else:
                        st.code(content, language="csharp")
    
    # Display compilation errors at the end
    if errors_dict:
        st.header("⚠️ Compilation Errors")
        for label, errors in errors_dict.items():
            with st.expander(f"{label} - Compilation Errors"):
                for error in errors:
                    st.error(error["error_message"])
                    st.text(f"Reward Path: {error['reward_path']}")

if __name__ == "__main__":
    main()