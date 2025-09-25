import streamlit as st
import os
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
from PIL import Image
import glob

# Set page config
st.set_page_config(
    page_title="Vehicle Damage Assessment Pipeline",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = 'idle'
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'results_data' not in st.session_state:
    st.session_state.results_data = {}
if 'consolidated_result' not in st.session_state:
    st.session_state.consolidated_result = None

# Constants
RAW_IMAGES_DIR = "raw_images"
RESULTS_DIR = "results"
BATCH_PROCESSOR_SCRIPT = "artifacts/pipeline/batch_processor.py"

def ensure_directories():
    """Ensure required directories exist"""
    os.makedirs(RAW_IMAGES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

def clear_raw_images():
    """Clear existing images in raw_images directory"""
    if os.path.exists(RAW_IMAGES_DIR):
        for file in os.listdir(RAW_IMAGES_DIR):
            file_path = os.path.join(RAW_IMAGES_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

def save_uploaded_files(uploaded_files):
    """Save uploaded files to raw_images directory"""
    ensure_directories()
    clear_raw_images()
    
    saved_files = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type.startswith('image/'):
            file_path = os.path.join(RAW_IMAGES_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(uploaded_file.name)
    
    return saved_files

def run_batch_processor():
    """Run the batch processor and return the process object"""
    try:
        # Change to the project directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Run the batch processor with output capture
        process = subprocess.Popen(
            ['python', 'batch_processor.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        return process
    except Exception as e:
        st.error(f"Error starting batch processor: {str(e)}")
        return None

def read_process_output(process):
    """Read and parse output from the batch processor"""
    output_lines = []
    current_step = None
    current_image = None
    
    try:
        # For Windows, we'll use a different approach to read non-blocking output
        import threading
        import queue
        
        # Create a queue to store output lines
        if not hasattr(st.session_state, 'output_queue'):
            st.session_state.output_queue = queue.Queue()
            
            def read_output():
                try:
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            st.session_state.output_queue.put(line.strip())
                        if process.poll() is not None:
                            break
                except:
                    pass
            
            # Start reading thread
            thread = threading.Thread(target=read_output, daemon=True)
            thread.start()
        
        # Get available lines from queue
        try:
            while not st.session_state.output_queue.empty():
                line = st.session_state.output_queue.get_nowait()
                output_lines.append(line)
        except queue.Empty:
            pass
        
        # Parse output for status updates
        for line in output_lines:
            if "Processing image" in line:
                # Extract image name from log
                parts = line.split("Processing image")
                if len(parts) > 1:
                    image_part = parts[1].strip()
                    # Extract filename from path
                    if "/" in image_part or "\\" in image_part:
                        current_image = os.path.basename(image_part.split()[0])
                    else:
                        current_image = image_part.split()[0]
            elif "Pipeline execution completed" in line:
                current_step = "pipeline_completed"
            elif "Results archived" in line:
                current_step = "archiving"
            elif "Consolidating damage assessments" in line:
                current_step = "consolidation"
            elif "Batch processing completed" in line:
                current_step = "completed"
        
        return output_lines, current_step, current_image
        
    except Exception as e:
        return [], None, None

def get_latest_consolidated_result():
    """Get the latest consolidated assessment result"""
    try:
        pattern = os.path.join(RESULTS_DIR, "consolidated_vehicle_assessment_*.json")
        files = glob.glob(pattern)
        if files:
            latest_file = max(files, key=os.path.getctime)
            with open(latest_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading consolidated result: {str(e)}")
    return None

def get_intermediate_results():
    """Get intermediate results from individual image processing"""
    results = {}
    try:
        for item in os.listdir(RESULTS_DIR):
            item_path = os.path.join(RESULTS_DIR, item)
            if os.path.isdir(item_path) and item.startswith('202'):
                # Extract image name from directory name
                parts = item.split('_', 2)
                if len(parts) >= 3:
                    image_name = parts[2]
                    results[image_name] = {
                        'directory': item_path,
                        'files': []
                    }
                    
                    # Get all files in the directory
                    for file in os.listdir(item_path):
                        file_path = os.path.join(item_path, file)
                        if os.path.isfile(file_path):
                            results[image_name]['files'].append({
                                'name': file,
                                'path': file_path,
                                'type': 'json' if file.endswith('.json') else 'image'
                            })
    except Exception as e:
        st.error(f"Error loading intermediate results: {str(e)}")
    
    return results

def display_json_data(json_data, title="JSON Data"):
    """Display JSON data in a formatted way"""
    st.subheader(title)
    
    if isinstance(json_data, dict):
        # Create expandable sections for different parts of the JSON
        for key, value in json_data.items():
            with st.expander(f"📋 {key.replace('_', ' ').title()}"):
                if isinstance(value, (dict, list)):
                    st.json(value)
                else:
                    st.write(value)
    else:
        st.json(json_data)

def main():
    st.title("🚗 Vehicle Damage Assessment Pipeline")
    st.markdown("Upload vehicle images to analyze damage, identify parts, and assess severity with cost estimates.")
    
    # Sidebar for file upload and controls
    with st.sidebar:
        st.header("📁 File Upload")
        
        uploaded_files = st.file_uploader(
            "Choose vehicle images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload multiple vehicle images for damage assessment"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files selected")
            
            # Show preview of uploaded files
            st.subheader("📸 Preview")
            for i, file in enumerate(uploaded_files[:3]):  # Show first 3 images
                try:
                    image = Image.open(file)
                    st.image(image, caption=file.name, width=200)
                except Exception as e:
                    st.error(f"Error loading {file.name}: {str(e)}")
            
            if len(uploaded_files) > 3:
                st.info(f"... and {len(uploaded_files) - 3} more files")
        
        st.divider()
        
        # Process button
        if st.button("🚀 Start Processing", type="primary", disabled=not uploaded_files or st.session_state.processing_status == 'processing'):
            if uploaded_files:
                # Save files and start processing
                saved_files = save_uploaded_files(uploaded_files)
                st.session_state.uploaded_files = saved_files
                st.session_state.processing_status = 'processing'
                st.session_state.results_data = {}
                st.session_state.consolidated_result = None
                st.rerun()
        
        # Clear results button
        if st.button("🗑️ Clear Results"):
            st.session_state.processing_status = 'idle'
            st.session_state.uploaded_files = []
            st.session_state.results_data = {}
            st.session_state.consolidated_result = None
            clear_raw_images()
            st.rerun()
    
    # Main content area
    if st.session_state.processing_status == 'idle':
        st.info("👆 Upload vehicle images using the sidebar to get started")
        
        # Show sample images or instructions
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📋 Instructions")
            st.markdown("""
            1. Upload multiple vehicle images
            2. Click 'Start Processing'
            3. View intermediate results
            4. Review consolidated assessment
            """)
        
        with col2:
            st.markdown("### 🔍 What We Analyze")
            st.markdown("""
            - **Damage Detection**: Identify scratches, dents, cracks
            - **Part Identification**: Locate affected vehicle parts
            - **Severity Assessment**: Classify damage severity
            - **Cost Estimation**: Calculate repair costs in INR
            """)
        
        with col3:
            st.markdown("### 📊 Output Features")
            st.markdown("""
            - Real-time processing status
            - Individual image results
            - Consolidated damage report
            - Cost breakdown analysis
            """)
    
    elif st.session_state.processing_status == 'processing':
        st.header("⚙️ Processing Images...")
        
        # Initialize processing state if not exists
        if 'processing_steps' not in st.session_state:
            st.session_state.processing_steps = {
                'initialization': {'status': 'pending', 'message': 'Initializing batch processor...'},
                'image_processing': {'status': 'pending', 'message': 'Processing individual images...'},
                'consolidation': {'status': 'pending', 'message': 'Consolidating damage assessment...'},
                'completion': {'status': 'pending', 'message': 'Finalizing results...'}
            }
            st.session_state.current_image = None
            st.session_state.processed_images = []
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Overall progress
            st.subheader("📊 Overall Progress")
            overall_progress = st.progress(0)
            overall_status = st.empty()
            
            # Step-by-step status
            st.subheader("📋 Processing Steps")
            
            # Step indicators
            steps_container = st.container()
            with steps_container:
                for step_key, step_info in st.session_state.processing_steps.items():
                    step_col1, step_col2, step_col3 = st.columns([0.1, 0.7, 0.2])
                    
                    with step_col1:
                        if step_info['status'] == 'completed':
                            st.success("✅")
                        elif step_info['status'] == 'in_progress':
                            st.info("🔄")
                        elif step_info['status'] == 'error':
                            st.error("❌")
                        else:
                            st.write("⏳")
                    
                    with step_col2:
                        step_name = step_key.replace('_', ' ').title()
                        if step_info['status'] == 'in_progress':
                            st.write(f"**{step_name}** - {step_info['message']}")
                        else:
                            st.write(f"{step_name} - {step_info['message']}")
                    
                    with step_col3:
                        if step_info['status'] == 'in_progress':
                            st.spinner()
        
        with col2:
            # Current processing info
            st.subheader("🔍 Current Status")
            
            if st.session_state.current_image:
                st.write(f"**Processing:** {st.session_state.current_image}")
            
            if st.session_state.processed_images:
                st.write(f"**Completed:** {len(st.session_state.processed_images)}/{len(st.session_state.uploaded_files)}")
                
                # Show completed images
                with st.expander("✅ Completed Images"):
                    for img in st.session_state.processed_images:
                        st.write(f"• {img}")
            
            # Processing time
            if 'process_start_time' not in st.session_state:
                st.session_state.process_start_time = time.time()
            
            elapsed_time = time.time() - st.session_state.process_start_time
            st.metric("⏱️ Elapsed Time", f"{elapsed_time:.1f}s")
        
        # Start the batch processor if not already started
        if 'batch_process' not in st.session_state:
            st.session_state.processing_steps['initialization']['status'] = 'in_progress'
            st.session_state.batch_process = run_batch_processor()
            
            if not st.session_state.batch_process:
                st.error("❌ Failed to start processing")
                st.session_state.processing_status = 'error'
                st.rerun()
        
        # Monitor the process
        process = st.session_state.batch_process
        
        if process and process.poll() is None:
            # Process is still running
            
            # Read real-time output from process
            output_lines, current_step, detected_image = read_process_output(process)
            
            # Update current image from process output
            if detected_image and detected_image != st.session_state.current_image:
                st.session_state.current_image = detected_image
                if detected_image not in st.session_state.processed_images:
                    # Mark previous image as completed if we moved to a new one
                    if st.session_state.current_image and st.session_state.current_image != detected_image:
                        if st.session_state.current_image not in st.session_state.processed_images:
                            st.session_state.processed_images.append(st.session_state.current_image)
            
            # Update step status based on process output and intermediate results
            intermediate_results = get_intermediate_results()
            
            # Check initialization
            if st.session_state.processing_steps['initialization']['status'] == 'in_progress':
                if len(intermediate_results) > 0 or elapsed_time > 10 or current_step:
                    st.session_state.processing_steps['initialization']['status'] = 'completed'
                    st.session_state.processing_steps['initialization']['message'] = 'Batch processor initialized successfully'
                    st.session_state.processing_steps['image_processing']['status'] = 'in_progress'
            
            # Update image processing status
            if st.session_state.processing_steps['image_processing']['status'] == 'in_progress':
                completed_count = len(intermediate_results)
                total_count = len(st.session_state.uploaded_files)
                
                # Update processed images list
                new_images = [img for img in intermediate_results.keys() if img not in st.session_state.processed_images]
                if new_images:
                    st.session_state.processed_images.extend(new_images)
                
                # Update current processing message
                if st.session_state.current_image:
                    st.session_state.processing_steps['image_processing']['message'] = f'Processing {st.session_state.current_image}... ({completed_count}/{total_count} completed)'
                else:
                    st.session_state.processing_steps['image_processing']['message'] = f'Processing images... ({completed_count}/{total_count} completed)'
                
                # Check if all images are processed or consolidation started
                if completed_count >= total_count or current_step == "consolidation":
                    st.session_state.processing_steps['image_processing']['status'] = 'completed'
                    st.session_state.processing_steps['image_processing']['message'] = f'All {total_count} images processed successfully'
                    st.session_state.processing_steps['consolidation']['status'] = 'in_progress'
                    st.session_state.current_image = None
            
            # Update consolidation status
            if current_step == "consolidation" and st.session_state.processing_steps['consolidation']['status'] != 'completed':
                st.session_state.processing_steps['consolidation']['status'] = 'in_progress'
                st.session_state.processing_steps['consolidation']['message'] = 'Consolidating and deduplicating damage assessments...'
            
            # Show real-time log output if available
            if output_lines:
                with st.expander("📋 Real-time Processing Log", expanded=False):
                    for line in output_lines[-10:]:  # Show last 10 lines
                        st.text(line)
            
            # Update overall progress
            total_steps = len(st.session_state.processing_steps)
            completed_steps = sum(1 for step in st.session_state.processing_steps.values() if step['status'] == 'completed')
            
            if st.session_state.processing_steps['image_processing']['status'] == 'in_progress':
                # During image processing, show sub-progress
                image_progress = len(intermediate_results) / len(st.session_state.uploaded_files) if st.session_state.uploaded_files else 0
                overall_progress_value = (completed_steps + image_progress) / total_steps
            else:
                overall_progress_value = completed_steps / total_steps
            
            overall_progress.progress(min(overall_progress_value, 0.95))
            
            # Update overall status message
            if st.session_state.processing_steps['consolidation']['status'] == 'in_progress':
                overall_status.text("🔄 Consolidating damage assessment...")
            elif st.session_state.processing_steps['image_processing']['status'] == 'in_progress':
                overall_status.text(f"🔄 Processing images... ({len(intermediate_results)}/{len(st.session_state.uploaded_files)})")
            else:
                overall_status.text("🔄 Initializing processing pipeline...")
            
            # Auto-refresh every 2 seconds
            time.sleep(2)
            st.rerun()
            
        elif process and process.poll() is not None:
            # Process completed
            if process.poll() == 0:  # Success
                # Mark all steps as completed
                for step_key in st.session_state.processing_steps:
                    st.session_state.processing_steps[step_key]['status'] = 'completed'
                
                st.session_state.processing_steps['consolidation']['message'] = 'Damage consolidation completed'
                st.session_state.processing_steps['completion']['status'] = 'completed'
                st.session_state.processing_steps['completion']['message'] = 'All processing completed successfully!'
                
                overall_progress.progress(1.0)
                overall_status.text("✅ Processing completed successfully!")
                
                # Load results
                st.session_state.processing_status = 'completed'
                st.session_state.results_data = get_intermediate_results()
                st.session_state.consolidated_result = get_latest_consolidated_result()
                
                # Clean up processing state
                for key in ['processing_steps', 'batch_process', 'current_image', 'processed_images', 'process_start_time']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                time.sleep(2)
                st.rerun()
            else:  # Error
                # Mark current step as error
                for step_key, step_info in st.session_state.processing_steps.items():
                    if step_info['status'] == 'in_progress':
                        step_info['status'] = 'error'
                        step_info['message'] = 'Processing failed'
                        break
                
                stderr_output = process.stderr.read() if process.stderr else "Unknown error"
                st.error(f"❌ Processing failed: {stderr_output}")
                overall_status.text("❌ Processing failed")
                st.session_state.processing_status = 'error'
        else:
            st.error("❌ Failed to start processing")
            st.session_state.processing_status = 'error'
    
    elif st.session_state.processing_status in ['completed', 'error']:
        if st.session_state.processing_status == 'completed':
            st.success("✅ Processing completed successfully!")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📊 Consolidated Results", "🔍 Intermediate Results", "📁 Raw Data"])
            
            with tab1:
                display_consolidated_results()
            
            with tab2:
                display_intermediate_results()
            
            with tab3:
                display_raw_data()
        
        else:  # error
            st.error("❌ Processing encountered an error. Please try again.")

def display_consolidated_results():
    """Display the consolidated assessment results"""
    if st.session_state.consolidated_result:
        result = st.session_state.consolidated_result
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_damages = len(result.get('consolidated_damages', []))
            st.metric("Total Damages", total_damages)
        
        with col2:
            total_cost = result.get('total_estimated_cost_inr', 0)
            st.metric("Total Cost (INR)", f"₹{total_cost:,.2f}")
        
        with col3:
            avg_severity = result.get('overall_severity_distribution', {})
            if avg_severity:
                max_severity = max(avg_severity.items(), key=lambda x: x[1])
                st.metric("Dominant Severity", max_severity[0].title())
        
        with col4:
            repair_time = result.get('estimated_repair_time_days', 0)
            st.metric("Repair Time", f"{repair_time} days")
        
        st.divider()
        
        # Detailed breakdown
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔧 Damage Details")
            damages = result.get('consolidated_damages', [])
            
            if damages:
                for i, damage in enumerate(damages):
                    with st.expander(f"Damage {i+1}: {damage.get('damage_type', 'Unknown')} on {damage.get('part_name', 'Unknown Part')}"):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write(f"**Severity:** {damage.get('severity', 'Unknown')}")
                            st.write(f"**Confidence:** {damage.get('confidence', 0):.2f}")
                            st.write(f"**Cost:** ₹{damage.get('estimated_cost_inr', 0):,.2f}")
                        
                        with col_b:
                            st.write(f"**Repair Time:** {damage.get('estimated_repair_days', 0)} days")
                            st.write(f"**Images:** {len(damage.get('source_images', []))}")
                            
                            if damage.get('source_images'):
                                st.write("**Source Images:**")
                                for img in damage.get('source_images', []):
                                    st.write(f"- {img}")
        
        with col2:
            st.subheader("📈 Summary")
            
            # Severity distribution
            severity_dist = result.get('overall_severity_distribution', {})
            if severity_dist:
                st.write("**Severity Distribution:**")
                for severity, count in severity_dist.items():
                    st.write(f"- {severity.title()}: {count}")
            
            # Cost breakdown
            st.write("**Cost Breakdown:**")
            st.write(f"- Total: ₹{result.get('total_estimated_cost_inr', 0):,.2f}")
            st.write(f"- Repair Time: {result.get('estimated_repair_time_days', 0)} days")
            
            # Processing info
            processing_info = result.get('processing_metadata', {})
            if processing_info:
                st.write("**Processing Info:**")
                st.write(f"- Images: {processing_info.get('total_images_processed', 0)}")
                st.write(f"- Timestamp: {processing_info.get('consolidation_timestamp', 'Unknown')}")
    else:
        st.warning("No consolidated results available")

def display_intermediate_results():
    """Display intermediate results for individual images"""
    if st.session_state.results_data:
        st.subheader("🔍 Individual Image Results")
        
        # Image selector
        image_names = list(st.session_state.results_data.keys())
        selected_image = st.selectbox("Select an image to view results:", image_names)
        
        if selected_image and selected_image in st.session_state.results_data:
            image_data = st.session_state.results_data[selected_image]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Show the processed image
                image_files = [f for f in image_data['files'] if f['type'] == 'image']
                if image_files:
                    try:
                        image_path = image_files[0]['path']
                        image = Image.open(image_path)
                        st.image(image, caption=selected_image, use_column_width=True)
                    except Exception as e:
                        st.error(f"Error loading image: {str(e)}")
            
            with col2:
                # Show JSON results
                json_files = [f for f in image_data['files'] if f['type'] == 'json']
                
                for json_file in json_files:
                    try:
                        with open(json_file['path'], 'r') as f:
                            json_data = json.load(f)
                        
                        file_type = json_file['name'].replace('.json', '').replace(f'{selected_image}_', '').replace(f'{selected_image}-', '')
                        display_json_data(json_data, f"{file_type.replace('_', ' ').title()} Results")
                        
                    except Exception as e:
                        st.error(f"Error loading {json_file['name']}: {str(e)}")
    else:
        st.warning("No intermediate results available")

def display_raw_data():
    """Display raw JSON data"""
    if st.session_state.consolidated_result:
        st.subheader("📄 Raw Consolidated Data")
        st.json(st.session_state.consolidated_result)
    
    if st.session_state.results_data:
        st.subheader("📁 Raw Intermediate Data")
        
        for image_name, image_data in st.session_state.results_data.items():
            with st.expander(f"Raw data for {image_name}"):
                json_files = [f for f in image_data['files'] if f['type'] == 'json']
                
                for json_file in json_files:
                    try:
                        with open(json_file['path'], 'r') as f:
                            json_data = json.load(f)
                        st.write(f"**{json_file['name']}:**")
                        st.json(json_data)
                    except Exception as e:
                        st.error(f"Error loading {json_file['name']}: {str(e)}")

if __name__ == "__main__":
    main()