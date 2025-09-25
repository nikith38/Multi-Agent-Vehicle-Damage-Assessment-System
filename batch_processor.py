#!/usr/bin/env python3
"""
Batch Processor for Zoop Main Pipeline

This script processes multiple images by:
1. Moving images one by one from raw_images to tests folder
2. Running the pipeline on each image
3. Archiving results to timestamped folders
4. Cleaning up between runs

Usage:
    python batch_processor.py
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import glob
import json
import time
import logging

class BatchProcessor:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.raw_images_dir = self.base_dir / "raw_images"
        self.tests_dir = self.base_dir / "tests"
        self.results_dir = self.base_dir / "results"
        self.enhanced_dir = self.base_dir / "enhanced"
        self.pipeline_script = self.base_dir / "run_pipeline.py"
        
        # Supported image extensions
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # Setup logging
        self.setup_logging()
        self.logger.info("BatchProcessor initialized")
        self.logger.debug(f"Base directory: {self.base_dir}")
        self.logger.debug(f"Raw images directory: {self.raw_images_dir}")
        self.logger.debug(f"Tests directory: {self.tests_dir}")
        self.logger.debug(f"Results directory: {self.results_dir}")
        self.logger.debug(f"Enhanced directory: {self.enhanced_dir}")
        self.logger.debug(f"Pipeline script: {self.pipeline_script}")
        
    def setup_logging(self):
        """Setup comprehensive logging for batch processing"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"batch_processing_{timestamp}.log"
        
        # Create logger
        self.logger = logging.getLogger('BatchProcessor')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.log_file = log_file
        self.logger.info(f"Logging initialized. Log file: {log_file}")
        
    def setup_directories(self):
        """Ensure all required directories exist"""
        self.logger.info("Setting up directories...")
        
        directories = [
            ("raw_images", self.raw_images_dir),
            ("tests", self.tests_dir),
            ("results", self.results_dir)
        ]
        
        for name, directory in directories:
            self.logger.debug(f"Checking {name} directory: {directory}")
            if directory.exists():
                self.logger.debug(f"{name} directory already exists")
            else:
                self.logger.info(f"Creating {name} directory: {directory}")
            directory.mkdir(exist_ok=True)
            self.logger.debug(f"{name} directory ready: {directory}")
        
        self.logger.info("All directories setup complete")
        
    def get_images_to_process(self):
        """Get list of images in raw_images directory"""
        self.logger.info("Discovering images to process...")
        self.logger.debug(f"Searching in directory: {self.raw_images_dir}")
        self.logger.debug(f"Supported extensions: {self.image_extensions}")
        
        images = []
        for ext in self.image_extensions:
            # Use case-insensitive glob pattern to avoid duplicates
            pattern = str(self.raw_images_dir / f"*{ext}")
            self.logger.debug(f"Searching pattern: {pattern}")
            found_images = glob.glob(pattern)
            self.logger.debug(f"Found {len(found_images)} images with extension {ext}")
            images.extend(found_images)
        
        # Remove duplicates and sort
        unique_images = sorted(list(set(images)))
        self.logger.info(f"Total unique images found: {len(unique_images)}")
        
        for i, img in enumerate(unique_images, 1):
            self.logger.debug(f"Image {i}: {Path(img).name} ({Path(img).stat().st_size} bytes)")
        
        return unique_images
    
    def clean_tests_directory(self):
        """Clean the tests directory before processing next image"""
        self.logger.debug("Cleaning tests directory...")
        self.logger.debug(f"Tests directory path: {self.tests_dir}")
        
        if not self.tests_dir.exists():
            self.logger.debug("Tests directory does not exist, nothing to clean")
            return
            
        items_removed = 0
        for item in self.tests_dir.iterdir():
            self.logger.debug(f"Removing from tests: {item.name} ({'file' if item.is_file() else 'directory'})")
            if item.is_file():
                item.unlink()
                items_removed += 1
            elif item.is_dir():
                shutil.rmtree(item)
                items_removed += 1
        
        self.logger.debug(f"Tests directory cleaned: {items_removed} items removed")
    
    def clean_enhanced_directory(self):
        """Clean the enhanced directory before processing next image"""
        self.logger.debug("Cleaning enhanced directory...")
        self.logger.debug(f"Enhanced directory path: {self.enhanced_dir}")
        
        if not self.enhanced_dir.exists():
            self.logger.debug("Enhanced directory does not exist, nothing to clean")
            return
            
        items_removed = 0
        for item in self.enhanced_dir.iterdir():
            self.logger.debug(f"Removing from enhanced: {item.name} ({'file' if item.is_file() else 'directory'})")
            if item.is_file():
                item.unlink()
                items_removed += 1
            elif item.is_dir():
                shutil.rmtree(item)
                items_removed += 1
        
        self.logger.debug(f"Enhanced directory cleaned: {items_removed} items removed")
    
    def move_image_to_tests(self, image_path):
        """Move image from raw_images to tests directory"""
        self.logger.info(f"Moving image to tests directory: {Path(image_path).name}")
        self.logger.debug(f"Source path: {image_path}")
        
        image_file = Path(image_path)
        destination = self.tests_dir / image_file.name
        self.logger.debug(f"Destination path: {destination}")
        
        # Verify source file exists
        if not image_file.exists():
            self.logger.error(f"Source image file does not exist: {image_path}")
            raise FileNotFoundError(f"Source image file not found: {image_path}")
        
        # Get file size before move
        file_size = image_file.stat().st_size
        self.logger.debug(f"Source file size: {file_size} bytes")
        
        # Clean both directories first
        self.logger.debug("Cleaning directories before move...")
        self.clean_tests_directory()
        self.clean_enhanced_directory()
        
        # Move the image
        self.logger.debug(f"Moving file from {image_file} to {destination}")
        shutil.move(str(image_file), str(destination))
        
        # Verify move was successful
        if destination.exists():
            moved_size = destination.stat().st_size
            self.logger.debug(f"Move successful. Destination file size: {moved_size} bytes")
            if moved_size != file_size:
                self.logger.warning(f"File size mismatch after move: {file_size} -> {moved_size}")
        else:
            self.logger.error(f"Move failed - destination file does not exist: {destination}")
            raise FileNotFoundError(f"Move failed - destination not found: {destination}")
        
        self.logger.info(f"Image successfully moved to tests directory: {destination.name}")
        return destination
    
    def run_pipeline(self):
        """Run the pipeline on the current image in tests directory"""
        self.logger.info("Starting pipeline execution...")
        self.logger.debug(f"Pipeline script: {self.pipeline_script}")
        self.logger.debug(f"Working directory: {self.base_dir}")
        self.logger.debug(f"Python executable: {sys.executable}")
        
        # Check if pipeline script exists
        if not self.pipeline_script.exists():
            self.logger.error(f"Pipeline script not found: {self.pipeline_script}")
            return False, "", f"Pipeline script not found: {self.pipeline_script}"
        
        # Log current directory contents before pipeline
        self.logger.debug("Directory contents before pipeline execution:")
        if self.tests_dir.exists():
            test_files = list(self.tests_dir.iterdir())
            self.logger.debug(f"Tests directory: {len(test_files)} items")
            for item in test_files:
                self.logger.debug(f"  - {item.name} ({'file' if item.is_file() else 'directory'})")
        else:
            self.logger.debug("Tests directory does not exist")
            
        if self.enhanced_dir.exists():
            enhanced_files = list(self.enhanced_dir.iterdir())
            self.logger.debug(f"Enhanced directory: {len(enhanced_files)} items")
            for item in enhanced_files:
                self.logger.debug(f"  - {item.name} ({'file' if item.is_file() else 'directory'})")
        else:
            self.logger.debug("Enhanced directory does not exist")
        
        try:
            self.logger.debug("Executing pipeline subprocess...")
            start_time = time.time()
            
            result = subprocess.run(
                [sys.executable, str(self.pipeline_script)],
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            execution_time = time.time() - start_time
            self.logger.debug(f"Pipeline execution completed in {execution_time:.2f} seconds")
            self.logger.debug(f"Return code: {result.returncode}")
            
            # Log stdout and stderr
            if result.stdout:
                self.logger.debug("Pipeline stdout:")
                for line in result.stdout.strip().split('\n'):
                    self.logger.debug(f"  STDOUT: {line}")
            else:
                self.logger.debug("Pipeline stdout: (empty)")
                
            if result.stderr:
                self.logger.debug("Pipeline stderr:")
                for line in result.stderr.strip().split('\n'):
                    self.logger.debug(f"  STDERR: {line}")
            else:
                self.logger.debug("Pipeline stderr: (empty)")
            
            success = result.returncode == 0
            self.logger.info(f"Pipeline execution {'successful' if success else 'failed'}")
            
            # Log directory contents after pipeline
            self.logger.debug("Directory contents after pipeline execution:")
            if self.tests_dir.exists():
                test_files = list(self.tests_dir.iterdir())
                self.logger.debug(f"Tests directory: {len(test_files)} items")
                for item in test_files:
                    self.logger.debug(f"  - {item.name} ({'file' if item.is_file() else 'directory'})")
            
            if self.enhanced_dir.exists():
                enhanced_files = list(self.enhanced_dir.iterdir())
                self.logger.debug(f"Enhanced directory: {len(enhanced_files)} items")
                for item in enhanced_files:
                    self.logger.debug(f"  - {item.name} ({'file' if item.is_file() else 'directory'})")
            
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            self.logger.error("Pipeline execution timed out after 5 minutes")
            return False, "", "Pipeline execution timed out after 5 minutes"
        except Exception as e:
            self.logger.error(f"Error running pipeline: {str(e)}")
            return False, "", f"Error running pipeline: {str(e)}"
    
    def archive_results(self, image_name, timestamp):
        """Archive results to timestamped folder"""
        self.logger.info(f"Starting result archiving for image: {image_name}")
        self.logger.debug(f"Timestamp: {timestamp}")
        
        # Create timestamped result folder
        result_folder = self.results_dir / f"{timestamp}_{Path(image_name).stem}"
        self.logger.debug(f"Creating result folder: {result_folder}")
        result_folder.mkdir(exist_ok=True)
        
        # Create enhanced subfolder within the result folder
        enhanced_subfolder = result_folder / "enhanced"
        self.logger.debug(f"Creating enhanced subfolder: {enhanced_subfolder}")
        enhanced_subfolder.mkdir(exist_ok=True)
        
        # Move all files from tests directory to result folder
        archived_files = []
        self.logger.debug(f"Archiving files from tests directory: {self.tests_dir}")
        
        if not self.tests_dir.exists():
            self.logger.warning("Tests directory does not exist during archiving")
        else:
            test_files = list(self.tests_dir.iterdir())
            self.logger.debug(f"Found {len(test_files)} items in tests directory")
            
            for item in test_files:
                if item.is_file():
                    self.logger.debug(f"Archiving from tests: {item.name} ({item.stat().st_size} bytes)")
                    destination = result_folder / item.name
                    shutil.move(str(item), str(destination))
                    archived_files.append(item.name)
                    self.logger.debug(f"Successfully moved to: {destination}")
                else:
                    self.logger.debug(f"Skipping directory in tests: {item.name}")
        
        # Move all files from enhanced directory to enhanced subfolder
        self.logger.debug(f"Archiving files from enhanced directory: {self.enhanced_dir}")
        
        if not self.enhanced_dir.exists():
            self.logger.warning("Enhanced directory does not exist during archiving")
        else:
            enhanced_files = list(self.enhanced_dir.iterdir())
            self.logger.debug(f"Found {len(enhanced_files)} items in enhanced directory")
            
            for item in enhanced_files:
                if item.is_file():
                    self.logger.debug(f"Archiving from enhanced: {item.name} ({item.stat().st_size} bytes)")
                    destination = enhanced_subfolder / item.name
                    shutil.move(str(item), str(destination))
                    archived_files.append(f"enhanced/{item.name}")
                    self.logger.debug(f"Successfully moved to: {destination}")
                else:
                    self.logger.debug(f"Skipping directory in enhanced: {item.name}")
        
        self.logger.info(f"Archiving complete. Total files archived: {len(archived_files)}")
        self.logger.debug(f"Archived files: {archived_files}")
        
        return result_folder, archived_files
    
    def create_batch_summary(self, results):
        """Create a summary of the batch processing results"""
        self.logger.info("Creating batch processing summary...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.results_dir / f"batch_summary_{timestamp}.json"
        self.logger.debug(f"Summary file path: {summary_file}")
        
        successful_count = sum(1 for r in results if r["success"])
        failed_count = sum(1 for r in results if not r["success"])
        
        summary = {
            "batch_timestamp": timestamp,
            "total_images": len(results),
            "successful": successful_count,
            "failed": failed_count,
            "results": results
        }
        
        self.logger.debug(f"Summary statistics - Total: {len(results)}, Successful: {successful_count}, Failed: {failed_count}")
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"Batch summary saved to: {summary_file.name}")
        except Exception as e:
            self.logger.error(f"Failed to save batch summary: {str(e)}")
            raise
        
        return summary_file
    
    def process_batch(self):
        """Process all images in the raw_images directory"""
        self.logger.info("Starting batch processing session...")
        self.logger.info("=" * 50)
        
        # Setup directories
        self.logger.info("Setting up directories...")
        self.setup_directories()
        
        # Get images to process
        self.logger.info("Discovering images to process...")
        images = self.get_images_to_process()
        
        if not images:
            self.logger.warning("No images found in raw_images directory!")
            self.logger.info(f"Please place images in: {self.raw_images_dir}")
            print("No images found in raw_images directory!")
            print(f"Please place images in: {self.raw_images_dir}")
            return
        
        self.logger.info(f"Found {len(images)} images to process:")
        print(f"Found {len(images)} images to process:")
        for i, img in enumerate(images, 1):
            image_name = Path(img).name
            self.logger.info(f"   {i}. {image_name}")
            print(f"   {i}. {image_name}")
        print()
        
        # Process each image
        results = []
        start_time = time.time()
        self.logger.info(f"Beginning batch processing of {len(images)} images...")
        
        for i, image_path in enumerate(images, 1):
            image_name = Path(image_path).name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            self.logger.info(f"Processing image {i}/{len(images)}: {image_name}")
            self.logger.info("-" * 30)
            print(f"Processing {i}/{len(images)}: {image_name}")
            print("-" * 30)
            
            try:
                # Move image to tests directory
                self.logger.info("Step 1: Moving image to tests directory...")
                print("Moving image to tests directory...")
                test_image_path = self.move_image_to_tests(image_path)
                
                # Run pipeline
                self.logger.info("Step 2: Running pipeline...")
                print("Running pipeline...")
                success, stdout, stderr = self.run_pipeline()
                
                if success:
                    self.logger.info("Pipeline completed successfully!")
                    print("Pipeline completed successfully!")
                    
                    # Wait for all files to be written to disk
                    self.logger.info("Step 3: Waiting for file system sync...")
                    print("Waiting for file system sync...")
                    time.sleep(5)
                    
                    # Archive results
                    self.logger.info("Step 4: Archiving results...")
                    print("Archiving results...")
                    result_folder, archived_files = self.archive_results(image_name, timestamp)
                    
                    self.logger.info(f"Results archived to: {result_folder.name}")
                    self.logger.info(f"Files archived: {len(archived_files)}")
                    print(f"Results archived to: {result_folder.name}")
                    print(f"Files archived: {len(archived_files)}")
                    
                    # Remove processed image from raw_images
                    self.logger.info("Step 5: Cleaning up processed image...")
                    self.logger.debug(f"Attempting to remove: {image_path}")
                    try:
                        os.remove(image_path)
                        self.logger.info(f"Successfully removed processed image: {image_name}")
                    except Exception as cleanup_error:
                        self.logger.error(f"Failed to remove processed image {image_name}: {str(cleanup_error)}")
                        # Don't fail the entire process for cleanup issues
                    
                    results.append({
                        "image": image_name,
                        "success": True,
                        "timestamp": timestamp,
                        "result_folder": result_folder.name,
                        "archived_files": archived_files,
                        "error": None
                    })
                    
                else:
                    self.logger.error("Pipeline failed!")
                    self.logger.error(f"Pipeline error: {stderr}")
                    print("Pipeline failed!")
                    print(f"Error: {stderr}")
                    
                    results.append({
                        "image": image_name,
                        "success": False,
                        "timestamp": timestamp,
                        "result_folder": None,
                        "archived_files": [],
                        "error": stderr
                    })
                
            except Exception as e:
                self.logger.error(f"Error processing {image_name}: {str(e)}")
                print(f"Error processing {image_name}: {str(e)}")
                results.append({
                    "image": image_name,
                    "success": False,
                    "timestamp": timestamp,
                    "result_folder": None,
                    "archived_files": [],
                    "error": str(e)
                })
            
            self.logger.info(f"Completed processing image {i}/{len(images)}: {image_name}")
            print()
        
        # Create batch summary
        total_time = time.time() - start_time
        self.logger.info(f"Batch processing completed in {total_time:.2f} seconds")
        summary_file = self.create_batch_summary(results)
        
        # Print final summary
        successful_count = sum(1 for r in results if r['success'])
        failed_count = sum(1 for r in results if not r['success'])
        
        self.logger.info("Batch Processing Complete!")
        self.logger.info("=" * 50)
        self.logger.info(f"Total time: {total_time:.2f} seconds")
        self.logger.info(f"Total images: {len(results)}")
        self.logger.info(f"Successful: {successful_count}")
        self.logger.info(f"Failed: {failed_count}")
        self.logger.info(f"Summary saved to: {summary_file.name}")
        
        print("Batch Processing Complete!")
        print("=" * 50)
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Total images: {len(results)}")
        print(f"Successful: {successful_count}")
        print(f"Failed: {failed_count}")
        print(f"Summary saved to: {summary_file.name}")
        print()
        
        # Show failed images if any
        failed_images = [r for r in results if not r["success"]]
        if failed_images:
            self.logger.warning(f"Found {len(failed_images)} failed images:")
            print("Failed Images:")
            for result in failed_images:
                self.logger.warning(f"   • {result['image']}: {result['error']}")
                print(f"   • {result['image']}: {result['error']}")
        
        self.logger.info(f"Results available in: {self.results_dir}")
        self.logger.info(f"Detailed logs saved to: {self.log_file}")
        print(f"Results available in: {self.results_dir}")
        print(f"Detailed logs saved to: {self.log_file}")

def main():
    """Main entry point"""
    processor = BatchProcessor()
    processor.process_batch()

if __name__ == "__main__":
    main()