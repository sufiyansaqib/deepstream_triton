#!/usr/bin/env python3
"""
Basic ReID Functionality Test
============================

Tests the basic ReID functionality with the enhanced tracker configuration.
Validates that ReID features are being extracted and processed correctly.
"""

import sys
import os
import time
import subprocess
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from tracking.global_track_manager import GlobalTrackManager, Detection
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReIDTester:
    """Test suite for ReID functionality"""
    
    def __init__(self):
        self.test_results = {}
        self.project_root = Path(__file__).parent.parent.parent
    
    def test_global_track_manager(self):
        """Test GlobalTrackManager functionality"""
        logger.info("Testing GlobalTrackManager...")
        
        try:
            # Initialize manager
            manager = GlobalTrackManager(reid_threshold=0.75)
            
            # Create test detections
            detection1 = Detection(
                camera_id=0,
                local_id=1,
                confidence=0.9,
                bbox=[100, 100, 50, 100],
                reid_features=np.random.rand(256),
                class_id=0
            )
            
            detection2 = Detection(
                camera_id=1,
                local_id=1,
                confidence=0.8,
                bbox=[150, 120, 50, 100],
                reid_features=detection1.reid_features + np.random.rand(256) * 0.1,  # Similar features
                class_id=0
            )
            
            # Test track creation and association
            import asyncio
            
            async def run_test():
                global_id1 = await manager.associate_detection(detection1)
                global_id2 = await manager.associate_detection(detection2)
                
                stats = manager.get_track_statistics()
                
                logger.info(f"Created tracks: {global_id1}, {global_id2}")
                logger.info(f"Statistics: {stats}")
                
                return global_id1, global_id2, stats
            
            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                global_id1, global_id2, stats = loop.run_until_complete(run_test())
                
                # Validate results
                if stats['total_global_tracks'] >= 1:
                    self.test_results['global_track_manager'] = True
                    logger.info("✅ GlobalTrackManager test passed")
                else:
                    self.test_results['global_track_manager'] = False
                    logger.error("❌ GlobalTrackManager test failed")
                    
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"❌ GlobalTrackManager test failed with error: {e}")
            self.test_results['global_track_manager'] = False
    
    def test_tracker_configuration(self):
        """Test tracker configuration files"""
        logger.info("Testing tracker configurations...")
        
        try:
            # Check if enhanced tracker config exists
            enhanced_config = self.project_root / "configs/reid/tracker_reid_enhanced.yml"
            deepstream_config = self.project_root / "configs/reid/deepstream_reid_enabled.txt"
            
            configs_exist = enhanced_config.exists() and deepstream_config.exists()
            
            if configs_exist:
                # Validate YAML syntax
                import yaml
                with open(enhanced_config, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Check critical ReID settings
                reid_enabled = (config_data.get('ReID', {}).get('reidType') == 2)
                reassoc_enabled = (config_data.get('TrajectoryManagement', {}).get('enableReAssoc') == 1)
                
                if reid_enabled and reassoc_enabled:
                    self.test_results['tracker_configuration'] = True
                    logger.info("✅ Tracker configuration test passed")
                else:
                    self.test_results['tracker_configuration'] = False
                    logger.error("❌ Tracker configuration missing critical ReID settings")
            else:
                self.test_results['tracker_configuration'] = False
                logger.error("❌ Tracker configuration files not found")
                
        except Exception as e:
            logger.error(f"❌ Tracker configuration test failed: {e}")
            self.test_results['tracker_configuration'] = False
    
    def test_deepstream_pipeline_syntax(self):
        """Test DeepStream pipeline configuration syntax"""
        logger.info("Testing DeepStream pipeline syntax...")
        
        try:
            config_file = self.project_root / "configs/reid/deepstream_reid_enabled.txt"
            
            if not config_file.exists():
                self.test_results['pipeline_syntax'] = False
                logger.error("❌ DeepStream config file not found")
                return
            
            # Basic syntax validation
            with open(config_file, 'r') as f:
                content = f.read()
            
            # Check for required sections
            required_sections = ['[application]', '[source0]', '[source1]', '[tracker]', '[primary-gie]']
            missing_sections = [section for section in required_sections if section not in content]
            
            # Check for ReID-specific configuration
            reid_config_path = "configs/reid/tracker_reid_enhanced.yml"
            has_reid_config = reid_config_path in content
            
            if not missing_sections and has_reid_config:
                self.test_results['pipeline_syntax'] = True
                logger.info("✅ DeepStream pipeline syntax test passed")
            else:
                self.test_results['pipeline_syntax'] = False
                logger.error(f"❌ Pipeline syntax issues: missing sections {missing_sections}, ReID config: {has_reid_config}")
                
        except Exception as e:
            logger.error(f"❌ Pipeline syntax test failed: {e}")
            self.test_results['pipeline_syntax'] = False
    
    def test_model_structure(self):
        """Test ReID model directory structure"""
        logger.info("Testing ReID model structure...")
        
        try:
            model_dir = self.project_root / "models/reid/reid_model"
            model_file = model_dir / "1/model.etlt"
            config_file = model_dir / "config.pbtxt"
            
            structure_valid = model_dir.exists() and model_file.exists() and config_file.exists()
            
            if structure_valid:
                # Check model file size (should be substantial)
                model_size = model_file.stat().st_size
                size_valid = model_size > 1024 * 1024  # At least 1MB
                
                if size_valid:
                    self.test_results['model_structure'] = True
                    logger.info(f"✅ Model structure test passed (model size: {model_size / 1024 / 1024:.1f}MB)")
                else:
                    self.test_results['model_structure'] = False
                    logger.error(f"❌ Model file too small: {model_size} bytes")
            else:
                self.test_results['model_structure'] = False
                logger.error("❌ Model structure incomplete")
                
        except Exception as e:
            logger.error(f"❌ Model structure test failed: {e}")
            self.test_results['model_structure'] = False
    
    def run_all_tests(self):
        """Run all ReID tests"""
        logger.info("🚀 Starting ReID functionality tests...")
        logger.info("=" * 50)
        
        # Run individual tests
        self.test_global_track_manager()
        self.test_tracker_configuration()
        self.test_deepstream_pipeline_syntax()
        self.test_model_structure()
        
        # Summary
        logger.info("=" * 50)
        logger.info("📊 Test Results Summary:")
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")
        
        logger.info(f"\n📈 Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All ReID tests passed! Ready for DeepStream integration.")
            return True
        else:
            logger.warning(f"⚠️  {total_tests - passed_tests} tests failed. Review configuration before proceeding.")
            return False

def main():
    """Main test execution"""
    tester = ReIDTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎯 Next steps:")
        print("1. Run DeepStream pipeline with ReID: ./scripts/reid/test_reid_tracking.sh")
        print("2. Monitor cross-camera associations in logs")
        print("3. Validate tracking accuracy with test videos")
    else:
        print("\n🔧 Fix the failing tests before proceeding with DeepStream integration.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())