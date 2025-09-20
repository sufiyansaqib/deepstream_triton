#!/usr/bin/env python3
"""
ReID Configuration Validation Test
=================================

Tests the ReID configuration files without requiring external dependencies.
Validates file structure, configuration syntax, and ReID settings.
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReIDConfigTester:
    """Test suite for ReID configuration validation"""
    
    def __init__(self):
        self.test_results = {}
        self.project_root = Path(__file__).parent.parent.parent
        logger.info(f"Project root: {self.project_root}")
    
    def test_directory_structure(self):
        """Test ReID directory structure"""
        logger.info("Testing ReID directory structure...")
        
        try:
            required_dirs = [
                "configs/reid",
                "models/reid", 
                "scripts/reid",
                "src/tracking",
                "tests/reid"
            ]
            
            missing_dirs = []
            for dir_path in required_dirs:
                full_path = self.project_root / dir_path
                if not full_path.exists():
                    missing_dirs.append(dir_path)
            
            if not missing_dirs:
                self.test_results['directory_structure'] = True
                logger.info("✅ Directory structure test passed")
            else:
                self.test_results['directory_structure'] = False
                logger.error(f"❌ Missing directories: {missing_dirs}")
                
        except Exception as e:
            logger.error(f"❌ Directory structure test failed: {e}")
            self.test_results['directory_structure'] = False
    
    def test_tracker_configuration(self):
        """Test tracker configuration files"""
        logger.info("Testing tracker configurations...")
        
        try:
            # Check if enhanced tracker config exists
            enhanced_config = self.project_root / "configs/reid/tracker_reid_enhanced.yml"
            deepstream_config = self.project_root / "configs/reid/deepstream_reid_enabled.txt"
            
            configs_exist = enhanced_config.exists() and deepstream_config.exists()
            
            if configs_exist:
                # Basic content validation
                with open(enhanced_config, 'r') as f:
                    yaml_content = f.read()
                
                # Check for critical ReID settings
                reid_checks = [
                    'reidType: 2' in yaml_content,
                    'enableReAssoc: 1' in yaml_content,
                    'reidExtractionInterval: 0' in yaml_content,
                    'addFeatureNormalization: 1' in yaml_content,
                    'featureMatchingAlgorithm: 1' in yaml_content
                ]
                
                all_reid_settings = all(reid_checks)
                
                if all_reid_settings:
                    self.test_results['tracker_configuration'] = True
                    logger.info("✅ Tracker configuration test passed")
                    logger.info(f"  - Enhanced config: {enhanced_config.name}")
                    logger.info(f"  - DeepStream config: {deepstream_config.name}")
                else:
                    self.test_results['tracker_configuration'] = False
                    logger.error("❌ Tracker configuration missing critical ReID settings")
                    logger.error(f"  - ReID checks passed: {sum(reid_checks)}/{len(reid_checks)}")
            else:
                self.test_results['tracker_configuration'] = False
                logger.error("❌ Tracker configuration files not found")
                if not enhanced_config.exists():
                    logger.error(f"  - Missing: {enhanced_config}")
                if not deepstream_config.exists():
                    logger.error(f"  - Missing: {deepstream_config}")
                
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
            required_sections = [
                '[application]', 
                '[source0]', 
                '[source1]', 
                '[tracker]', 
                '[primary-gie]',
                '[streammux]',
                '[sink0]',
                '[sink1]'
            ]
            
            missing_sections = [section for section in required_sections if section not in content]
            
            # Check for ReID-specific configuration
            reid_config_path = "configs/reid/tracker_reid_enhanced.yml"
            has_reid_config = reid_config_path in content
            
            # Check for proper file paths
            proper_paths = [
                '/workspace/configs/reid/tracker_reid_enhanced.yml' in content,
                '/workspace/configs/config_infer_triton.txt' in content
            ]
            
            if not missing_sections and has_reid_config and all(proper_paths):
                self.test_results['pipeline_syntax'] = True
                logger.info("✅ DeepStream pipeline syntax test passed")
                logger.info(f"  - All required sections present")
                logger.info(f"  - ReID tracker configuration linked")
                logger.info(f"  - Proper file paths configured")
            else:
                self.test_results['pipeline_syntax'] = False
                logger.error("❌ Pipeline syntax issues:")
                if missing_sections:
                    logger.error(f"  - Missing sections: {missing_sections}")
                if not has_reid_config:
                    logger.error(f"  - ReID config not linked")
                if not all(proper_paths):
                    logger.error(f"  - Improper file paths")
                
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
                
                # Check config file content
                with open(config_file, 'r') as f:
                    config_content = f.read()
                
                config_checks = [
                    'name: "reid_model"' in config_content,
                    'platform: "tensorrt_plan"' in config_content,
                    'dims: [ 3, 256, 128 ]' in config_content,
                    'dims: [ 256 ]' in config_content
                ]
                
                config_valid = all(config_checks)
                
                if size_valid and config_valid:
                    self.test_results['model_structure'] = True
                    logger.info(f"✅ Model structure test passed")
                    logger.info(f"  - Model size: {model_size / 1024 / 1024:.1f}MB")
                    logger.info(f"  - Config validation: {sum(config_checks)}/{len(config_checks)} checks passed")
                else:
                    self.test_results['model_structure'] = False
                    if not size_valid:
                        logger.error(f"❌ Model file too small: {model_size} bytes")
                    if not config_valid:
                        logger.error(f"❌ Model config validation failed: {sum(config_checks)}/{len(config_checks)}")
            else:
                self.test_results['model_structure'] = False
                logger.error("❌ Model structure incomplete:")
                if not model_dir.exists():
                    logger.error(f"  - Missing directory: {model_dir}")
                if not model_file.exists():
                    logger.error(f"  - Missing model file: {model_file}")
                if not config_file.exists():
                    logger.error(f"  - Missing config file: {config_file}")
                
        except Exception as e:
            logger.error(f"❌ Model structure test failed: {e}")
            self.test_results['model_structure'] = False
    
    def test_script_structure(self):
        """Test ReID script structure"""
        logger.info("Testing ReID script structure...")
        
        try:
            scripts_dir = self.project_root / "scripts/reid"
            
            required_scripts = [
                "convert_reid_model.sh",
                "test_reid_tracking.sh"
            ]
            
            missing_scripts = []
            executable_issues = []
            
            for script_name in required_scripts:
                script_path = scripts_dir / script_name
                if not script_path.exists():
                    missing_scripts.append(script_name)
                else:
                    # Check if executable
                    if not os.access(script_path, os.X_OK):
                        executable_issues.append(script_name)
            
            if not missing_scripts and not executable_issues:
                self.test_results['script_structure'] = True
                logger.info("✅ Script structure test passed")
                logger.info(f"  - All scripts present and executable")
            else:
                self.test_results['script_structure'] = False
                if missing_scripts:
                    logger.error(f"❌ Missing scripts: {missing_scripts}")
                if executable_issues:
                    logger.error(f"❌ Non-executable scripts: {executable_issues}")
                
        except Exception as e:
            logger.error(f"❌ Script structure test failed: {e}")
            self.test_results['script_structure'] = False
    
    def run_all_tests(self):
        """Run all ReID configuration tests"""
        logger.info("🚀 Starting ReID configuration validation...")
        logger.info("=" * 60)
        
        # Run individual tests
        self.test_directory_structure()
        self.test_tracker_configuration()
        self.test_deepstream_pipeline_syntax()
        self.test_model_structure()
        self.test_script_structure()
        
        # Summary
        logger.info("=" * 60)
        logger.info("📊 Configuration Validation Results:")
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {test_name.replace('_', ' ').title()}: {status}")
        
        success_rate = (passed_tests / total_tests) * 100
        logger.info(f"\n📈 Overall Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if passed_tests == total_tests:
            logger.info("🎉 All configuration tests passed! ReID setup is ready.")
            return True
        else:
            logger.warning(f"⚠️  {total_tests - passed_tests} tests failed. Review configuration.")
            return False

def main():
    """Main test execution"""
    tester = ReIDConfigTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎯 Configuration validated successfully!")
        print("Next steps:")
        print("1. Test ReID tracking: ./scripts/reid/test_reid_tracking.sh")
        print("2. Convert ReID model: ./scripts/reid/convert_reid_model.sh")
        print("3. Monitor performance and validate cross-camera associations")
    else:
        print("\n🔧 Fix the configuration issues before proceeding.")
        print("Check the logs above for specific problems to address.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())