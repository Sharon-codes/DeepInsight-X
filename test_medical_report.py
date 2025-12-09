# test_medical_report.py
"""
Test script to verify medical report generation in Flask app.
"""
import os
import sys
import torch
import numpy as np
from datetime import datetime

# Add the Core directory to Python path
sys.path.append('Core')

# Import the report generation function
from app import generate_detailed_report

def test_medical_report_generation():
    """Test the medical report generation function."""
    print("Testing medical report generation...")
    
    # Test data
    predicted_class_names = ['Pneumonia', 'Cardiomegaly', 'Effusion']
    confidence_scores = {
        'Pneumonia': '0.8756',
        'Cardiomegaly': '0.7234', 
        'Effusion': '0.6543'
    }
    heatmap_filenames = [
        'heatmap_Pneumonia_test_image_20241201_123456.png',
        'heatmap_Cardiomegaly_test_image_20241201_123456.png',
        'heatmap_Effusion_test_image_20241201_123456.png'
    ]
    
    # Generate report
    report_content = generate_detailed_report(
        predicted_class_names, 
        confidence_scores, 
        heatmap_filenames
    )
    
    print("Generated Medical Report:")
    print("=" * 50)
    print(report_content)
    print("=" * 50)
    
    # Verify report content
    assert "Chest X-ray Analysis Report" in report_content
    assert "Predicted Findings:" in report_content
    assert "Pneumonia" in report_content
    assert "Cardiomegaly" in report_content
    assert "Effusion" in report_content
    assert "Visual Explanations" in report_content
    assert "Disclaimer" in report_content
    
    print("✅ Medical report generation test PASSED!")
    
    # Test with no findings
    print("\nTesting with no findings...")
    empty_report = generate_detailed_report([], {}, [])
    
    assert "No specific pathologies detected" in empty_report
    print("✅ Empty findings test PASSED!")
    
    return True

def test_model_loading():
    """Test if the model can be loaded (without actually loading it)."""
    print("\nTesting model loading setup...")
    
    try:
        # Test if we can import the model class
        from utils.model_utils import MultiLabelResNet, TARGET_PATHOLOGIES
        
        print(f"✅ Model class imported successfully")
        print(f"✅ Target pathologies loaded: {len(TARGET_PATHOLOGIES)} classes")
        print(f"   Pathologies: {TARGET_PATHOLOGIES}")
        
        # Test if we can create a model instance (without loading weights)
        model = MultiLabelResNet(num_classes=len(TARGET_PATHOLOGIES), pretrained=False)
        print(f"✅ Model instance created successfully")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test FAILED: {e}")
        return False

def test_grad_cam():
    """Test if Grad-CAM can be imported and initialized."""
    print("\nTesting Grad-CAM setup...")
    
    try:
        from utils.grad_cam import GradCAM
        print("✅ Grad-CAM class imported successfully")
        
        # Test if we can create a Grad-CAM instance (without a real model)
        from utils.model_utils import MultiLabelResNet, TARGET_PATHOLOGIES
        
        # Create a dummy model for testing
        model = MultiLabelResNet(num_classes=len(TARGET_PATHOLOGIES), pretrained=False)
        
        # Test Grad-CAM initialization
        grad_cam = GradCAM(model, 'base_model.layer4')
        print("✅ Grad-CAM instance created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Grad-CAM test FAILED: {e}")
        return False

def test_data_loading():
    """Test if data loading components work."""
    print("\nTesting data loading components...")
    
    try:
        from utils.data_loader import ChestXrayDataset
        from utils.preprocessing import create_transforms, TARGET_IMAGE_SIZE
        print("✅ Data loading components imported successfully")
        
        # Test transform creation
        train_transform = create_transforms(is_train=True)
        val_transform = create_transforms(is_train=False)
        print("✅ Transforms created successfully")
        
        print(f"✅ Target image size: {TARGET_IMAGE_SIZE}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test FAILED: {e}")
        return False

def main():
    """Run all tests."""
    print("Running Flask App Medical Report Tests")
    print("=" * 50)
    
    tests = [
        test_medical_report_generation,
        test_model_loading,
        test_grad_cam,
        test_data_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Flask app medical report generation is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
