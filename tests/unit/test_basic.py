import sys
import os
# Add src to path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.vm import NinthVM

def test_basic_operations():
    vm = NinthVM()
    
    # Test basic math
    result = vm.execute("3 4 [ADD]")
    assert result.item() == 7.0, f"Expected 7.0, got {result.item()}"
    
    # Test tensor creation
    result = vm.execute("[2 4] [ZEROS]")
    assert result.shape == (2, 4), f"Expected shape (2, 4), got {result.shape}"
    
    # Test stack operations
    result = vm.execute("5 [DUP] [ADD]")
    assert result.item() == 10.0, f"Expected 10.0, got {result.item()}"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_basic_operations()