"""
Pytest configuration and fixtures
"""

import sys
import os

# Add src to Python path for all tests
tests_dir = os.path.dirname(__file__)
project_root = os.path.dirname(tests_dir)
src_path = os.path.join(project_root, "src")
src_path = os.path.abspath(src_path)

if src_path not in sys.path:
    sys.path.insert(0, src_path)
