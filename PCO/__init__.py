"""
PCO - Point Cloud Octree file format library

A clean, modular library for reading and writing octree-indexed point cloud files.

Quick Start:
    from PCO import PCOReader, PCOWriter
    
    # Writing
    writer = PCOWriter()
    writer.write('output.pco', points, root, colors=colors)
    
    # Reading
    reader = PCOReader('input.pco')
    reader.load_metadata()
    data = reader.read_node('r0')
"""

# Main classes - most common imports
from .pco_reader import PCOReader
from .pco_writer import PCOWriter

# Format utilities - for advanced usage
from .pco_format import PCOFormat, OctreeUtils, DoubleBufferMerge, printf, printd

# Version info
__version__ = '2.0.0'
__all__ = ['PCOReader', 'PCOWriter', 'PCOFormat', 'OctreeUtils', 'DoubleBufferMerge', 'printf', 'printd']
