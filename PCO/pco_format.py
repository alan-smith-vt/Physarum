"""
PCO Format Specification and Shared Utilities

This module defines the binary format specification for Point Cloud Octree (PCO) files
and provides shared utilities used by both readers and writers.
"""

import numpy as np
import struct
from datetime import datetime
import threading
from multiprocessing import shared_memory
import pickle

class PCOFormat:
    """
    Defines the PCO binary file format specification.
    
    File Structure:
    - Header (44 bytes): magic number, metadata, schema
    - Data section: interleaved point data organized by octree nodes
    - Index section: lookup table for node locations
    """
    
    # Magic number for file format validation
    MAGIC = b'PCLOUD01'
    HEADER_SIZE = 56
    
    # Schema field definitions (bitmask)
    FIELD_XYZ = 0b0001
    FIELD_RGB = 0b0010
    FIELD_INTENSITY = 0b0100
    FIELD_NORMAL = 0b1000
    
    # Field sizes: (count, struct_format)
    FIELD_SIZES = {
        FIELD_XYZ: (3, 'f'),        # 3 floats (x, y, z)
        FIELD_RGB: (3, 'B'),         # 3 unsigned bytes (r, g, b)
        FIELD_INTENSITY: (1, 'f'),   # 1 float
        FIELD_NORMAL: (3, 'f'),      # 3 floats (nx, ny, nz)
    }
    
    # Index entry structure
    INDEX_ENTRY_SIZE = 24 + 8 + 4  # node_id (24 bytes) + offset (8) + count (4)
    NODE_ID_MAX_LENGTH = 24

    @staticmethod
    def read_header(filename):
        """
        Read header information from a PCO file.
        
        Args:
            filename: path to PCO file
        
        Returns:
            dict with header fields: max_depth, root_min, root_size, total_points, 
            schema, bytes_per_point, index_offset
        """
        with open(filename, 'rb') as f:
            magic = f.read(8)
            if magic != PCOFormat.MAGIC:
                raise ValueError(f"Invalid file format. Expected {PCOFormat.MAGIC}, got {magic}")
            
            max_depth = struct.unpack('H', f.read(2))[0]
            root_min = struct.unpack('ddd', f.read(24))
            root_size = struct.unpack('f', f.read(4))[0]
            total_points = struct.unpack('Q', f.read(8))[0]
            schema = struct.unpack('B', f.read(1))[0]
            bytes_per_point = struct.unpack('B', f.read(1))[0]
            index_offset = struct.unpack('Q', f.read(8))[0]
    
        return {
            'max_depth': max_depth,
            'root_min': root_min,
            'root_size': root_size,
            'total_points': total_points,
            'schema': schema,
            'bytes_per_point': bytes_per_point,
            'index_offset': index_offset
        }
    
    @staticmethod
    def get_dtype(schema):
        """Build numpy dtype from schema bitmask"""
        fields = []
        if schema & PCOFormat.FIELD_XYZ:
            fields.extend([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        if schema & PCOFormat.FIELD_RGB:
            fields.extend([('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
        if schema & PCOFormat.FIELD_INTENSITY:
            fields.append(('intensity', 'f4'))
        if schema & PCOFormat.FIELD_NORMAL:
            fields.extend([('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
        return np.dtype(fields)
    
    @staticmethod
    def calculate_bytes_per_point(schema):
        """Calculate total bytes per point from schema"""
        total = 0
        for field_flag, (count, dtype) in PCOFormat.FIELD_SIZES.items():
            if schema & field_flag:
                total += count * (4 if dtype == 'f' else 1)
        return total
    
    @staticmethod
    def infer_schema(colors=None, intensities=None, normals=None):
        """Infer schema bitmask from provided data"""
        schema = PCOFormat.FIELD_XYZ  # Always have XYZ
        if colors is not None:
            schema |= PCOFormat.FIELD_RGB
        if intensities is not None:
            schema |= PCOFormat.FIELD_INTENSITY
        if normals is not None:
            schema |= PCOFormat.FIELD_NORMAL
        return schema

    @staticmethod
    def build_merge_index(input_files):
        """
        Read metadata from multiple PCO files and build merge index structure.
        
        Args:
            input_files: list of PCO file paths to merge
        
        Returns:
            tuple: (merged_indices, final_indices, metadata_dict)
                - merged_indices: list of (node_id, [(file, offset, count), ...])
                - final_indices: dict of {node_id: (byte_position, point_count)}
                - metadata_dict: {'bytes_per_point', 'root_min', 'root_size', 
                                 'max_depth', 'total_points', 'headers'}
        """
        from collections import defaultdict
        from tqdm import tqdm
        
        merged_indices = defaultdict(list)
        headers = []
        total_points = 0
        
        # Read all metadata
        for input_file in tqdm(input_files, desc="Reading metadata"):
            header = PCOFormat.read_header(input_file)
            #print(f"Header in dbm {header}")
            
            # Read index
            index = {}
            with open(input_file, 'rb') as f:
                f.seek(header['index_offset'])
                num_nodes = struct.unpack('I', f.read(4))[0]
                for _ in range(num_nodes):
                    node_id = f.read(24).decode('ascii').rstrip('\0')
                    offset = struct.unpack('Q', f.read(8))[0]
                    count = struct.unpack('I', f.read(4))[0]
                    index[node_id] = (offset, count)
            
            headers.append(header)
            total_points += header['total_points']
            
            # Collect node references
            for node_id, (offset, point_count) in index.items():
                merged_indices[node_id].append((input_file, offset, point_count))
        
        # Build final index
        merged_indices_list = list(merged_indices.items())
        final_indices = {}
        current_position = PCOFormat.HEADER_SIZE
        bytes_per_point = headers[0]['bytes_per_point']
        
        for node_id, offsets in tqdm(merged_indices_list, desc="Building index"):
            total_point_count = sum(point_count for _, _, point_count in offsets)
            final_indices[node_id] = (current_position, total_point_count)
            current_position += total_point_count * bytes_per_point
        
        metadata = {
            'bytes_per_point': bytes_per_point,
            'root_min': headers[0]['root_min'],
            'root_size': headers[0]['root_size'],
            'max_depth': headers[0]['max_depth'],
            'schema' : headers[0]['schema'],
            'total_points': total_points,
            'headers': headers
        }
        
        return merged_indices, final_indices, metadata

class OctreeUtils:
    """Utility functions for octree spatial operations"""
    
    @staticmethod
    def point_to_node_id(point, root_min, root_size, max_depth, stop_probs=None):
        """
        Convert point to octree node ID with optional LOD sampling.
        
        Args:
            point: 3D coordinate
            root_min: minimum corner of root node
            root_size: size of root node
            max_depth: maximum tree depth
            stop_probs: optional array of stopping probabilities per level (for LOD)
        
        Returns:
            node_id: string like 'r' or 'r0154' representing the octree path
        """
        from random import random
        
        node_id = 'r'
        current_min = root_min.copy()
        current_size = root_size
        
        for level in range(max_depth):
            # LOD sampling (if enabled)
            if stop_probs and random() < stop_probs[level]:
                return node_id
            
            center = current_min + current_size / 2
            octant = 0
            if point[0] >= center[0]: octant += 1
            if point[1] >= center[1]: octant += 2
            if point[2] >= center[2]: octant += 4
            
            node_id += str(octant)
            current_size /= 2
            
            if octant & 0b001: current_min[0] = center[0]
            if octant & 0b010: current_min[1] = center[1]
            if octant & 0b100: current_min[2] = center[2]
        
        return node_id
    
    @staticmethod
    def node_id_to_grid_coords(node_id):
        """Convert node ID to grid coordinates"""
        coords = [0, 0, 0]
        for i, char in enumerate(node_id[1:]):
            octant = int(char)
            shift = len(node_id) - 2 - i
            if octant & 1: coords[0] += (1 << shift)
            if octant & 2: coords[1] += (1 << shift)
            if octant & 4: coords[2] += (1 << shift)
        return coords
    
    @staticmethod
    def grid_coords_to_node_id(x, y, z, level):
        """Convert grid coordinates to node ID"""
        node_id = 'r'
        for depth in range(level):
            bit_pos = level - 1 - depth
            octant = 0
            if x & (1 << bit_pos): octant += 1
            if y & (1 << bit_pos): octant += 2
            if z & (1 << bit_pos): octant += 4
            node_id += str(octant)
        return node_id
    
    @staticmethod
    def calculate_root_bounds(points, padding=3.0):
        """
        Calculate root node bounds from point cloud.
        
        Args:
            points: Nx3 array of coordinates
            padding: multiplier for extent (default 3.0)
        
        Returns:
            dict with 'root_min' and 'root_size'
        """
        mins = points.min(axis=0).astype(np.float64)
        maxs = points.max(axis=0).astype(np.float64)
        center = (mins + maxs) / 2
        extent = (maxs - mins).max()
        root_size = extent * padding
        root_min = center - root_size / 2

        return {
            'root_min': root_min,
            'root_size': root_size
        }

class DoubleBufferMerge:
    #100 MB buffer for testing with small point cloud
    def __init__(self, merged_indices, final_indices, metadata, output_fileName, writer, buffer_size=100 * 1024**2):
        self.buffer_size = buffer_size
        self.buffer_a = shared_memory.SharedMemory(create=True, size=buffer_size)
        self.buffer_b = shared_memory.SharedMemory(create=True, size=buffer_size)
        self.active_buffer = self.buffer_a
        self.position = 0  # Current write position in active buffer
        self.bytes_per_point = metadata['bytes_per_point']
        self.bytes_flushed = 0

        #Init the final file
        self.pco_writer = writer
        self.output_file = open(output_fileName, 'r+b')
        self.output_file.seek(PCOFormat.HEADER_SIZE) #seek past header

        #Store indices
        self.merged_indices = merged_indices
        self.merged_indices_list = list(merged_indices.items())
        self.final_indices = final_indices
        self.final_indices_list = list(final_indices.items())

        #Pickle indices to SHM
        merged_data = pickle.dumps(self.merged_indices_list)
        final_data = pickle.dumps(self.final_indices_list)
        self.merged_shm = shared_memory.SharedMemory(create=True, size=len(merged_data))
        self.final_shm = shared_memory.SharedMemory(create=True, size=len(final_data))
        self.merged_shm.buf[:len(merged_data)] = merged_data
        self.final_shm.buf[:len(final_data)] = final_data

        #Flush thread management
        self.flush_thread = None

        #Track whos working on what buffer
        self.pending = {
            self.buffer_a.name: set(),
            self.buffer_b.name: set()
        }

        # Track pending flushes
        self.flush_pending = {
            self.buffer_a.name: None,  # None or (buffer_object, num_bytes)
            self.buffer_b.name: None
        }

    def get_shm_indices(self):
        return (self.merged_shm.name, self.final_shm.name)

    def write_singleThread(self, chunk_start, chunk_end):
        start_bytes_final = self.final_indices_list[chunk_start][1][0]
        end_bytes_final = (self.final_indices_list[chunk_end][1][0]
            +self.final_indices_list[chunk_end][1][1]*self.bytes_per_point)
        total_bytes = end_bytes_final - start_bytes_final
        
        # Check if we need to swap
        if self.position + total_bytes > self.buffer_size:
            # BANDAID: For single-threaded, flush immediately and swap manually
            # Flush current buffer
            if self.position > 0:
                print(f"Single-threaded: flushing current buffer {self.active_buffer.name}")
                data = bytes(self.active_buffer.buf[:self.position])
                self.output_file.write(data)
                self.output_file.flush()
                self.bytes_flushed += self.position
            
            # Swap to other buffer (don't use _prepare_swap which marks for flush)
            self.active_buffer = self.buffer_b if self.active_buffer == self.buffer_a else self.buffer_a
            self.position = 0
            position_to_return = 0
        else:
            position_to_return = self.position
        
        self.position += total_bytes
        return self.active_buffer.name, position_to_return

    def write(self, chunk_start, chunk_end):
        """
        Multi-threaded version: marks buffers for async flush, returns status
        
        Returns:
            tuple: (buffer_name, start_position, can_proceed)
                   can_proceed=False means you need to wait for workers to finish before calling write() again
        """
        start_bytes_final = self.final_indices_list[chunk_start][1][0]
        end_bytes_final = (self.final_indices_list[chunk_end][1][0]
            +self.final_indices_list[chunk_end][1][1]*self.bytes_per_point)
        total_bytes = end_bytes_final - start_bytes_final
        
        # Check if we need to swap
        if self.position + total_bytes > self.buffer_size:
            # Check if the buffer we're about to swap to still has pending flush
            next_buffer = self.buffer_b if self.active_buffer == self.buffer_a else self.buffer_a
            if self.flush_pending[next_buffer.name] is not None:
                print(f"WARNING: Next buffer {next_buffer.name} still pending flush!")
                # DON'T block - return immediately and let main loop handle it
                return None, None, False
            
            # Mark current buffer for flush and swap
            position_to_return = 0
            self._prepare_swap()
        else:
            position_to_return = self.position
        
        self.position += total_bytes
        return self.active_buffer.name, position_to_return, True

    def _prepare_swap(self):
        """Mark current buffer for flush and swap to other buffer"""
        # Store flush info for current buffer
        buffer_to_flush = self.active_buffer
        bytes_in_buffer = self.position
        
        self.flush_pending[buffer_to_flush.name] = (buffer_to_flush, bytes_in_buffer)
        
        # Swap to other buffer
        self.active_buffer = self.buffer_b if self.active_buffer == self.buffer_a else self.buffer_a
        self.position = 0

    def assign_worker(self, worker_id, buffer_name):
        """Main script calls this when assigning work to a worker"""
        self.pending[buffer_name].add(worker_id)

    def release_worker(self, worker_id, buffer_name):
        """
        Main script calls this when worker completes.
        Returns True if a flush was triggered.
        """
        self.pending[buffer_name].discard(worker_id)
        
        # Check if this buffer needs flushing and is now ready
        # BLOCKING IF PREVIOUS FLUSH IS STILL PENDING
        if self.flush_pending[buffer_name] is not None and len(self.pending[buffer_name]) == 0:
            self._do_flush(buffer_name)
            return True
        
        return False

    def _do_flush(self, buffer_name):
        """Actually start the flush for a buffer"""
        buffer_obj, num_bytes = self.flush_pending[buffer_name]
        
        # Wait for previous flush thread if still running
        # BLOCKING
        if self.flush_thread and self.flush_thread.is_alive():
            print("Waiting for previous flush to complete...")
            self.flush_thread.join()
        
        # Start new flush thread
        self.flush_thread = threading.Thread(
            target=self._flush_buffer,
            args=(buffer_obj, num_bytes)
        )
        self.flush_thread.start()
        
        # Clear the pending flush
        self.flush_pending[buffer_name] = None
        
        # Update cumulative bytes written
        self.bytes_flushed += num_bytes

    def wait_for_flush(self):
        """Block until current flush completes"""
        if self.flush_thread and self.flush_thread.is_alive():
            self.flush_thread.join()

    def _flush_buffer(self, buffer, num_bytes):
        """Internal method run in thread"""
        if not self.output_file:
            raise ValueError("Output file not set")
        
        print(f"Flushing {num_bytes / 1024**3:.2f} GB to disk...")
        data = bytes(buffer.buf[:num_bytes])
        self.output_file.write(data)
        self.output_file.flush()
        print(f"Flush complete")

    def close_shms(self):
        # Wait for any pending flush
        self.wait_for_flush()
        
        # Close output file
        if self.output_file:
            self.output_file.close()
        
        try:
            self.merged_shm.close()
            self.merged_shm.unlink()
            self.final_shm.close()
            self.final_shm.unlink()
            self.buffer_a.close()
            self.buffer_a.unlink()
            self.buffer_b.close()
            self.buffer_b.unlink()
        except Exception as e:
            print(f"Warning: Failed to cleanup shms: {e}")
            return 0
        return 1

    def final_flush(self):
        """Flush the active buffer at the end of processing"""
        # Wait for any pending flush to complete first
        if self.flush_thread and self.flush_thread.is_alive():
            print("Waiting for previous flush to complete...")
            self.flush_thread.join()
        
        # If there's data in the active buffer, flush it
        if self.position > 0:
            print(f"Final flush of {self.position / 1024**3:.2f} GB...")
            data = bytes(self.active_buffer.buf[:self.position])
            self.output_file.write(data)
            self.output_file.flush()
            self.bytes_flushed += self.position
            self.position = 0
        
        # Also wait for any other pending flush
        self.close_shms()



def printf(message):
    """Timestamped print for debugging"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}]: {message}")


def printd(message, debug=False):
    """Timestamped debug print"""
    if debug:
        print(f"[{datetime.now().strftime('%H:%M:%S')}]: {message}")
