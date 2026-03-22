"""
PCO Writer - Create Point Cloud Octree files

This module provides high-level writing operations for PCO format files.
"""

import numpy as np
import struct
from collections import defaultdict
from tqdm import tqdm
from .pco_format import PCOFormat, OctreeUtils, printf, printd


class PCOWriter:
    """
    Writer for Point Cloud Octree (PCO) binary files.
    
    Usage:
        writer = PCOWriter()
        
        # Calculate root bounds
        root = writer.calculate_root_bounds(points)
        
        # Write file
        writer.write('output.pco', points, root, max_depth=7, colors=rgb_data)
    """
    
    def __init__(self, debug=False):
        self.debug = debug
    
    def calculate_root_bounds(self, points, padding=3.0):
        """
        Calculate root node bounds from point cloud.
        Wrapper for OctreeUtils.calculate_root_bounds.
        """
        return OctreeUtils.calculate_root_bounds(points, padding)
    
    def _assign_points_vectorized(self, points, root_min, root_size, max_depth, use_lod=False):
        """
        Fully vectorized octree node assignment using integer encoding.
        
        Returns:
            dict: {node_id: [point_indices]}
        """
        n_points = len(points)
        
        # Encode node paths as integers: each octant (0-7) uses 3 bits
        node_codes = np.zeros(n_points, dtype=np.uint64)
        final_depths = np.full(n_points, max_depth, dtype=np.uint8)
        
        # Generate random numbers upfront if using LOD
        if use_lod:
            stop_probs_array = np.array([1.0 / (2 ** (max_depth - d + 2)) for d in range(max_depth)])
            random_values = np.random.random((n_points, max_depth))
        
        # Process all levels for all points
        current_min = np.tile(root_min, (n_points, 1)).astype(np.float32)
        current_size = np.float32(root_size)
        
        for level in range(max_depth):
            center = current_min + current_size / 2
            
            # Calculate octants for all points
            octants = np.zeros(n_points, dtype=np.uint8)
            octants += (points[:, 0] >= center[:, 0]).astype(np.uint8)
            octants += (points[:, 1] >= center[:, 1]).astype(np.uint8) * 2
            octants += (points[:, 2] >= center[:, 2]).astype(np.uint8) * 4
            
            # Encode octant into node_code
            node_codes = (node_codes << 3) | octants.astype(np.uint64)
            
            # Update bounds
            current_size /= 2
            current_min[:, 0] = np.where(octants & 1, center[:, 0], current_min[:, 0])
            current_min[:, 1] = np.where(octants & 2, center[:, 1], current_min[:, 1])
            current_min[:, 2] = np.where(octants & 4, center[:, 2], current_min[:, 2])
            
            # LOD: determine stopping points
            if use_lod and level < max_depth - 1:
                still_active = final_depths == max_depth
                stops_here = (random_values[:, level] < stop_probs_array[level]) & still_active
                final_depths[stops_here] = level + 1
        
        # Group points by (depth, node_code) pairs
        node_assignments = defaultdict(list)
        
        for i in range(n_points):
            depth = final_depths[i]
            code = node_codes[i] >> (3 * (max_depth - depth))
            key = (depth, code)
            node_assignments[key].append(i)
        
        # Convert keys to string node IDs
        result = {}
        for (depth, code), indices in node_assignments.items():
            node_id = 'r'
            for level in range(depth):
                octant = (code >> (3 * (depth - 1 - level))) & 0b111
                node_id += str(octant)
            result[node_id] = indices
        
        return result
    
    def write(self, filename, points, root, max_depth=7, colors=None, intensities=None, normals=None, use_lod=True, debug=False):
        """
        Write points to binary file with octree indexing.
        
        Args:
            filename: output file path
            points: Nx3 array of XYZ coordinates
            root: dict with 'root_min' and 'root_size'
            max_depth: maximum octree depth
            colors: optional Nx3 array of RGB values (0-255)
            intensities: optional N array of intensity values
            normals: optional Nx3 array of normal vectors
            use_lod: enable level-of-detail sampling
        """
        # Infer schema
        schema = PCOFormat.infer_schema(colors, intensities, normals)
        bytes_per_point = PCOFormat.calculate_bytes_per_point(schema)
        dtype = PCOFormat.get_dtype(schema)
        
        root_min = root['root_min']
        root_size = root['root_size']
        
        # Assign points to octree nodes
        printf("Assigning points to octree nodes...")
        node_assignments = self._assign_points_vectorized(
            points, root_min, root_size, max_depth, use_lod
        )
        
        # Write file
        with open(filename, 'wb') as f:
            # Write header
            f.write(PCOFormat.MAGIC)#8
            f.write(struct.pack('H', max_depth))#2
            f.write(struct.pack('fff', *root_min))#12
            f.write(struct.pack('f', root_size))#4
            f.write(struct.pack('Q', len(points)))#8 (offset 26)
            f.write(struct.pack('B', schema))#1
            f.write(struct.pack('B', bytes_per_point))#1
            index_offset_pos = f.tell()#returns 36
            f.write(struct.pack('Q', 0))  # Placeholder for index offset
            
            # Write data section
            printd("Writing point data...", debug)
            index = {}
            for node_id in tqdm(sorted(node_assignments.keys()), disable=not self.debug):
                point_indices = node_assignments[node_id]
                node_count = len(point_indices)
                node_offset = f.tell()
                
                # Create structured array
                node_data = np.zeros(node_count, dtype=dtype)
                
                if schema & PCOFormat.FIELD_XYZ:
                    node_data['x'] = points[point_indices, 0]
                    node_data['y'] = points[point_indices, 1]
                    node_data['z'] = points[point_indices, 2]
                
                if schema & PCOFormat.FIELD_RGB and colors is not None:
                    node_data['r'] = colors[point_indices, 0]
                    node_data['g'] = colors[point_indices, 1]
                    node_data['b'] = colors[point_indices, 2]
                
                if schema & PCOFormat.FIELD_INTENSITY and intensities is not None:
                    node_data['intensity'] = intensities[point_indices]
                
                if schema & PCOFormat.FIELD_NORMAL and normals is not None:
                    node_data['nx'] = normals[point_indices, 0]
                    node_data['ny'] = normals[point_indices, 1]
                    node_data['nz'] = normals[point_indices, 2]
                
                f.write(node_data.tobytes())
                index[node_id] = (node_offset, node_count)
            
            # Write index section
            index_start = f.tell()
            f.write(struct.pack('I', len(index)))
            
            for node_id, (offset, count) in index.items():
                node_id_bytes = node_id.encode('ascii').ljust(PCOFormat.NODE_ID_MAX_LENGTH, b'\0')
                f.write(node_id_bytes)
                f.write(struct.pack('Q', offset))
                f.write(struct.pack('I', count))
            
            # Update header with index offset
            f.seek(index_offset_pos)
            f.write(struct.pack('Q', index_start))
        
        printf(f"Written {len(points):,} points across {len(index):,} nodes to {filename}")
    
    def write_temp_header(self, filename, root, max_depth=7, colors=None, intensities=None, normals=None, metadata=None):
        """
        Write header for streaming/incremental writes.
        
        This creates a file with header + placeholder data section.
        Used for multi-pass writing where data is written incrementally.
        
        Args:
            filename: output file path
            data_section_size: total bytes for data section
            root: dict with 'root_min' and 'root_size'
            max_depth: maximum octree depth
            colors/intensities/normals: determines schema
        
        Returns:
            schema, bytes_per_point for use in subsequent writes
        """
        if metadata is not None:
            schema = metadata['schema']
            bytes_per_point = metadata['bytes_per_point']
            root_min = metadata['root_min']
            root_size = metadata['root_size']
            max_depth = metadata['max_depth']
        else:
            schema = PCOFormat.infer_schema(colors, intensities, normals)
            bytes_per_point = PCOFormat.calculate_bytes_per_point(schema)
            root_min = root['root_min']
            root_size = root['root_size']
        
        with open(filename, 'wb') as f:
            # Write header
            f.write(PCOFormat.MAGIC)
            f.write(struct.pack('H', max_depth))
            f.write(struct.pack('fff', *root_min))
            f.write(struct.pack('f', root_size))
            f.write(struct.pack('Q', 0))  # Total points (unknown yet)
            f.write(struct.pack('B', schema))
            f.write(struct.pack('B', bytes_per_point))
            f.write(struct.pack('Q', 0))  # Index offset (unknown yet)
            
            # Write placeholder data section
            #f.write(b'\x00' * data_section_size)
        
        return schema, bytes_per_point
    
    def write_indices(self, filename, index):
        """
        Write index section to file (for streaming/merge operations).
        
        Args:
            filename: file to update
            index: dict of {node_id: (byte_offset, point_count)}
            index_start: byte position where index begins
        """
        with open(filename, 'r+b') as f:
            # Write index
            f.seek(0,2)
            index_start = f.tell()
            f.write(struct.pack('I', len(index)))
            
            total_points = 0
            for node_id, (offset, count) in index.items():
                node_id_bytes = node_id.encode('ascii').ljust(PCOFormat.NODE_ID_MAX_LENGTH, b'\0')
                f.write(node_id_bytes)
                f.write(struct.pack('Q', offset))
                f.write(struct.pack('I', count))
                total_points+=count
                
            f.seek(26)
            f.write(struct.pack('Q', total_points))
             
            # Update header with index offset
            f.seek(36)  # Position of index_offset field in header
            f.write(struct.pack('Q', index_start))


    def write_temp_body(self, filename, points, colors=None, intensities=None, normals=None, use_lod=True, debug=False):
        """
        Write point data to a file with existing header (for streaming/incremental writes).
        
        Args:
            filename: file to write to (header must already exist)
            points: Nx3 array of XYZ coordinates
            root: dict with 'root_min' and 'root_size'
            max_depth: maximum octree depth
            colors: optional Nx3 array of RGB values (0-255)
            intensities: optional N array of intensity values
            normals: optional Nx3 array of normal vectors
            use_lod: enable level-of-detail sampling
        """
        # Read header
        header = PCOFormat.read_header(filename)
        schema = header['schema']
        max_depth = header['max_depth']
        dtype = PCOFormat.get_dtype(schema)
        root_min = header['root_min']
        root_size = header['root_size']
        
        # Assign points to octree nodes (vectorized)
        printd("Assigning points to octree nodes...", debug=debug)
        node_assignments = self._assign_points_vectorized(points, root_min, root_size, max_depth, use_lod)

        # Write file
        with open(filename, 'r+b') as f:
            # Skip Header
            f.seek(PCOFormat.HEADER_SIZE)

            # Data section
            printd("Writing point data...")
            index = {}
            for node_id in sorted(node_assignments.keys()):
                point_indices = node_assignments[node_id]
                node_count = len(point_indices)
                node_offset = f.tell()
                
                # Create structured array
                node_data = np.zeros(node_count, dtype=dtype)
                
                if schema & PCOFormat.FIELD_XYZ:
                    node_data['x'] = points[point_indices, 0]
                    node_data['y'] = points[point_indices, 1]
                    node_data['z'] = points[point_indices, 2]
                
                if schema & PCOFormat.FIELD_RGB and colors is not None:
                    node_data['r'] = colors[point_indices, 0]
                    node_data['g'] = colors[point_indices, 1]
                    node_data['b'] = colors[point_indices, 2]
                
                if schema & PCOFormat.FIELD_INTENSITY and intensities is not None:
                    node_data['intensity'] = intensities[point_indices]
                
                if schema & PCOFormat.FIELD_NORMAL and normals is not None:
                    node_data['nx'] = normals[point_indices, 0]
                    node_data['ny'] = normals[point_indices, 1]
                    node_data['nz'] = normals[point_indices, 2]
                
                f.write(node_data.tobytes())
                index[node_id] = (node_offset, node_count)
            
            # # Index section
            # index_start = f.tell()
            # f.write(struct.pack('I', len(index)))
            
            # for node_id, (offset, count) in index.items():
            #     node_id_bytes = node_id.encode('ascii').ljust(PCOFormat.NODE_ID_MAX_LENGTH, b'\0')
            #     f.write(node_id_bytes)
            #     f.write(struct.pack('Q', offset))
            #     f.write(struct.pack('I', count))
            
            # # Update header with index offset
            # f.seek(36)  # Position of index_offset field in header
            # f.write(struct.pack('Q', index_start))
        self.write_indices(filename, index)
        
        printd(f"Written {len(points):,} points across {len(index):,} nodes to {filename}", debug=debug)
