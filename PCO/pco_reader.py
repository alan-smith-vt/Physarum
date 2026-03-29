"""
PCO Reader - Read and query Point Cloud Octree files

This module provides high-level reading operations for PCO format files.
"""

import numpy as np
import struct
from .pco_format import PCOFormat, OctreeUtils, printf


class PCOReader:
    """
    Reader for Point Cloud Octree (PCO) binary files.
    
    Usage:
        reader = PCOReader('myfile.pco')
        reader.load_metadata()
        
        # Read specific nodes
        data = reader.read_node('r0')
        
        # Read by level
        data = reader.read_level(3)
        
        # Spatial queries
        data = reader.query_ball(center=[0,0,0], radius=5.0, level=5)
    """
    
    def __init__(self, filename=None):
        self.filename = filename
        self.header = None
        self.index = None
    
    def load_metadata(self, filename=None):
        """
        Read file header and index.
        
        Args:
            filename: optional, overrides constructor filename
        
        Returns:
            tuple: (header_dict, index_dict)
        """
        if filename:
            self.filename = filename
        
        if not self.filename:
            raise ValueError("No filename provided")
        
        with open(self.filename, 'rb') as f:
            # Read magic number
            magic = f.read(8)
            if magic != PCOFormat.MAGIC:
                raise ValueError(f"Invalid file format. Expected {PCOFormat.MAGIC}, got {magic}")
            
            # Read header fields
            max_depth = struct.unpack('H', f.read(2))[0]
            root_min = struct.unpack('ddd', f.read(24))
            root_size = struct.unpack('f', f.read(4))[0]
            total_points = struct.unpack('Q', f.read(8))[0]
            schema = struct.unpack('B', f.read(1))[0]
            bytes_per_point = struct.unpack('B', f.read(1))[0]
            index_offset = struct.unpack('Q', f.read(8))[0]
            
            # Read index
            f.seek(index_offset)
            num_nodes = struct.unpack('I', f.read(4))[0]
            
            index = {}
            for _ in range(num_nodes):
                node_id = f.read(24).decode('ascii').rstrip('\0')
                offset = struct.unpack('Q', f.read(8))[0]
                count = struct.unpack('I', f.read(4))[0]
                index[node_id] = (offset, count)
        
        self.header = {
            'max_depth': max_depth,
            'root_min': root_min,
            'root_size': root_size,
            'total_points': total_points,
            'schema': schema,
            'bytes_per_point': bytes_per_point,
            'index_offset': index_offset,
            'num_nodes': num_nodes
        }
        self.index = index
        
        return self.header, self.index
    
    def parse_binary_data(self, bdata, schema=None):
        """
        Parse interleaved binary data into structured arrays.
        
        Args:
            bdata: raw binary data
            schema: optional schema override (uses header schema if None)
        
        Returns:
            dict with keys like 'xyz', 'rgb', 'intensity', 'normal'
        """
        if schema is None:
            schema = self.header['schema']
        
        dtype = PCOFormat.get_dtype(schema)
        structured_data = np.frombuffer(bdata, dtype=dtype)
        
        result = {}
        if schema & PCOFormat.FIELD_XYZ:
            result['xyz'] = np.column_stack([
                structured_data['x'],
                structured_data['y'],
                structured_data['z']
            ]).astype(np.float64) + np.array(self.header['root_min'], dtype=np.float64)
        
        if schema & PCOFormat.FIELD_RGB:
            result['rgb'] = np.column_stack([
                structured_data['r'],
                structured_data['g'],
                structured_data['b']
            ])
        
        if schema & PCOFormat.FIELD_INTENSITY:
            result['intensity'] = structured_data['intensity']
        
        if schema & PCOFormat.FIELD_NORMAL:
            result['normal'] = np.column_stack([
                structured_data['nx'],
                structured_data['ny'],
                structured_data['nz']
            ])
        
        return result
    
    def read_node(self, node_id):
        """
        Read raw binary data for a single node.
        
        Args:
            node_id: string like 'r0' or 'r0154'
        
        Returns:
            binary data for the node
        """
        if self.index is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")
        
        offset, count = self.index[node_id]
        bytes_per_point = self.header['bytes_per_point']
        
        with open(self.filename, 'rb') as f:
            f.seek(offset)
            return f.read(bytes_per_point * count)
    
    def read_nodes(self, node_ids):
        """
        Read raw binary data for multiple nodes.
        
        Args:
            node_ids: list of node ID strings
        
        Returns:
            concatenated binary data for all nodes
        """
        if self.index is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")
        
        all_bytes = []
        bytes_per_point = self.header['bytes_per_point']
        
        with open(self.filename, 'rb') as f:
            for node_id in node_ids:
                offset, count = self.index[node_id]
                f.seek(offset)
                all_bytes.append(f.read(bytes_per_point * count))
        
        return b''.join(all_bytes)
    
    def read_level(self, level):
        """
        Read all nodes at a specific octree level.
        
        Args:
            level: tree level (0 = root)
        
        Returns:
            binary data for all nodes at that level
        """
        node_ids = [nid for nid in self.index.keys() if len(nid) == level + 1]
        return self.read_nodes(node_ids)
    
    def read_up_to_level(self, level):
        """
        Read all nodes up to and including specified level.
        
        Args:
            level: maximum tree level to include
        
        Returns:
            binary data for all nodes at levels 0 through level
        """
        bytes_per_point = self.header['bytes_per_point']
        
        # Find ranges to skip (deeper than target level)
        skip_ranges = []
        for node_id, (offset, count) in sorted(self.index.items(), key=lambda x: x[1][0]):
            if len(node_id) > level + 1:
                skip_ranges.append((offset, offset + count * bytes_per_point))
        skip_ranges = sorted(skip_ranges, key=lambda x: x[0])
        
        all_bytes = []
        with open(self.filename, 'rb') as f:
            f.seek(PCOFormat.HEADER_SIZE)
            
            for skip_start, skip_end in skip_ranges:
                byte_length = skip_start - f.tell()
                if byte_length > 0:
                    all_bytes.append(f.read(byte_length))
                f.seek(skip_end)
            
            # Read final segment
            final_length = self.header['index_offset'] - f.tell()
            if final_length > 0:
                all_bytes.append(f.read(final_length))
        
        return b''.join(all_bytes)
    
    def query_ball(self, center, radius, level):
        """
        Query points within a sphere.
        
        Args:
            center: 3D point [x, y, z]
            radius: sphere radius
            level: octree level to query at
        
        Returns:
            binary data for all points in intersecting nodes
        """
        center = np.array(center)
        mins = center - radius
        maxs = center + radius
        root_min = np.array(self.header['root_min'])
        
        node_min = OctreeUtils.point_to_node_id(
            mins, root_min, self.header['root_size'], level
        )
        node_max = OctreeUtils.point_to_node_id(
            maxs, root_min, self.header['root_size'], level
        )
        
        min_coords = OctreeUtils.node_id_to_grid_coords(node_min)
        max_coords = OctreeUtils.node_id_to_grid_coords(node_max)
        
        # Enumerate all nodes in bounding box
        intersecting_nodes = []
        for x in range(min_coords[0], max_coords[0] + 1):
            for y in range(min_coords[1], max_coords[1] + 1):
                for z in range(min_coords[2], max_coords[2] + 1):
                    node_id = OctreeUtils.grid_coords_to_node_id(x, y, z, level)
                    if node_id in self.index:
                        intersecting_nodes.append(node_id)
        
        return self.read_nodes(intersecting_nodes)
    
    def extract_z_slice(self, z, level):
        """
        TODO: MODIFY TO BE SIMPLER. JUST GRAB A LAYER OF NODES
        AT THE Z COORINDATE AND LEVEL. NO RADIUS NEEDED
        Extract a horizontal slice at given z coordinate.
        
        Args:
            z: z-coordinate for slice
            level: octree level to sample at
        
        Returns:
            binary data for points in slice
        """
        root_size = self.header['root_size']
        root_min = np.array(self.header['root_min'])
        radius = (root_size / 2**level) * 0.9
        
        intersecting_nodes = []
        for x in np.linspace(root_min[0], root_min[0] + root_size):
            for y in np.linspace(root_min[1], root_min[1] + root_size):
                point = np.array([x, y, z])
                mins = point - radius
                maxs = point + radius
                
                node_min = OctreeUtils.point_to_node_id(mins, root_min, root_size, level)
                node_max = OctreeUtils.point_to_node_id(maxs, root_min, root_size, level)
                
                min_coords = OctreeUtils.node_id_to_grid_coords(node_min)
                max_coords = OctreeUtils.node_id_to_grid_coords(node_max)
                
                for gx in range(min_coords[0], max_coords[0] + 1):
                    for gy in range(min_coords[1], max_coords[1] + 1):
                        for gz in range(min_coords[2], max_coords[2] + 1):
                            node_id = OctreeUtils.grid_coords_to_node_id(gx, gy, gz, level)
                            if node_id in self.index:
                                intersecting_nodes.append(node_id)
        
        unique_nodes = list(set(intersecting_nodes))
        return self.read_nodes(unique_nodes)
    
    def save_as_ply(self, bdata, filename, schema=None):
        """
        Save binary point data as PLY file using Open3D.
        
        Args:
            bdata: binary data from read operations
            filename: output PLY file path
            schema: optional schema override
        """
        #print(f"Start of save as ply file")
        import open3d as o3d
        
        if schema is None:
            schema = self.header['schema']
        
        results = self.parse_binary_data(bdata, schema)
        points = results['xyz']
        #print(f"points len: {len(points)}")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        #print(f"pcd points len: {len(pcd.points)}")
        
        if schema & PCOFormat.FIELD_RGB:
            colors = results['rgb'].astype(np.float64) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        if schema & PCOFormat.FIELD_NORMAL:
            normals = results['normal'].astype(np.float64)
            pcd.normals = o3d.utility.Vector3dVector(normals)
        
        if schema & PCOFormat.FIELD_INTENSITY and not (schema & PCOFormat.FIELD_RGB):
            # Store intensity as grayscale if no RGB present
            intensity = results['intensity'].astype(np.float64)
            intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
            gray_colors = np.column_stack([intensity_norm, intensity_norm, intensity_norm])
            pcd.colors = o3d.utility.Vector3dVector(gray_colors)
        
        success = o3d.io.write_point_cloud(filename, pcd)
        #print(f"File saved successfully: {success}")
    
    def get_info(self):
        """Get human-readable info about the file"""
        if self.header is None:
            return "No metadata loaded"
        
        schema = self.header['schema']
        fields = []
        if schema & PCOFormat.FIELD_XYZ: fields.append('XYZ')
        if schema & PCOFormat.FIELD_RGB: fields.append('RGB')
        if schema & PCOFormat.FIELD_INTENSITY: fields.append('Intensity')
        if schema & PCOFormat.FIELD_NORMAL: fields.append('Normal')
        
        return (
            f"PCO File: {self.filename}\n"
            f"  Points: {self.header['total_points']:,}\n"
            f"  Nodes: {self.header['num_nodes']:,}\n"
            f"  Max Depth: {self.header['max_depth']}\n"
            f"  Fields: {', '.join(fields)}\n"
            f"  Bytes/Point: {self.header['bytes_per_point']}\n"
            f"  Root Min: {self.header['root_min']}\n"
            f"  Root Size: {self.header['root_size']:.2f}"
        )
