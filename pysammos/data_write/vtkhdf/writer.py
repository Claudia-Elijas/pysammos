from . import core as v5i
import h5py
import pyvista as pv

class VTKHDFWriter:
    def __init__(self, node_dimensions, node_spacing, origin, path):
        self.node_dimensions = tuple(node_dimensions)
        self.node_spacing = tuple(node_spacing)
        self.path = path
        self.origin = origin #v5i.origin_of_centered_image(self.node_dimensions, self.node_spacing, 2)
        
    def write(self, data_dict):
        box = pv.ImageData(dimensions=self.node_dimensions, spacing=self.node_spacing, origin=self.origin)

        for key, value in data_dict.items():
            #print(f"saving {key}")
            if value is not None:
                dims = value.ndim
                if dims == 1:
                    data = value.reshape(*self.node_dimensions, order='C')
                    v5i.set_point_scalar(box, data, key)
                elif dims == 2:
                    data = value.reshape((*self.node_dimensions, 3), order='C')
                    v5i.set_point_vector(box, data, key)
                elif dims == 3:
                    ncomp = len(value[0,...].flatten(order='C'))
                    data = value.reshape((*self.node_dimensions, ncomp), order='C')
                    v5i.set_point_tensor(box, data, key, components=ncomp)
                else:
                    print(f"{key}: unknown type, shape = {value.shape}")

        filename = self.path + ".vtkhdf"
        with h5py.File(filename, "w") as f:
            v5i.write_vtkhdf(f, box, compression='gzip', compression_opts=4)
        print(f"  File successfully written to {filename}")
        
    def write_polydisperse(self, data_dict, n_phases, phase_indepen_field_names):
        box = pv.ImageData(dimensions=self.node_dimensions, spacing=self.node_spacing, origin=self.origin)
        suffixes = ['_bulk'] + [f"_phase{i}" for i in range(1, n_phases)] # nphases should include hte bulk
        for key, value in data_dict.items():
            if key in phase_indepen_field_names:
                if value is not None:
                    dims = value.ndim
                    if dims == 1:
                        data = value.reshape(*self.node_dimensions, order='C')
                        v5i.set_point_scalar(box, data, key)
                    elif dims == 2:
                        data = value.reshape((*self.node_dimensions, 3), order='C')
                        v5i.set_point_vector(box, data, key)
                    elif dims == 3:
                        ncomp = len(value[0,...].flatten(order='C'))
                        data = value.reshape((*self.node_dimensions, ncomp), order='C')
                        v5i.set_point_tensor(box, data, key, components=ncomp)
                    else:
                        print(f"{key}: unknown type, shape = {value.shape}")
            else:
                if value is not None:
                    for p in range(n_phases):
                        value_phase = value[:, p, ...]
                        dims = value_phase.ndim
                        key_suffix = key + suffixes[p]
                        if dims == 1:
                            data = value_phase.reshape(*self.node_dimensions, order='C')
                            v5i.set_point_scalar(box, data, key_suffix)
                        elif dims == 2:
                            data = value_phase.reshape((*self.node_dimensions, 3), order='C')
                            v5i.set_point_vector(box, data, key_suffix)
                        elif dims == 3:
                            #box.point_data[key_suffix] = value_phase.reshape(-1, 9, order='F')
                            ncomp = len(value_phase[0,...].flatten(order='F'))
                            data = value_phase.reshape((*self.node_dimensions, ncomp), order='C')
                            v5i.set_point_tensor(box, data, key_suffix, components=ncomp)
                        else:
                            print(f"{key}: unknown type, shape = {value_phase.shape}")

        filename = self.path + ".vtkhdf"
        with h5py.File(filename, "w") as f:
            v5i.write_vtkhdf(f, box, compression='gzip', compression_opts=4)

        print(f"File successfully written to {filename}")

