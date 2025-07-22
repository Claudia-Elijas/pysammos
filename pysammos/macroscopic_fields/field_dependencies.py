
field_dependencies = {
                "density_particle": ["volume_fraction","density_mixture"],
                "velocity": ["momentum_density", "density_mixture"],
                "velocity_gradient" : ["velocity"],
                "shear_rate_tensor": ["velocity_gradient"],
                "kinetic_tensor": ["velocity", "velocity_gradient"],
                "total_stress_tensor": ["kinetic_tensor", "contact_tensor"],
                "pressure": ["total_stress_tensor"],
                "inertial_number": ["shear_rate_tensor", "pressure", "density_particle", "d43", "d32"],
                "frictional_coefficient": ["total_stress_tensor", "pressure"],
                "granular_temperature": ["kinetic_tensor", "density_mixture"],
                            }


def get_fields_to_compute(fields_to_export):
    
        fields_to_compute = set()
        # recursive functions for dependencies
        def add_field(field):
            if field not in fields_to_compute:
                fields_to_compute.add(field)
                for dep in field_dependencies.get(field, []):
                    add_field(dep)

        # Add all user-selected outputs and their dependencies
        for field, include in fields_to_export.items():
            if include:
                add_field(field)

        return fields_to_compute
