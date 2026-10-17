import configparser
import ast

def str2bool(s):
    return s.lower() in ("true", "1", "yes")

# def parse_dict(section, config):
#     return {k: None if v == "None" else v for k, v in config.items(section)}

def parse_dict(section, config):
    result = {}

    for k, v in config.items(section):
        if v == "None":
            result[k] = None
            continue

        v = v.strip()

        # Convert "[a, b, c]" -> ["a", "b", "c"]
        if v.startswith("[") and v.endswith("]"):
            try:
                parsed = ast.literal_eval(v)

                # Handle unquoted names like [x, y, z]
                if not isinstance(parsed, list):
                    raise ValueError

                result[k] = parsed

            except Exception:
                inner = v[1:-1]
                result[k] = [item.strip() for item in inner.split(",")]
        else:
            result[k] = v

    return result

def parse_bool_dict(section, config):
    return {k: str2bool(v) for k, v in config.items(section)}

def optional_float(section, key):
    """Value of `key` in a config section as a float, or None if the key is missing, empty or None."""
    value = section.get(key, "None").strip()
    if value.lower() in ("none", ""):
        return None
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{key} must be a number or None, not {value!r}") from None

def parse_grid_info(config):
    grid_cfg = config["grid_info"]
    return {
        "grid_dimension": int(grid_cfg["grid_dimension"]),
        "grid_axes": grid_cfg["grid_axes"],
        "automatic_grid": grid_cfg.getboolean("automatic_grid"),
        "x_min": float(grid_cfg["x_min"]) if grid_cfg["x_min"] != "None" else None,
        "x_max": float(grid_cfg["x_max"]) if grid_cfg["x_max"] != "None" else None,
        "y_min": float(grid_cfg["y_min"]) if grid_cfg["y_min"] != "None" else None,
        "y_max": float(grid_cfg["y_max"]) if grid_cfg["y_max"] != "None" else None,
        "z_min": float(grid_cfg["z_min"]) if grid_cfg["z_min"] != "None" else None,
        "z_max": float(grid_cfg["z_max"]) if grid_cfg["z_max"] != "None" else None,
        "x_transect": float(grid_cfg["x_transect"]) if grid_cfg["x_transect"] != "None" else None,
        "y_transect": float(grid_cfg["y_transect"]) if grid_cfg["y_transect"] != "None" else None,
        "z_transect": float(grid_cfg["z_transect"]) if grid_cfg["z_transect"] != "None" else None,
        "x_axis_periodic": grid_cfg.getboolean("x_axis_periodic"),
        "y_axis_periodic": grid_cfg.getboolean("y_axis_periodic"),
        "z_axis_periodic": grid_cfg.getboolean("z_axis_periodic"),
        # optional length of the periodic box along each axis; without it the bounds of the data are used
        "x_box_length": optional_float(grid_cfg, "x_box_length"),
        "y_box_length": optional_float(grid_cfg, "y_box_length"),
        "z_box_length": optional_float(grid_cfg, "z_box_length"),
    }

import configparser

def load_config(path_to_ini="config.ini"):
    config = configparser.ConfigParser()
    config.optionxform = str  
    config.read(path_to_ini)

    config_data = {
        "particles_path": config["paths"]["particles_path"],
        "contacts_path": config["paths"]["contacts_path"],
        "output_path": config["paths"]["output_path"],
        "t0": int(config["timesteps"]["t0"]),
        "tf": int(config["timesteps"]["tf"]),
        "dt": int(config["timesteps"]["dt"]),
        "t_pad": int(config["timesteps"]["t_pad"]),
        "DEM_software": config["DEM_software"]["name"],
        "partialignore": str2bool(config["flags"]["partialignore"]),
        "key_mapping": parse_dict("key_mapping", config),   # Keys will now keep their capitalization
        "grid_info": parse_grid_info(config),
        "fields_to_export": parse_bool_dict("fields_to_export", config),
        "smoothing_function": config["smoothing_function"]["type"],
        "h5_output": str2bool(config["output_options"]["h5_output"]),
        "vkthdf_output": str2bool(config["output_options"]["vkthdf_output"]),
    }

    return config_data



