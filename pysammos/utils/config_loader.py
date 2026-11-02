import configparser
import ast
import configparser

# helper functions for parsing config files

def str2bool(s):
    return s.lower() in ("true", "1", "yes")

def parse_simulation_box(config):
    """
    [simulation_box] section -> [(xmin, xmax), (ymin, ymax), (zmin, zmax)], None for 'None' or missing
    entries; None if the section is missing (allowed only without periodic axes).
    """
    if not config.has_section("simulation_box"):
        return None
    box = []
    for axis in ("x", "y", "z"):
        value = config["simulation_box"].get(f"{axis}_range", "None").strip()
        if value == "None":
            box.append(None)
        else:
            lo, hi = (float(v) for v in value.strip("[]()").split(","))
            box.append((lo, hi))
    return box

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
    }

# Config loader function

def load_config(path_to_ini="config.ini") -> dict:
    """
    Read a pysammos .ini file and return the keyword arguments of CoarseGraining,
    so that  CoarseGraining(**load_config("config.ini"))  works.
    """
    config = configparser.ConfigParser()
    config.optionxform = str
    if not config.read(path_to_ini):
        raise FileNotFoundError(f"Config file not found: {path_to_ini}")

    kwargs = {
        "particle_path": config["paths"]["particles_path"],
        "contacts_path": config["paths"]["contacts_path"],
        "output_path": config["paths"]["output_path"],
        "start_timestep": int(config["timesteps"]["t0"]),
        "end_timestep": int(config["timesteps"]["tf"]),
        "dt_time_step": int(config["timesteps"]["dt"]),
        "t_pad": int(config["timesteps"]["t_pad"]),
        "DEM_software": config["DEM_software"]["name"],
        "DEM_keymap": parse_dict("key_mapping", config),
        "grid_info": parse_grid_info(config),
        "simulation_box": parse_simulation_box(config),
        "weight_function": config["smoothing_function"]["type"],
        "fields_to_export": parse_bool_dict("fields_to_export", config),
        "ignore_phases": str2bool(config["flags"]["partialignore"]),
        "time_average": config.getboolean("output_options", "time_average", fallback=False),
        "vkthdf_output": str2bool(config["output_options"]["vkthdf_output"]),
        "h5_output": str2bool(config["output_options"]["h5_output"]),
    }

    # optional settings: only passed if present, so the defaults stay in CoarseGraining.__init__
    if config.has_section("advanced"):
        adv = config["advanced"]
        if "velocity_gradient_method" in adv:
            kwargs["velocity_gradient_method"] = adv["velocity_gradient_method"]
        if "search_sampling_factor" in adv:
            kwargs["search_sampling_factor"] = int(adv["search_sampling_factor"])

    return kwargs



