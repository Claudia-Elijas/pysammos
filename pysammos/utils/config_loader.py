"""
Read a pysammos configuration (.ini) file into the keyword arguments of ``CoarseGraining``.

Only a few entries are required (the data paths, the first time step, the DEM software, the grid
dimension and axes, and the key mapping). Every other entry has a default, listed in ``DEFAULTS``,
so a section or entry can be left out when its default is wanted, e.g. no ``[simulation_box]`` and no
``*_axis_periodic`` entries for a case without periodic boundaries. Writing ``None`` is the same as
leaving the entry out.

Unknown entries (e.g. a typo such as ``x_axis_periodc``) are reported with a warning, since with
defaults a misspelt entry would otherwise be silently replaced by its default.
"""
import ast
import configparser
import warnings

from ..macroscopic_fields.gridded.derived_fields import EXPORT_GROUPS

AXES = ("x", "y", "z")

# Entries that must be given: (section, entry)
REQUIRED = [
    ("paths", "particles_path"),
    ("paths", "contacts_path"),
    ("timesteps", "t0"),
    ("timesteps", "tf"),
    ("timesteps", "t_pad"),
    ("DEM_software", "name"),
    ("grid_info", "grid_dimension"),
    ("grid_info", "grid_axes"),

]

# Defaults of the optional entries, per section. None means "not given".
DEFAULTS = {
    "paths": {"output_path": "./PysammosCG/"},
    "timesteps": {"dt": 1},                      # tf: None -> same as t0 (one time step)
    "smoothing_function": {"type": "Lucy"},
    "flags": {"partialignore": True},
    "grid_info": {
        "automatic_grid": True,                              # None -> True unless grid limits are given
        **{f"{a}_{m}": None for a in AXES for m in ("min", "max")},
        **{f"{a}_transect": None for a in AXES},
        **{f"{a}_axis_periodic": False for a in AXES},
    },
    "simulation_box": {f"{a}_range": None for a in AXES},
    "output_options": {"vkthdf_output": True, "h5_output": True, "time_average": False},
    "advanced": {"velocity_gradient_method": None, "search_sampling_factor": None},  # None -> CoarseGraining default
}

# Entries of [key_mapping] read by the data readers; None (or missing) means "not in the data".
KEY_MAPPING_KEYS = ["Global_ID", "Particle_Position", "Particle_Velocity", "Particle_Diameter",
                    "Particle_Radius", "Particle_Density", "Particle_Volume", "Particle_Mass",
                    "Coordination_Number", "Particle_i_ID", "Particle_j_ID", "Force_ij", "Contact_ij"]

# Entries of [fields_to_export]; a missing entry is False (not exported).
EXPORT_FLAGS = list(EXPORT_GROUPS) + ["granular_temperature_slices"]


# ---------------------------------------------------------------------------------------------
# reading single entries
# ---------------------------------------------------------------------------------------------
def _is_none(value):
    return value is None or value.strip() in ("", "None", "none")


def _raw(config, section, key):
    """The text of an entry, or None if the section or entry is missing or set to None."""
    if not config.has_section(section) or key not in config[section]:
        return None
    value = config[section][key]
    return None if _is_none(value) else value.strip()


def str2bool(s):
    s = s.strip().lower()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"Not a boolean: {s!r} (use True or False).")


def get(config, section, key, kind=str, default=None):
    """
    Read one entry, converted with ``kind`` (str, int, float, bool, or 'range' for "lo, hi").
    Missing section, missing entry or None -> ``default``.
    """
    value = _raw(config, section, key)
    if value is None:
        return default
    try:
        if kind is bool:
            return str2bool(value)
        if kind == "range":
            lo, hi = (float(v) for v in value.strip("[]()").split(","))
            return (lo, hi)
        return kind(value)
    except ValueError as err:
        raise ValueError(f"[{section}] {key} = {value!r}: {err}") from None


def parse_list_or_str(value):
    """'[a, b, c]' -> ['a', 'b', 'c'] (quoted or not); anything else is returned as it is."""
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        return [item.strip() for item in value[1:-1].split(",")]
    return value


# ---------------------------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------------------------
def check_required(config):
    missing = [f"[{s}] {k}" for s, k in REQUIRED if _raw(config, s, k) is None]
    if not config.has_section("key_mapping"):
        missing.append("[key_mapping] section")
    if missing:
        raise ValueError("Missing required entries in the config file: " + ", ".join(missing))


def warn_unknown(config):
    """Warn about entries that pysammos does not read (usually typos)."""
    known = {s: set(d) for s, d in DEFAULTS.items()}
    for s, k in REQUIRED:
        known.setdefault(s, set()).add(k)
    known["key_mapping"] = set(KEY_MAPPING_KEYS)
    known["fields_to_export"] = set(EXPORT_FLAGS)
    for section in config.sections():
        if section not in known:
            warnings.warn(f"Unknown section [{section}] in the config file is ignored.", stacklevel=3)
            continue
        for key in config[section]:
            if key not in known[section]:
                warnings.warn(f"Unknown entry '{key}' in [{section}] is ignored.", stacklevel=3)


# ---------------------------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------------------------
def parse_key_mapping(config):
    """All reader keys, None where not given."""
    keymap = {k: None for k in KEY_MAPPING_KEYS}
    for k in KEY_MAPPING_KEYS:
        value = _raw(config, "key_mapping", k)
        if value is not None:
            keymap[k] = parse_list_or_str(value)
    return keymap


def parse_fields_to_export(config):
    """All export flags, False where not given"""
    return {k: get(config, "fields_to_export", k, bool, False) for k in EXPORT_FLAGS}


def parse_grid_info(config):
    d = DEFAULTS["grid_info"]
    grid_info = {
        "grid_dimension": get(config, "grid_info", "grid_dimension", int),
        "grid_axes": get(config, "grid_info", "grid_axes", str),
    }
    for key in d:
        if key.endswith(("_min", "_max", "_transect")):
            grid_info[key] = get(config, "grid_info", key, float, d[key])
        elif key.endswith("_periodic"):
            grid_info[key] = get(config, "grid_info", key, bool, d[key])
    limits_given = any(grid_info[f"{a}_{m}"] is not None for a in AXES for m in ("min", "max"))
    grid_info["automatic_grid"] = get(config, "grid_info", "automatic_grid", bool, not limits_given)
    return grid_info


def parse_simulation_box(config):
    """[(xmin, xmax), (ymin, ymax), (zmin, zmax)], None where not given; None if nothing is given."""
    box = [get(config, "simulation_box", f"{a}_range", "range", None) for a in AXES]
    return None if all(b is None for b in box) else box


# ---------------------------------------------------------------------------------------------
# config loader
# ---------------------------------------------------------------------------------------------
def load_config(path_to_ini="config.ini") -> dict:
    """
    Read a pysammos .ini file and return the keyword arguments of CoarseGraining,
    so that  CoarseGraining(**load_config("config.ini"))  works.

    Missing optional entries take the values in ``DEFAULTS``; missing required entries
    (``REQUIRED``) raise a ValueError listing them.
    """
    config = configparser.ConfigParser()
    config.optionxform = str                 # keep the case of the entries (key mapping)
    if not config.read(path_to_ini):
        raise FileNotFoundError(f"Config file not found: {path_to_ini}")

    check_required(config)
    warn_unknown(config)

    t0 = get(config, "timesteps", "t0", int)
    kwargs = {
        "particle_path": get(config, "paths", "particles_path"),
        "contacts_path": get(config, "paths", "contacts_path"),
        "output_path": get(config, "paths", "output_path", str, DEFAULTS["paths"]["output_path"]),
        "start_timestep": t0,
        "end_timestep": get(config, "timesteps", "tf", int, t0),
        "dt_time_step": get(config, "timesteps", "dt", int, DEFAULTS["timesteps"]["dt"]),
        "t_pad": get(config, "timesteps", "t_pad", int),
        "DEM_software": get(config, "DEM_software", "name"),
        "DEM_keymap": parse_key_mapping(config),
        "grid_info": parse_grid_info(config),
        "simulation_box": parse_simulation_box(config),
        "weight_function": get(config, "smoothing_function", "type", str, DEFAULTS["smoothing_function"]["type"]),
        "fields_to_export": parse_fields_to_export(config),
        "ignore_phases": get(config, "flags", "partialignore", bool, DEFAULTS["flags"]["partialignore"]),
        "time_average": get(config, "output_options", "time_average", bool, DEFAULTS["output_options"]["time_average"]),
        "vkthdf_output": get(config, "output_options", "vkthdf_output", bool, DEFAULTS["output_options"]["vkthdf_output"]),
        "h5_output": get(config, "output_options", "h5_output", bool, DEFAULTS["output_options"]["h5_output"]),
    }

    # optional settings: only passed if given, so the defaults stay in CoarseGraining.__init__
    method = get(config, "advanced", "velocity_gradient_method")
    if method is not None:
        kwargs["velocity_gradient_method"] = method
    factor = get(config, "advanced", "search_sampling_factor", int)
    if factor is not None:
        kwargs["search_sampling_factor"] = factor

    if not any(kwargs["fields_to_export"].values()):
        warnings.warn("No field is set to True in [fields_to_export]: nothing will be exported.", stacklevel=2)
    return kwargs