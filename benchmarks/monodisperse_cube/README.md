# Benchmarking README

## Purpose

This document defines the required steps and configuration for running benchmarking tests so that all developers obtain consistent and comparable results.  
Any future benchmarking must follow this procedure exactly unless explicitly agreed otherwise.

he script that performs the Pysammos Coarse-graining of this benchmark simulation is `pysammos/benchmarks/monodisperse_cube/run_sweep_pysammos.ipynb`. The plotting of the results against other benchmarks is located at  `pysammos/benchmarks/monodisperse_cube/plot.ipynb`.

---

## Before Running Pysammos 

### 1. Contact Data Source (Critical)

For benchmarking, the point data reader must use JP’s contact data reader.

**Required code change:**

- Go to the module: `pysammos.data_read.mfix.point_data`
- Open the function: `contacts`
- Comment out the default contact reader code and uncomment the code block immediately below the comment: #only to benchmark with JP's contact data reader, such that it looks as below: 

```python
# poly_output = InputConnection.GetOutput()
# F_ij = get_point_data_variable(Force_ij_string, poly_output).astype(np.float32) if Force_ij_string else None
# Particle_i = get_point_data_variable(Particle_i_string, poly_output).astype(np.float32) if Particle_i_string else None
# Particle_j = get_point_data_variable(Particle_j_string, poly_output).astype(np.float32) if Particle_j_string else None
# Contact_ij = get_point_data_variable(Contact_ij_string, poly_output).astype(np.float32) if Contact_ij_string else None

# only to benchmark with JP's contact data reader
poly_output = InputConnection.GetOutput().GetCellData(); print("Contact Data loaded as Cell Data")
F_ij = vtk_to_numpy(poly_output.GetArray(Force_ij_string)).astype(np.float32)
contact_ids = vtk_to_numpy(poly_output.GetArray(Particle_i_string)).astype(np.float32)
Particle_i = contact_ids[:, 0]
Particle_j = contact_ids[:, 1]
Contact_ij = vtk_to_numpy(poly_output.GetArray(Contact_ij_string)).astype(np.float32)
```

---

### 2. Smoothing functions c/w (Critical)

The benchmarks use a specific range-to-width ratio. For consistency, in the function `calc_cutoff` of the module `pysammos.spatial_weights_resolution`, change them to the following: 

```python
if function == 'Lucy':
    c = w
elif function == 'Gaussian':
    c = 3*w
elif function == 'HeavySide':
    c = w
```

---

### 4. Kernel Restart 

If any code changes are made before re-running the benchmark, you mush restart the notebook. 

- Select **“Restart & Run All”**

This ensures no cached variables are used and results are fully reproducible.

---

## Coarse-Graining of Benchmark Model  

### 1. Sensitivity Setting

- The sensitivity parameter must be increased to at least: sensitivity ≥ 10000
- This is required to obtain a smooth match across all w/d values.

---

### 2. The tested CG widths 

The CG widths for which the sweep is carried out must be the following: 

```python
# cg values used by JP: 
cgWidths = np.array([
  0.41666667, 0.5, 0.58333333, 0.66666667, 0.75,
  0.83333333, 0.91666667, 1.0, 1.08333333, 1.16666667,
  1.25, 1.33333333, 1.5, 1.75, 2.0,
  2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0
])
```

Each smoothing function uses a different multiple of the baseline CG widths.

```python
dp = 0.02  # particle diameter in m

Lucy_widths = cgWidths * dp * 3
Gaussian_widths = cgWidths * dp 
HeavySide_widths = cgWidths * dp * 3
```
---

### 3. Reference Point Location

All benchmark comparisons must be performed at the following reference point: P(x,y,z) = [0.0, 0.0, 0.42725]

This point must be used as the center of a 3×3×3 coarse-graining grid by entering it in the `CG.sweep_CG_widths()` function. 
The index of the point within the grid is printed when running that function. Tends to be index=13. 

```python
CG.sweep_CG_widths(
  w_d=eval(f"{cg_func}_widths") / 0.02,
  center=np.array([0.0, 0.0, 0.42725])
)
```


