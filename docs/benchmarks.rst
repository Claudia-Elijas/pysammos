Benchmarks
==========

This sections walks through the developer users through therequired steps and configuration for running benchmarking tests so that they obtain consistent and comparable results.  
Any future benchmarking must follow this procedure exactly unless explicitly agreed otherwise.


.. toctree::
   :maxdepth: 2

   benchmarks/monodisperse_cube

.. figure:: /_static/benchmark_monodisperse.png
   :width: 600px
   :align: left
   :alt: Benchmark comparison of coarse-graining software.

   **Benchmark of a monodisperse cubic lattice.**
   A monodisperse cubic lattice of grains with diameter 0.02 m and density
   2500 kg m\ :sup:`-3`, evaluated at the point :math:`P=(0,0,0.42725)` m.
   The system was coarse-grained with Pysammos (coloured triangles) and
   other coarse-graining software (void shapes): EDEM, Granulysed, Iota,
   and MercuryCG. Different smoothing functions (Gaussian, Heaviside,
   and Lucy) are compared where available.
   Panels show examples of a scalar (a), a vector (b–c), and tensors
   (d–g), corresponding to volume fraction, velocity, and contact and
   kinetic tensors.

