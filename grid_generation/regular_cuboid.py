import numpy as np


class Grid_Generation: 
    def __init__(self, smoothing_length, particle_bounds, grid_dimensions, grid_axes, max_particle_diameter, automatic_range, custom_grid_range, custom_grid_transects):
        self.c = smoothing_length
        self.bounds = particle_bounds
        self.dimensions = grid_dimensions
        self.axes = grid_axes
        self.dmax = max_particle_diameter
        self.automatic_range = automatic_range
        self.custom_grid_range = custom_grid_range
        self.custom_grid_transects = custom_grid_transects
    
    # Automatic range determination
    def Automatic_Range(self):

            delta = 2.5 * self.c + 0.5 * self.dmax # distance from the boundary of the domain
            
            min = np.zeros(3) ; max = np.zeros(3)

            for i in range(3):
                min[i] = self.bounds[i,0] + delta
                max[i] = self.bounds[i,1] - delta
            
            x_range = [min[0], max[0]] 
            y_range = [min[1], max[1]] 
            z_range = [min[2], max[2]] 

            return x_range, y_range, z_range
    
    # Create grid points
    @staticmethod
    def Create_grid_points(X_range, Y_range, Z_range, X_transect, Y_transect, Z_transect, c, high_res_scaling=1.5, dimensions=3, axes='xyz'):


        # General grid parameters
        # X - dimension
        if X_range is not None and all(v is not None for v in X_range):
            #print("X range",X_range)
            xmin, xmax = X_range # unpack the range
            Nx = int(np.ceil(high_res_scaling * (xmax - xmin) / c)+1) # calculate number of grid points
            if Nx <= 1:
                raise ValueError("The number of grid points in x direction is <= 1. Increase X_range")
            dx = (xmax - xmin) / (Nx - 1) # calculate spacing of the grid
            x = np.linspace(xmin,xmax,Nx) # generate grid points
        else: 
            x = None ; dx = None ; Nx = None
        # -------------------------------------------------
        # Y - dimension
        if Y_range is not None and all(v is not None for v in Y_range):
            ymin, ymax = Y_range # unpack the range
            #print("Y range",Y_range)
            Ny = int(np.ceil(high_res_scaling * (ymax - ymin) / c)+1) # calculate number of grid points
            if Ny <= 1:
                raise ValueError("The number of grid points in y direction is <= 1. Increase Y_range")
            dy = (ymax - ymin) / (Ny - 1) # calculate spacing of the grid
            y = np.linspace(ymin,ymax,Ny) # generate grid points
            #print("dy")
        else:
            y = None ; dy = None ; Ny = None
        # -------------------------------------------------
        # Z - dimension
        if Z_range is not None and all(v is not None for v in Z_range):
            #print("Z range" ,Z_range)
            zmin, zmax = Z_range # unpack the range
            #print(zmin, zmax)
            Nz = int(np.ceil(high_res_scaling * (zmax - zmin) / c)+1) # calculate number of grid points
            if Nz <= 1:
                raise ValueError("The number of grid points in z direction is <= 1. Increase Z_range")
            dz = (zmax - zmin) / (Nz - 1) # calculate spacing of the grid
            z = np.linspace(zmin,zmax,Nz) # generate grid points
        else:
            z = None ; dz = None ; Nz = None
        # -------------------------------------------------

        #                      * * * * 

        # ------------------------------------------------
        # 1-D grid
        if dimensions == 1:
            if axes == 'x':
                # Check that transect values are provided
                if x is None:
                    raise ValueError("Axes x: X_range is None, please provide a value.")
                if Y_transect is None:
                    raise ValueError("Axes x: Y_transect is None, please provide a value.")
                if Z_transect is None:
                    raise ValueError("Axes x: Z_transect is None, please provide a value.")
                # Generate x-coordinates
                grid_points_x = x 
                grid_points = np.column_stack((grid_points_x, np.full_like(grid_points_x, Y_transect), np.full_like(grid_points_x, Z_transect))) # Create 3D points using broadcasting
                nodes = np.array([Nx, 1, 1])
                spacing = np.array([dx])

            elif axes == 'y':
                # Check that transect values are provided
                if y is None:
                    raise ValueError("Axes y: Y_range is None, please provide a value.")
                if X_transect is None:
                    raise ValueError("Axes y: X_transect is None, please provide a value.")
                if Z_transect is None:
                    raise ValueError("Axes y: Z_transect is None, please provide a value.")
                # Generate y-coordinates
                grid_points_y = y
                grid_points = np.column_stack((np.full_like(grid_points_y, X_transect), grid_points_y, np.full_like(grid_points_y, Z_transect)))
                nodes = np.array([1, Ny, 1])
                spacing = np.array([dy])
            
            elif axes == 'z':
                # Check that transect values are provided
                if z is None:
                    raise ValueError("Axes z: Z_range is None, please provide a value.")
                if X_transect is None:
                    raise ValueError("Axes z: X_transect is None, please provide a value.")
                if Y_transect is None:
                    raise ValueError("Axes z: Y_transect is None, please provide a value.")
                # Generate z-coordinates
                grid_points_z = z
                grid_points = np.column_stack((np.full_like(grid_points_z, X_transect), np.full_like(grid_points_z, Y_transect), grid_points_z))
                nodes = np.array([1, 1, Nz])
                spacing = np.array([dz])
            
            else:
                raise ValueError("Invalid axes for 1D grid. Must be 'x', 'y', or 'z'.")
        # -------------------------------------------------
        # 2-D grid
        elif dimensions == 2:
            
            if axes == 'xy':
                # Check that transect values are provided
                if x is None:
                    raise ValueError("Axes xy: X_range is None, please provide a value.")
                if y is None:
                    raise ValueError("Axes xy: Y_range is None, please provide a value.")
                if Z_transect is None:
                    raise ValueError("Axes xy: Z_transect is None, please provide a value.")
                # Generate grid
                #print("xy 2d")
                xx, yy = np.meshgrid(x, y, indexing='ij')  # Generate grid
                x_cg = np.reshape(xx, (Nx * Ny))  # Reshape the grid points in x direction
                y_cg = np.reshape(yy, (Nx * Ny))  # Reshape the grid points in y direction
                z_cg = np.full_like(x_cg, Z_transect)  # Create a constant array for z
                grid_points = np.column_stack((x_cg, y_cg, z_cg))  # Combine x, y, and z
                nodes = np.array([Nx, Ny, 1])
                spacing = np.array([dx, dy]) 
                #print("done")
            
            elif axes == 'xz':
                # Check that transect values are provided
                if x is None:
                    raise ValueError("Axes xz: X_range is None, please provide a value.")
                if z is None:
                    raise ValueError("Axes xz: Z_range is None, please provide a value.")
                if Y_transect is None:
                    raise ValueError("Axes xz: Y_transect is None, please provide a value.")
                # Generate grid
                xx, zz = np.meshgrid(x, z, indexing='ij')  # Generate grid
                x_cg = np.reshape(xx, (Nx * Nz))  # Reshape the grid points in x direction
                z_cg = np.reshape(zz, (Nx * Nz))  # Reshape the grid points in z direction
                y_cg = np.full_like(x_cg, Y_transect)  # Create a constant array for y
                grid_points = np.column_stack((x_cg, y_cg, z_cg))  # Combine x, y, and z
                nodes = np.array([Nx, 1, Nz])
                spacing = np.array([dx, dz])
            
            elif axes == 'yz':
                # Check that transect values are provided
                if y is None:
                    raise ValueError("Axes yz: Y_range is None, please provide a value.")
                if z is None:  
                    raise ValueError("Axes yz: Z_range is None, please provide a value.")
                if X_transect is None:
                    raise ValueError("Axes yz: X_transect is None, please provide a value.")
                # Generate grid
                yy, zz = np.meshgrid(y, z, indexing='ij')  # Generate grid
                y_cg = np.reshape(yy, (Ny * Nz))  # Reshape the grid points in y direction
                z_cg = np.reshape(zz, (Ny * Nz))  # Reshape the grid points in z direction
                x_cg = np.full_like(y_cg, X_transect)  # Create a constant array for x
                grid_points = np.column_stack((x_cg, y_cg, z_cg))  # Combine x, y, and z
                nodes = np.array([1, Ny, Nz])
                spacing = np.array([dy, dz])
            
            else:
                raise ValueError("Invalid axes for 2D grid. Must be 'xy', 'xz', or 'yz'.")
        # --------------------------------------------------
        # 3-D grid
        elif dimensions == 3:
            xx,yy,zz = np.meshgrid(x,y,z,indexing='ij') # generate grid
            x_cg = np.reshape(xx,(Nx*Ny*Nz))
            y_cg = np.reshape(yy,(Nx*Ny*Nz))
            z_cg = np.reshape(zz,(Nx*Ny*Nz))
            grid_points = np.transpose([x_cg,y_cg,z_cg]) # prepare the array of CG points
            nodes = np.array([Nx, Ny, Nz])
            spacing = np.array([dx, dy, dz])
        # --------------------------------------------------
        # Raise error if dimensions are not 1, 2, or 3
        else:
            raise ValueError("Invalid dimensions. Must be 1, 2, or 3.")
                
        return grid_points, nodes, spacing
    
    # Make grid
    def Generate(self):
     
            # 1. get ranges and or transects
            if self.automatic_range == True:
                print("Generating Grid with Automatic Grid Ranges")
                x_range_, y_range_, z_range_ = self.Automatic_Range()
                # 1D grid
                if self.dimensions == 1: 
                    if self.axes == 'x': 
                        y_transect = y_range_[0] + 0.5 * (y_range_[1] - y_range_[0]) ; y_range = None
                        z_transect = z_range_[0] + 0.5 * (z_range_[1] - z_range_[0]) ; z_range = None
                        x_transect = None ; x_range = x_range_
                        print(f"Automatic bounds: x_range = {x_range}, y_transect = {y_transect}, z_transect = {z_transect}")
                    elif self.axes == 'y':
                        x_transect = x_range_[0] + 0.5 * (x_range_[1] - x_range_[0]) ; x_range = None
                        z_transect = z_range_[0] + 0.5 * (z_range_[1] - z_range_[0]) ; z_range = None
                        y_transect = None ; y_range = y_range_
                        print(f"Automatic bounds: y_range = {y_range}, x_transect = {x_transect}, z_transect = {z_transect}")
                    elif self.axes == 'z':
                        x_transect = x_range_[0] + 0.5 * (x_range_[1] - x_range_[0]) ; x_range = None
                        y_transect = y_range_[0] + 0.5 * (y_range_[1] - y_range_[0]) ; y_range = None
                        z_transect = None ; z_range = z_range_
                        print(f"Automatic bounds: z_range = {z_range}, x_transect = {x_transect}, y_transect = {y_transect}")
                    else:
                        raise ValueError("Invalid axes for 1D grid. Must be 'x', 'y', or 'z'.")
                # 2D grid
                elif self.dimensions == 2:
                    if self.axes == 'xy':
                        z_transect = z_range_[0] + 0.5 * (z_range_[1] - z_range_[0]) ; z_range = None
                        x_transect = None ; x_range = x_range_
                        y_transect = None ; y_range = y_range_
                        print(f"Automatic bounds: x_range = {x_range}, y_range = {y_range}, z_transect = {z_transect}")
                    elif self.axes == 'xz':
                        y_transect = y_range_[0] + 0.5 * (y_range_[1] - y_range_[0]) ; y_range = None
                        z_transect = None ; z_range = z_range_
                        x_transect = None ; x_range = x_range_
                        print(f"Automatic bounds: x_range = {x_range}, y_transect = {y_transect}, z_range = {z_range}")
                    elif self.axes == 'yz':
                        x_transect = x_range_[0] + 0.5 * (x_range_[1] - x_range_[0]) ; x_range = None
                        z_transect = None ; z_range = z_range_
                        y_transect = None ; y_range = y_range_
                        print(f"Automatic bounds: y_range = {y_range}, x_transect = {x_transect}, z_range = {z_range}")
                    else:
                        raise ValueError("Invalid axes for 2D grid. Must be 'xy', 'xz', or 'yz'.")
                elif self.dimensions == 3:
                    x_range = x_range_ ; y_range = y_range_ ; z_range = z_range_
                    x_transect = None ; y_transect = None ; z_transect = None
                    print(f"Automatic bounds: x_range = {x_range}, y_range = {y_range}, z_range = {z_range}")
                            
            elif self.automatic_range == False:
                print("Generating Grid with Customised Grid Ranges")
                x_0, x_1, y_0, y_1, z_0, z_1 = self.custom_grid_range
                x_tr, y_tr, z_tr = self.custom_grid_transects
                # flag that the user has not provided ranges bigger than the domain
                # b = self.bounds
                # print(f" bounds {b}")
                # # Adjust ranges and checks based on dimensions
                # if x_0 and x_1 is not None:
                #     if x_0 < b[0,0] or x_1 > b[0,1]:
                #         raise ValueError("The provided X ranges are bigger than the domain.")
                # if y_0 and y_1 is not None:
                #     if y_0 < b[1,0] or y_1 > b[1,1]:
                #         raise ValueError("The provided Y ranges are bigger than the domain.")
                # if z_0 and z_1 is not None:
                #     if z_0 < b[2,0] or z_1 > b[2,1]:
                #         raise ValueError("The provided Z ranges are bigger than the domain.")
                # # Check that transect values are provided
                # if x_tr is not None: 
                #     if x_tr < b[0,0] or x_tr > b[0,1]:
                #         raise ValueError("The provided X transect is outside the domain.")
                # if y_tr is not None:
                #     if y_tr < b[1,0] or y_tr > b[1,1]:
                #         raise ValueError("The provided Y transect is outside the domain.")
                # if z_tr is not None:
                #     if z_tr < b[2,0] or z_tr > b[2,1]:
                #         raise ValueError("The provided Z transect is outside the domain.")
                # put together the ranges and/or transects
                
                x_range = [x_0, x_1] ; y_range = [y_0, y_1] ; z_range = [z_0, z_1]
                y_transect = y_tr ; x_transect = x_tr ; z_transect = z_tr
                print(f"Customised grid bounds: x = {x_range}, y = {y_range}, z = {z_range}, x_transect = {x_transect}, y_transect = {y_transect}, z_transect = {z_transect}")

            # 2. generate the CG grid points
            GridPoints , Nodes, Spacing = self.Create_grid_points(X_range=x_range, Y_range=y_range, Z_range=z_range, 
                                X_transect = x_transect, Y_transect = y_transect, Z_transect = z_transect,
                                c = self.c, 
                                high_res_scaling = 2, 
                                dimensions = self.dimensions, axes = self.axes) 
        
            # 3. calculate the ranges length
            # Ranges_Length = np.array([
            #                 0 if np.all(r) is None else r[1] - r[0] 
            #                 for r in [x_range, y_range, z_range]
            #             ])
            Ranges_Length = np.array([
            0 if r is None or any(v is None for v in r) else r[1] - r[0]
            for r in [x_range, y_range, z_range]
        ])
                    
            return GridPoints, Nodes, Spacing, Ranges_Length
    

