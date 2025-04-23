"""Module for solving 2D plane truss problems."""

import numpy as np
import matplotlib.pyplot as plt

class PlaneTrussProblem:
    """Class for solving 2D plane truss problems."""
    
    def __init__(self, nodes, elements, elasticity_modulus, cross_sectional_area):
        """
        Initialize the plane truss problem.

        Parameters:
        nodes (list of tuples): List of node coordinates (x, y).
        elements (list of tuples): List of elements defined by node indices.
        elasticity_modulus (float): Elasticity modulus of the material.
        cross_sectional_area (float): Cross-sectional area of the truss members.
        
        Example:
        >>> nodes = [(0, 0), (1, 0), (1, 1), (0, 1)]
        >>> elements = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
        >>> elasticity_modulus = 210e9
        >>> cross_sectional_area = 0.01
        >>> truss = PlaneTrussProblem(nodes, elements, elasticity_modulus, cross_sectional_area)
        
        """
        
        self.nodes = np.array(nodes)
        self.elements = np.array(elements)
        self.E = elasticity_modulus
        self.A = cross_sectional_area
        self.num_nodes = len(nodes)
        self.num_elements = len(elements)
        
        # Perform initial checks
        if self.num_nodes < 2:
            raise ValueError("At least two nodes are required.")
        if self.num_elements < 1:
            raise ValueError("At least one element is required.")
        if self.nodes.shape[1] != 2:
            raise ValueError("Node coordinates must be 2D (x, y).")
        if self.elements.shape[1] != 2:
            raise ValueError("Elements must be defined by two node indices.")
        
        # Initialize global stiffness matrix
        self.K_global = np.zeros((2 * self.num_nodes, 2 * self.num_nodes))
        
        # Compute the plane truss elements' lengths
        self.L = np.zeros(self.num_elements)
        for i, (n1, n2) in enumerate(self.elements):
            x1, y1 = self.nodes[n1]
            x2, y2 = self.nodes[n2]
            self.L[i] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
        # Compute the angles of the elements in degrees
        self.angles = np.zeros(self.num_elements)
        for i, (n1, n2) in enumerate(self.elements):
            x1, y1 = self.nodes[n1]
            x2, y2 = self.nodes[n2]
            self.angles[i] = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # Angles in degrees
            if self.angles[i] < 0:
                self.angles[i] += 360
            
    def plot_plane_truss(self, show_node_indices=True, show_element_indices=True):
        """
        Function to plot the plane truss:
        - Nodes are represented as circles and optionally labeled with their indices.
        - Elements are represented as lines connecting the nodes and optionally labeled with their indices.

        Parameters:
        show_node_indices (bool): If True, display the node indices on the plot.
        show_element_indices (bool): If True, display the element indices on the plot.
        
        """
        plt.figure(figsize=(8, 6))
        
        # Plot elements
        for i, (n1, n2) in enumerate(self.elements):
            x1, y1 = self.nodes[n1]
            x2, y2 = self.nodes[n2]
            plt.plot([x1, x2], [y1, y2], 'b-', lw=2)
            if show_element_indices:
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                plt.text(mid_x, mid_y, f"E{i}", color='blue', fontsize=12, ha='center', va='center',
                         bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'))
        
        # Plot nodes
        for i, (x, y) in enumerate(self.nodes):
            plt.plot(x, y, 'ro', markersize=8)
            if show_node_indices:
                plt.text(x + 0.05, y + 0.05, f"N{i}", color='red', fontsize=12, ha='left', va='bottom',
                         bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))
        
        plt.title("Plane Truss Structure")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.axis('equal')
        plt.grid(True, alpha=0.5, linestyle='--', linewidth=0.5)
        plt.show()
        
    def get_angle(self, element_index):
        """
        Get the angle of the element in degrees.

        Parameters:
        element_index (int): Index of the element.

        Returns:
        float: Angle of the element in degrees.
        
        """
        if element_index < 0 or element_index >= self.num_elements:
            raise ValueError("Element index out of range.")
        
        return self.angles[element_index]
    
    def get_length(self, element_index):
        """
        Get the length of the element.

        Parameters:
        element_index (int): Index of the element.

        Returns:
        float: Length of the element.
        
        """
        if element_index < 0 or element_index >= self.num_elements:
            raise ValueError("Element index out of range.")
        
        return self.L[element_index]
    
    def ElementStiffness(self, element_index):
        """
        Compute the element stiffness matrix for a given element.
        The size of the element stiffness matrix is 4x4.

        Parameters:
        element_index (int): Index of the element.

        Returns:
        np.ndarray: Stiffness matrix of the element.
        
        """
        if element_index < 0 or element_index >= self.num_elements:
            raise ValueError("Element index out of range.")
        
        x = self.angles[element_index] * np.pi / 180  # Convert angle to radians
        c = np.cos(x)
        s = np.sin(x)
        L = self.L[element_index]
        k = (self.E * self.A / L) * np.array([[c**2, c*s, -c**2, -c*s],
                                               [c*s, s**2, -c*s, -s**2],
                                               [-c**2, -c*s, c**2, c*s],
                                               [-c*s, -s**2, c*s, s**2]])
        return k
    
    def assemble_global_stiffness(self):
        """
        Assemble the global stiffness matrix for the entire truss structure.

        Returns:
        np.ndarray: Global stiffness matrix.
        
        """
        for i, (n1, n2) in enumerate(self.elements):
            k_local = self.ElementStiffness(i)
            dof_indices = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
            for a in range(4):
                for b in range(4):
                    self.K_global[dof_indices[a], dof_indices[b]] += k_local[a, b]
        
        return self.K_global
    
    def add_inclined_support(self, node_index, angle):
        """
        Add an inclined support to the truss at a specific node.

        Parameters:
        node_index (int): Index of the node where the support is applied.
        angle (float): Angle of the inclined support in degrees.
        
        """
        if node_index < 0 or node_index >= self.num_nodes:
            raise ValueError("Node index out of range.")
        
        transformation_matrix = np.eye(2 * self.num_nodes)
        x = angle * np.pi / 180  # Convert angle to radians
        transformation_matrix[2*node_index, 2*node_index] = np.cos(x)
        transformation_matrix[2*node_index, 2*node_index + 1] = np.sin(x)
        transformation_matrix[2*node_index + 1, 2*node_index] = -np.sin(x)
        transformation_matrix[2*node_index + 1, 2*node_index + 1] = np.cos(x)
        self.K_global = transformation_matrix @ self.K_global @ (transformation_matrix.T)
    
    def set_external_constraints(self, constrained_dofs):
        """
        Partition the global stiffness matrix by eliminating the rows and columns corresponding to the constrained degrees of freedom.

        Parameters:
        constrained_dofs (list of int): List of constrained degrees of freedom.

        Returns:
        np.ndarray: Partitioned global stiffness matrix.
        
        """
        free_dofs = np.setdiff1d(np.arange(2 * self.num_nodes), constrained_dofs)
        return self.K_global[np.ix_(free_dofs, free_dofs)]
    
    def solve(self, external_forces, constrained_dofs):
        """
        Solve the truss problem for the given external forces and constraints.

        Parameters:
        external_forces (np.ndarray): External forces applied to the nodes.
        constrained_dofs (list of int): List of constrained degrees of freedom.

        Returns:
        np.ndarray: Displacements of the nodes.
        
        """
        
        # Partition the global stiffness matrix
        K_free = self.set_external_constraints(constrained_dofs)
        
        # Solve for displacements
        free_dofs = np.linalg.solve(K_free, external_forces)
        displacements = np.zeros(2 * self.num_nodes)
        free_dof_indices = np.setdiff1d(np.arange(2 * self.num_nodes), constrained_dofs)
        displacements[free_dof_indices] = free_dofs
        self.displacements = displacements
        return displacements
    
    def get_displacement(self, node_index):
        """
        Get the displacement of a specific node.

        Parameters:
        node_index (int): Index of the node.

        Returns:
        tuple: Displacement of the node in x and y directions.
        
        """
        if node_index < 0 or node_index >= self.num_nodes:
            raise ValueError("Node index out of range.")
        
        return self.displacements[2*node_index], self.displacements[2*node_index + 1]
    
    def get_reaction_forces(self):
        """
        Calculate the reaction forces.

        Returns:
        np.ndarray: Reaction forces.
        
        """
        reaction_forces = np.zeros(2 * self.num_nodes)
        reaction_forces = self.K_global @ self.displacements
        self.reaction_forces = reaction_forces
        return reaction_forces
    
    def get_element_stress(self, element_index):
        """
        Calculate the stress in a specific element.

        Parameters:
        element_index (int): Index of the element.

        Returns:
        float: Stress in the element.
        
        """
        if element_index < 0 or element_index >= self.num_elements:
            raise ValueError("Element index out of range.")
        
        theta = self.angles[element_index]
        L = self.L[element_index]
        n1, n2 = self.elements[element_index]
        displacement_vector = np.array([self.displacements[2*n1], self.displacements[2*n1 + 1],
                                        self.displacements[2*n2], self.displacements[2*n2 + 1]])
        
        x = theta * np.pi / 180
        c = np.cos(x)
        s = np.sin(x)
        
        stress = (self.E / L) * np.array([[-c, -s, c, s]]) @ displacement_vector
        return stress[0]
    
    def get_element_force(self, element_index):
        """
        Calculate the force in a specific element.

        Parameters:
        element_index (int): Index of the element.

        Returns:
        float: Force in the element.
        
        """
        if element_index < 0 or element_index >= self.num_elements:
            raise ValueError("Element index out of range.")
        
        stress = self.get_element_stress(element_index)
        force = stress * self.A
        return force
    