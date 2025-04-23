"""Module for solving spring problems."""

import numpy as np

class SpringProblem:
    """Class for solving spring problems."""

    def __init__(self, nodes, elements, stiffness):
        """
        Initialize the spring problem.

        Parameters:
        nodes (list of tuples): List of node coordinates (x, y).
        elements (list of tuples): List of elements defined by node indices.
        stiffness (list of floats): List of spring stiffness values for each element.
        
        Example:
        >>> nodes = [(0, 0), (1, 0), (1, 1), (0, 1)]
        >>> elements = [(0, 1), (1, 2), (2, 3), (3, 0)]
        >>> stiffness = [100, 200, 150, 300]
        >>> spring = SpringProblem(nodes, elements, stiffness)
        
        """
        
        self.nodes = np.array(nodes)
        self.elements = np.array(elements)
        self.stiffness = np.array(stiffness)
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
        self.K_global = np.zeros((self.num_nodes, self.num_nodes))
        
    def ElementStiffness(self, element_index):
        """
        Calculate the local stiffness matrix for a given element.

        Parameters:
        element_index (int): Index of the element.

        Returns:
        np.ndarray: Local stiffness matrix for the element.
        
        """
        if element_index < 0 or element_index >= self.num_elements:
            raise ValueError("Element index out of range.")
        
        k = self.stiffness[element_index]
        local_stiffness = np.array([[k, -k], [-k, k]])
        return local_stiffness
    
    def assemble_global_stiffness(self):
        """
        Assemble the global stiffness matrix from local stiffness matrices.

        Returns:
        np.ndarray: Global stiffness matrix.
        
        """
        for i, (n1, n2) in enumerate(self.elements):
            k_local = self.ElementStiffness(i)
            self.K_global[n1, n1] += k_local[0, 0]
            self.K_global[n1, n2] += k_local[0, 1]
            self.K_global[n2, n1] += k_local[1, 0]
            self.K_global[n2, n2] += k_local[1, 1]
            
        return self.K_global
    
    def set_external_constraints(self, constrained_dofs):
        """
        Partition the global stiffness matrix by eliminating the rows and columns corresponding to the constrained degrees of freedom.

        Parameters:
        constrained_dofs (list of int): List of constrained degrees of freedom.

        Returns:
        np.ndarray: Partitioned global stiffness matrix.
        
        """
        free_dofs = np.setdiff1d(np.arange(self.num_nodes), constrained_dofs)
        return self.K_global[np.ix_(free_dofs, free_dofs)]
    
    def solve(self, external_forces, constrained_dofs):
        """
        Solve the spring problem for the given external forces and constraints.

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
        displacements = np.zeros(self.num_nodes)
        free_dof_indices = np.setdiff1d(np.arange(self.num_nodes), constrained_dofs)
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
        
        return self.displacements[node_index]
    
    def get_reaction_forces(self):
        """
        Calculate the reaction forces.

        Returns:
        np.ndarray: Reaction forces.
        
        """
        reaction_forces = np.zeros(self.num_nodes)
        reaction_forces = np.dot(self.K_global, self.displacements)
        self.reaction_forces = reaction_forces
        return reaction_forces
    
    def get_element_force(self, element_index):
        """
        Calculate the force vector in a specific element.

        Parameters:
        element_index (int): Index of the element.

        Returns:
        np.ndarray: Force vector in the element.
        
        """
        if element_index < 0 or element_index >= self.num_elements:
            raise ValueError("Element index out of range.")
        
        n1, n2 = self.elements[element_index]
        u1 = self.displacements[n1]
        u2 = self.displacements[n2]
        displacement_vector = np.array([u1, u2])
        force_vector = self.ElementStiffness(element_index).dot(displacement_vector)
        return force_vector
    