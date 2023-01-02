# get triangles from mesh
elementType = gmsh.model.mesh.get_element_type("Triangle", 1)
triagEdgeNodes = gmsh.model.mesh.get_element_edge_nodes(elementType) #gmsh.model.mesh.getElementEdgeNodes(elementType)
triagFaceNodes = gmsh.model.mesh.get_element_face_nodes(elementType, 3)

# Edge and face tags can then be retrieved by providing their nodes:
gmsh.model.mesh.createEdges()
gmsh.model.mesh.createFaces()
edgeTags, edgeOrientations = gmsh.model.mesh.getEdges(triagEdgeNodes)
faceTags, faceOrientations = gmsh.model.mesh.getFaces(3, triagFaceNodes)

# Since element edge and face nodes are returned in the same order as the
# elements, one can easily keep track of which element(s) each edge or face is

#triag, test =gmsh.model.mesh.get_elements_by_type(elementType)
#print(edgeTags)


# %% Settings
r_1 = 2e-3                # [m]: inner radius (wire)
r_2 = 3.5e-3              # [m]: outer radius (shell)
depth = 300e-3            # [m]: Depth of wire (lz)
model_name = "coax"       # str : model name

I = 16                    # [A]   : applied current
J_0 = I/(np.pi*r_1**2)    # [A/m] : current density in the wire
mu_0 = 4*np.pi*1e-7       # [H/m] : permeability of vacuum (and inside of wire)
mu_shell = 5*mu_0         # [H/m] : permeability of shell

# Handling of gmsh data in Python context. (Provide to students)
# This implementation is well suited for small meshes.
@dataclass
class Mesh:
    node_tag_data: np.ndarray
    node_data: np.ndarray
    elementTypes: np.ndarray
    element_tags: np.ndarray
    nodeTags_elements: np.ndarray

    # treat nodes
    @property
    def num_node(self):
        # number of nodes
        return int(len(self.node_data) / 3)

    @property
    def node(self):
        # nodes
        node = np.reshape(self.node_data, (self.num_node, 3))
        # coordinates of nodes. x-coordinate in first column
        # and y-coordinate in second column
        node = node[:, 0:2]
        return node

    @property
    def node_tag(self):
        # ID of nodes
        node_tag = self.node_tag_data - np.ones(len(self.node_tag_data))
        node_tag = node_tag.astype('int')
        np.put_along_axis(self.node, np.c_[node_tag, node_tag], self.node, axis=0)
        return node_tag

        # treat elements

    @property
    def ind_elements(self):
        # index of elements
        return np.where(self.elementTypes == 2)[0]

    @property
    def elements(self):
        # elements
        return np.array(self.nodeTags_elements[self.ind_elements[0]])

    @property
    def num_elements(self):
        # number of elements
        return int(len(self.elements) / 3)

    @property
    def elem_to_node(self):
        # Associate elements (triangles) and their respective nodes.
        # Connection between elements and nodes.
        # Each line contains the indices of the contained nodes
        # OUTPUT: Array of N x 3 NODES
        elem_to_node = np.reshape(self.elements, (self.num_elements, 3)) - np.ones(
            (self.num_elements, 1))
        elem_to_node = elem_to_node.astype('int')
        return elem_to_node

    @property
    def all_edges(self):
        # Contains all edges
        return np.r_[self.elem_to_node[:, [0, 1]],
                     self.elem_to_node[:, [1, 2]],
                     self.elem_to_node[:, [0, 2]]]

    @property
    def edge_to_node(self):
        # Associate edges and their respective nodes.
        return np.unique(np.sort(self.all_edges), axis=0)

def entity_in_physical_group(physical_group_data: dict, entity2node: np.ndarray, identifier):
    """
    Computes the indices of all entities that are in the physical group
    specified by the identifier

    Parameters
    ----------
    physical_group_data : dict
        dict with physical groups. Key is the ID of the group and value is a tuple with (dimension, name, indices of all nodes)
    entity2node : np.ndarray
        (K,N) array with K entities and N nodes per entity
    identifier : int or str
        Identifier of the physical group. The ID or the name of the group are accepted

    Returns
    -------
    entity_in_physical_group : np.ndarray
        (M,) array. With M being the number of entities in the physical group
    """
    if type(identifier) is str:
        for p in physical_group_data.keys():
            if physical_group_data[p][1] == identifier:
                identifier = p

    out = -1 * np.ones(entity2node.shape[0], dtype=int)
    for k in range(entity2node.shape[0]):
        if np.isin(entity2node[k, :], physical_group_data[identifier][2]).all():
            out[k] = k

    return out[out != -1]