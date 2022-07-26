"""A convenience class for generating multiple protein structures simultaneously."""

import numpy as np
import torch
from sidechainnet.StructureBuilder import StructureBuilder 
from sidechainnet.build_info import NUM_COORDS_PER_RES
from sidechainnet.sequence import VOCAB



class BatchedStructureBuilder(object):
    """BatchedStructureBuilder enables users to generate a batch of protein structures."""

    def __init__(self, seq, device, ang_batch=None, crd_batch=None, nerf_method="standard"):
        """Construct a object capable of generating batches of StructureBuilders.
        A BatchedStructureBuilder essentially is a container for multiple StructureBuilder
        objects, one for each protein in the given batch. As such, it implements much
        of the same functionality, but simply iterates over multiple proteins.
        Args:
            seq: A string of length L describing the sequences
                of the proteins in this batch.
            ang_batch (torch.float32 tensor, optional): Optional tensor containing angles
                that represent protein structure. Defaults to None.
            crd_batch (torch.float32 tensor, optional): Optional tensor containing
                cartesian coordinates that represent protein structure. Defaults to None.
            nerf_method (str, optional): Which NeRF implementation to use. "standard" uses
                the standard NeRF formulation described in many papers. "sn_nerf" uses an
                optimized version with less vector normalizations. Defaults to
                "standard".
        Raises:
            ValueError: May raise ValueError when asked to generate structures from angles
            for structures that have missing angles.
        """
        # Validate input data
        if (ang_batch is None and crd_batch is None) or (ang_batch is not None and
                                                         crd_batch is not None):
            raise ValueError("You must provide exactly one of either coordinates (crd) "
                             "or angles (ang).")
        if ang_batch is not None:
            self.ang_or_crd_batch = ang_batch
            self.uses_coords = False
        else:
            self.ang_or_crd_batch = crd_batch
            self.uses_coords = True

        self.structure_builders = []
        self.unbuildable_structures = []
        for i, ang_or_crd in enumerate(self.ang_or_crd_batch):
            try:
                self.structure_builders.append(
                    StructureBuilder(seq, ang_or_crd, device=device, nerf_method=nerf_method))
            except ValueError as e:
                if self.uses_coords:
                    raise e
                # This means that we attempted to create StructureBuilder objects using
                # incomplete angle tensors (this is undefined/unsupported).
                self.unbuildable_structures.append(i)
                self.structure_builders.append(None)

    def build(self, hydrogen=False):
        """Build and return 3D coordinates for the previously specified protein batch.
        Returns:
            Torch.float32 tensor: returns a tensor list of 
            constructed coordinates for each protein
        """
        # modified by oz to add hydrogens and extract terminal atoms
        #terminal_coords = []
        all_coords = []
        for sb in self.structure_builders:
            sb.build()
            if hydrogen:
                sb.add_hydrogens()
                oxt = sb.terminal_atoms['OXT']
                # TODO: append OXT coords to the end
                coord = sb.coords.clone()
                coord[-1, :] = oxt
            all_coords.append(coord.unsqueeze(0))
        
        return torch.cat(all_coords, dim=0)


    def to_pdb(self, idx, path, title=None):
        """Generate protein structure & create a PDB file for specified protein.
        Args:
            idx (int): index of the StructureBuilder to visualize.
            path (str): Path to save PDB file.
            title (str, optional): Title of generated structure (default = 'pred').
        """
        if not 0 <= idx < len(self.structure_builders):
            raise ValueError("provided index is not available.")
        if idx in self.unbuildable_structures:
            self._missing_residue_error(idx)
        return self.structure_builders[idx].to_pdb(path, title)


    def _missing_residue_error(self, structure_idx):
        """Raise a ValueError describing missing residues."""
        missing_loc = np.where((self.ang_or_crd_batch[structure_idx] == 0).all(axis=-1))
        raise ValueError(f"Building atomic coordinates from angles is not supported "
                         f"for structures with missing residues. Missing residues = "
                         f"{list(missing_loc[0])}. Protein structures with missing "
                         "residues are only supported if built directly from "
                         "coordinates (also supported by StructureBuilder).")

    def __delitem__(self, key):
        raise NotImplementedError("Deletion is not supported.")

    def __getitem__(self, key):
        return self.structure_builders[key]

    def __setitem__(self, key, value):
        self.structure_builders[key] = value


