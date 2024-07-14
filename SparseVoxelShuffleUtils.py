import torch
import spconv
from torch import nn
import spconv.pytorch as spconv

from SparseVoxelShuffle import *

class VoxelUnshuffleInvConv3D(nn.Module):
    """
    VoxelUnshuffle reverses the upscaling process of VoxelShuffle by redistributing the features
    from upscaled positions back to the original tensor positions according to the `ind_dict` map.

    Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale (int): The factor by which the resolution of each dimension was originally increased.
        pairing_style (str): The pairing style used in the shuffle process, either 'sequential' or 'strided'.
        layout_style (str): The layout style used in the shuffle process.
        indice_key (str): The key for the indice dict used in spconv.
    """
    def __init__(self, in_channels, out_channels, scale, pairing_style='sequential', layout_style='xyz', indice_key=None):
        super(VoxelUnshuffleInvConv3D, self).__init__()
        self.scale = scale
        self.pairing_style = pairing_style
        self.layout_style = layout_style
        self.indice_key = indice_key
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Validate in_channels
        assert in_channels % (scale ** 3) == 0, "in_channels should be divisible by scale^3"
        assert self.indice_key is not None, "You must determine the indice key of sparse tensor"
        
        self.block_size = self.scale ** 3
        self.shuffle = VoxelShuffle(upscale_factor=scale, pairing_style=pairing_style, layout_style=layout_style)

        self.weights_list = nn.ParameterList(
            [nn.Parameter(torch.randn(in_channels)) for _ in range(self.block_size * out_channels)]
        )

    def forward(self, x):
        """
        Reverse the shuffle operation to restore the original tensor configuration.

        :param x: The SparseConvTensor after shuffling.
        :return: A new SparseConvTensor with indices and features reshuffled to their original locations.
        """
        
        # Apply VoxelShuffle to x
        device = x.features.device
        self.shuffle = self.shuffle.to(device)
        shuffled_x, vs_dict = self.shuffle(x)
        
        # Initialize a new tensor to hold the unshuffled features
        out_indices = x.indice_dict[self.indice_key].indices
        out_features = torch.zeros((out_indices.shape[0], self.out_channels))
        new_in_channels = shuffled_x.features.shape[1]
        
        inv_conv_mapping = x.indice_dict[self.indice_key].pair_fwd # Record the mapping information        
        self.weights_list = self.weights_list.to(device)
        
        # Iterate over the indice_dict and apply convolution weights
        for line_idx in range(inv_conv_mapping.size(1)):
            start_idx = line_idx * self.block_size
            end_idx = start_idx + self.block_size
            
            mapping = inv_conv_mapping[:,line_idx]
            features_block = shuffled_x.features[start_idx:end_idx]
            valid_mask = mapping != -1

            for i in range(self.out_channels):
                    weight = self.weights_list[i * self.block_size:(i + 1) * self.block_size]
                    for j, map_val in enumerate(mapping):
                        if map_val != -1:
                            out_features[map_val, i] = torch.sum(weight[j] * features_block.view(-1))
        
        # Create a new SparseConvTensor with the reshuffled features and original indices
        out_tensor = spconv.SparseConvTensor(features=out_features, indices=out_indices, spatial_shape=x.spatial_shape, batch_size=x.batch_size)
        
        return out_tensor