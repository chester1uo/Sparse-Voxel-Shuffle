import torch
import spconv
from torch import nn
import spconv.pytorch as spconv

def calculate_order(offsets, order, upscale_factor):
    """
    Calculate the traversal order for 3D grid cells based on a specified coordinate order.

    :param offsets: A torch.Tensor of shape (n, 3) where each row represents the (x, y, z) coordinates.
    :param order: A string or list of characters ['x', 'y', 'z'] indicating the order of coordinates.
    :param upscale_factor: An integer factor used to scale the importance of each coordinate.
    :return: A torch.Tensor containing the indices for traversal order.
    """
    # Mapping from characters to the corresponding column index in offsets
    coord_map = {'x': 0, 'y': 1, 'z': 2}
    
    # Initialize the index values
    index_values = torch.zeros_like(offsets[:, 0], dtype=torch.float32)
    
    # Compute the indices based on the specified order and the upscale factor
    for i, coord in enumerate(order):
        index_values += offsets[:, coord_map[coord]] * (upscale_factor ** (2 - i))
    
    # Compute the argsort of the computed indices to get the order
    order_indices = torch.argsort(index_values)
    
    return order_indices


class VoxelShuffle(nn.Module):
    """
    VoxelShuffle is designed to perform upscaling of 3D sparse convolutional tensors.
    This module reorders and reshapes the input tensor based on an upscale factor,
    which increases the resolution of the input features.

    Parameters:
        upscale_factor (int): The factor by which to increase the resolution of each dimension.
        pairing_style (str): Determines how channels are reshaped and paired.
                             Can be 'sequential' or 'strided'. 'Sequential' reshapes the channels one after the other,
                             while 'strided' rearranges the channels across all dimensions.
        layout_style (str): Specifies the order of dimensions in the output tensor, e.g., 'xyz', 'xzy', etc.

    Attributes:
        upscale_factor (int): Stored upscale factor for volume calculation.
        pairing_style (str): Stored pairing style for channel reshaping.
        layout_style (str): Stored layout style for dimensional ordering.
    """
    def __init__(self, upscale_factor, pairing_style='sequential', layout_style='xyz'):
        super(VoxelShuffle, self).__init__()
        self.upscale_factor = upscale_factor
        assert pairing_style == 'sequential' or pairing_style == 'strided', "Only support style: 'sequential' or 'strided'"
        self.pairing_style = pairing_style
        self.layout_style = layout_style

    def forward(self, x):
        # Check if the input is a SparseConvTensor
        assert isinstance(x, spconv.SparseConvTensor), "Input must be a SparseConvTensor"
        
        # Get the shape of the features
        features = x.features
        indices = x.indices
        spatial_shape = x.spatial_shape
        batch_size = x.batch_size
        
        # Check the number of channels
        
        num_points = indices.shape[0]
        num_channels = features.shape[1]
        assert num_channels % (self.upscale_factor ** 3) == 0, "Number of channels must be divisible by upscale_factor^3"
        output_channels = int(num_channels / (self.upscale_factor * self.upscale_factor * self.upscale_factor))
        
        # Calculate new shape and number of channels
        new_channels = num_channels // (self.upscale_factor ** 3)
        new_shape = (spatial_shape[0] * self.upscale_factor,
                     spatial_shape[1] * self.upscale_factor,
                     spatial_shape[2] * self.upscale_factor)
        
        volume = self.upscale_factor * self.upscale_factor * self.upscale_factor
        # Reshape features based on pairing style
        if self.pairing_style == 'sequential':
            features = features.view(-1, volume, output_channels).contiguous().view(num_points * volume, output_channels)
        elif self.pairing_style == 'strided':
            features = features.view(output_channels, -1, volume).permute(0, 2, 1).contiguous().view(num_points * volume, output_channels)

        # Create a grid of offsets for i, j, k
        i, j, k = torch.meshgrid(
            torch.arange(self.upscale_factor),
            torch.arange(self.upscale_factor),
            torch.arange(self.upscale_factor),
            indexing='ij'
        )

        num_points = indices.shape[0]
        # Stack the offsets to shape (8, 3)
        offsets = torch.stack((i, j, k), dim=-1).reshape(-1, 3)
        # Sort the offsets to achieve the correct order
        
        order = calculate_order(offsets, self.layout_style, self.upscale_factor)
        offsets = offsets[order]

        # Repeat and reshape the indices to match the number of offsets
        indices_repeated = indices.repeat_interleave(self.upscale_factor**3, dim=0)

        # Expand offsets to match the indices
        offsets_expanded = offsets.repeat(num_points, 1).to(indices_repeated.device)

        # Apply the offsets to the corresponding columns
        new_indices = indices_repeated.clone()
        new_indices[:, 1:] = indices_repeated[:, 1:] * self.upscale_factor + offsets_expanded

        index_map = {}
        for orig, new in zip(indices_repeated.cpu().numpy(), new_indices.cpu().numpy()):
            orig_tuple = tuple(orig)
            new_list = list(new)
            if orig_tuple not in index_map:
                index_map[orig_tuple] = []
            index_map[orig_tuple].append(new_list)

        # Create new SparseConvTensor
        new_x = spconv.SparseConvTensor(features, new_indices, new_shape, batch_size)
        
        return new_x, index_map

class VoxelUnshuffle(nn.Module):
    """
    VoxelUnshuffle reverses the upscaling process of VoxelShuffle by redistributing the features
    from upscaled positions back to the original tensor positions according to the `ind_dict` map.

    Parameters:
        scale (int): The factor by which the resolution of each dimension was originally increased.
        pairing_style (str): The pairing style used in the shuffle process, either 'sequential' or 'strided'.
    """
    def __init__(self, scale, pairing_style='sequential'):
        super(VoxelUnshuffle, self).__init__()
        self.scale = scale
        self.pairing_style = pairing_style

    def forward(self, x, ind_dict):
        """
        Reverse the shuffle operation to restore the original tensor configuration.

        :param x: The SparseConvTensor after shuffling.
        :param ind_dict: Dictionary mapping original indices to their corresponding shuffled positions.
        :return: A new SparseConvTensor with indices and features reshuffled to their original locations.
        """
        features = x.features
        batch_size = x.batch_size
        new_shape = [int(dim // self.scale) for dim in x.spatial_shape]
        volume = self.scale ** 3

        # Initialize the tensor to store the unshuffled features
        num_channels_per_point = features.shape[1] * volume
        original_features = torch.zeros((len(ind_dict), num_channels_per_point), device=features.device)

        # Prepare to reverse the shuffled indices
        original_indices = torch.tensor(list(ind_dict.keys()), device=features.device, dtype=torch.int)
        
        for original_idx, origin_indices in enumerate(ind_dict):
            shuffled_indices = ind_dict[origin_indices]
            for i, idx in enumerate(shuffled_indices):
                if self.pairing_style == 'sequential':
                    origin_feat = features[original_idx*volume+i]
                    start = i * features.shape[1]
                    end = (i + 1) * features.shape[1]
                    original_features[original_idx, start:end] = origin_feat
                elif self.pairing_style == 'strided':
                    origin_feat = features[original_idx*volume+i]
                    original_features[original_idx, i::volume] = origin_feat
        
        # Create new SparseConvTensor
        new_x = spconv.SparseConvTensor(original_features, original_indices, new_shape, batch_size)
        return new_x
