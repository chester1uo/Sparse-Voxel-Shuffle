import torch
import spconv
from torch import nn
import spconv.pytorch as spconv


class VoxelShuffle(nn.Module):
    def __init__(self, upscale_factor, pairing_style='sequential'):
        super(VoxelShuffle, self).__init__()
        self.upscale_factor = upscale_factor
        assert pairing_style == 'sequential' or pairing_style == 'strided', "Only support style: 'sequential' or 'strided'"
        self.pairing_style = pairing_style

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
            torch.arange(upscale_factor),
            torch.arange(upscale_factor),
            torch.arange(upscale_factor),
            indexing='ij'
        )

        num_points = indices.shape[0]
        # Stack the offsets to shape (8, 3)
        offsets = torch.stack((i, j, k), dim=-1).reshape(-1, 3)
        # Sort the offsets to achieve the correct order
        order = torch.argsort(offsets[:, 0] * (upscale_factor**2) + offsets[:, 2] * upscale_factor + offsets[:, 1])
        offsets = offsets[order]

        # Repeat and reshape the indices to match the number of offsets
        indices_repeated = indices.repeat_interleave(upscale_factor**3, dim=0)

        # Expand offsets to match the indices
        offsets_expanded = offsets.repeat(num_points, 1).to(indices_repeated.device)

        # Apply the offsets to the corresponding columns
        new_indices = indices_repeated.clone()
        new_indices[:, 1:] = indices_repeated[:, 1:] * upscale_factor + offsets_expanded

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