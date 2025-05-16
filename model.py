from typing import Optional, Tuple

from diffusers import UNet1DModel


class UNetModelWrapper(UNet1DModel):
    def __init__(
        self,
        sample_size: int = 1024,
        sample_rate: Optional[int] = None,
        in_channels: int = 1,
        out_channels: int = 1,
        extra_in_channels: int = 0,
        time_embedding_type: str = "positional",
        flip_sin_to_cos: bool = True,
        use_timestep_embedding: bool = True,
        freq_shift: float = 0.0,
        down_block_types: Tuple[str] = (
            "DownBlock1D",
            "DownBlock1D",
            "AttnDownBlock1D",
            "AttnDownBlock1D",
        ),
        up_block_types: Tuple[str] = (
            "AttnUpBlock1D",
            "AttnUpBlock1D",
            "UpBlock1D",
            "UpBlock1D",
        ),
        mid_block_type: Tuple[str] = "UNetMidBlock1D",
        out_block_type: str = None,
        block_out_channels: Tuple[int] = (64, 128, 256, 256),
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        layers_per_block: int = 2,
        downsample_each_block: bool = False,
    ):
        return super().__init__(
            sample_size=sample_size,
            sample_rate=sample_rate,
            in_channels=in_channels,
            out_channels=out_channels,
            extra_in_channels=extra_in_channels,
            time_embedding_type=time_embedding_type,
            flip_sin_to_cos=flip_sin_to_cos,
            use_timestep_embedding=use_timestep_embedding,
            freq_shift=freq_shift,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            mid_block_type=mid_block_type,
            out_block_type=out_block_type,
            block_out_channels=block_out_channels,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            layers_per_block=layers_per_block,
            downsample_each_block=downsample_each_block,
        )

    def forward(self, t, x, y=None, *args, **kwargs):
        x = x.reshape(x.shape[0], 1, x.shape[-1])
        v = super().forward(sample=x, timestep=t, return_dict=False)[0]
        return v.reshape(x.shape[0], x.shape[-1])
