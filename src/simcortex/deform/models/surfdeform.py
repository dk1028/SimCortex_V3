from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F





class ConvGNAct(nn.Module):
    """
    Conv3D + GroupNorm + LeakyReLU (+ optional Dropout3D)
    Works better than BatchNorm with small batch sizes.
    """
    def __init__(self, cin, cout, k=3, s=1, groups=8, dropout=0.0):
        super().__init__()
        p = k // 2  # k=3 -> p=1
        self.conv = nn.Conv3d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False)

        g = min(groups, cout)
        while g > 1 and (cout % g) != 0:
            g -= 1
        self.gn = nn.GroupNorm(g, cout)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.do = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        x = self.do(x)
        return x


class ResBlock3D(nn.Module):
    """
    Lightweight residual block.
    """
    def __init__(self, c, k=3, groups=8, dropout=0.0):
        super().__init__()
        self.b1 = ConvGNAct(c, c, k=k, s=1, groups=groups, dropout=dropout)
        self.b2 = ConvGNAct(c, c, k=k, s=1, groups=groups, dropout=dropout)

    def forward(self, x):
        return x + self.b2(self.b1(x))


class GaussianFilter(nn.Module):

    def __init__(
        self,
        C: int = 3,
        sigma: float = 1.0,
        truncate: float = 2.0,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")

        self.C = int(C)
        self.sigma = float(sigma)
        self.truncate = float(truncate)
        self.padding_mode = str(padding_mode)

        radius = max(1, int(math.ceil(self.truncate * self.sigma)))
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")

        kernel = torch.exp(
            -(xx * xx + yy * yy + zz * zz) / (2.0 * self.sigma * self.sigma)
        )
        kernel = kernel / kernel.sum()
        kernel = kernel[None, None].repeat(self.C, 1, 1, 1, 1)

        self.register_buffer("kernel", kernel)
        self.radius = radius

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.radius,) * 6, mode=self.padding_mode)
        return F.conv3d(x, weight=self.kernel, padding=0, groups=self.C)

# -------------------------
# Single-Encoder MRI U-Net that outputs multi-scale SVFs
# -------------------------
class MUNetV2(nn.Module):
    """
    Single-branch MRI encoder-decoder.

    The encoder receives only the one-channel MRI volume. The decoder uses MRI
    skip features and predicts four full-resolution stationary velocity fields.
    """
    def __init__(
        self,
        C_in: int = 1,                    # one MRI input channel
        C_hid=(8, 16, 32, 64, 128, 128),
        K: int = 3,
        gn_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if int(C_in) != 1:
            raise ValueError(
                f"MRI-only MUNetV2 requires C_in=1, got {C_in}"
            )
        dropout = float(dropout)

        gn_groups = int(gn_groups)
        if gn_groups <= 0:
            raise ValueError(
                f"gn_groups must be > 0, got {gn_groups}"
            )

        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"dropout must satisfy 0 <= dropout < 1, got {dropout}"
            )

        # MRI channels
        Cm = [int(value) for value in C_hid]
        if len(Cm) != 6:
            raise ValueError(f"C_hid must contain exactly 6 channel values, got {len(Cm)}: {Cm}")
        if any(value <= 0 for value in Cm):
            raise ValueError(
                f"C_hid must contain positive channel values, got {Cm}"
            )
        # ---- MRI encoder (6 stages) ----
        self.m1 = nn.Sequential(ConvGNAct(1,   Cm[0], k=K, s=1, groups=gn_groups), ResBlock3D(Cm[0], k=K, groups=gn_groups))
        self.m2 = nn.Sequential(ConvGNAct(Cm[0], Cm[1], k=K, s=1, groups=gn_groups), ResBlock3D(Cm[1], k=K, groups=gn_groups))
        self.m3 = nn.Sequential(ConvGNAct(Cm[1], Cm[2], k=K, s=2, groups=gn_groups), ResBlock3D(Cm[2], k=K, groups=gn_groups))  # /2
        self.m4 = nn.Sequential(ConvGNAct(Cm[2], Cm[3], k=K, s=2, groups=gn_groups), ResBlock3D(Cm[3], k=K, groups=gn_groups, dropout=dropout))  # /4
        self.m5 = nn.Sequential(ConvGNAct(Cm[3], Cm[4], k=K, s=2, groups=gn_groups), ResBlock3D(Cm[4], k=K, groups=gn_groups, dropout=dropout))  # /8
        self.m6 = nn.Sequential(ConvGNAct(Cm[4], Cm[5], k=K, s=1, groups=gn_groups), ResBlock3D(Cm[5], k=K, groups=gn_groups, dropout=dropout))

        # ---- Decoder (uses MRI skips) ----
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.d5 = nn.Sequential(ConvGNAct(Cm[5] + Cm[4], Cm[4], k=K, s=1, groups=gn_groups), ResBlock3D(Cm[4], k=K, groups=gn_groups, dropout=dropout))
        self.d4 = nn.Sequential(ConvGNAct(Cm[4] + Cm[3], Cm[3], k=K, s=1, groups=gn_groups), ResBlock3D(Cm[3], k=K, groups=gn_groups, dropout=dropout))
        self.d3 = nn.Sequential(ConvGNAct(Cm[3] + Cm[2], Cm[2], k=K, s=1, groups=gn_groups), ResBlock3D(Cm[2], k=K, groups=gn_groups))
        self.d2 = nn.Sequential(ConvGNAct(Cm[2] + Cm[1], Cm[1], k=K, s=1, groups=gn_groups), ResBlock3D(Cm[1], k=K, groups=gn_groups))
        self.d1 = nn.Sequential(ConvGNAct(Cm[1] + Cm[0], Cm[0], k=K, s=1, groups=gn_groups), ResBlock3D(Cm[0], k=K, groups=gn_groups))

        # ---- Multi-scale flow heads ----
        self.flow1 = nn.Conv3d(Cm[3], 3, K, 1, padding=K // 2)
        self.flow2 = nn.Conv3d(Cm[2], 3, K, 1, padding=K // 2)
        self.flow3 = nn.Conv3d(Cm[1], 3, K, 1, padding=K // 2)
        self.flow4 = nn.Conv3d(Cm[0], 3, K, 1, padding=K // 2)

        for layer in [self.flow1, self.flow2, self.flow3, self.flow4]:
            nn.init.normal_(layer.weight, 0, 1e-5)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(
                f"MUNetV2 expects a 5D tensor (B,C,D,H,W), got {tuple(x.shape)}"
            )
        if x.shape[1] != 1:
            raise ValueError(
                f"Expected one MRI channel, got input shape {tuple(x.shape)}"
            )

        # ----- MRI encoder -----
        m1 = self.m1(x)        # full
        m2 = self.m2(m1)       # full
        m3 = self.m3(m2)       # /2
        m4 = self.m4(m3)       # /4
        m5 = self.m5(m4)       # /8
        m6 = self.m6(m5)       # /8

        # ----- Decoder with MRI skip connections -----
        x = torch.cat([m6, m5], dim=1)  # /8
        x = self.d5(x)
        x = self.up(x)                  # /4

        x = torch.cat([x, m4], dim=1)   # /4
        x = self.d4(x)
        svf1 = self.up(self.up(self.flow1(x)))  # /4 -> full

        x = self.up(x)                  # /2
        x = torch.cat([x, m3], dim=1)   # /2
        x = self.d3(x)
        svf2 = self.up(self.flow2(x))   # /2 -> full

        x = self.up(x)                  # full
        x = torch.cat([x, m2], dim=1)   # full
        x = self.d2(x)
        svf3 = self.flow3(x)            # full

        x = torch.cat([x, m1], dim=1)   # full
        x = self.d1(x)
        svf4 = self.flow4(x)            # full

        return svf1, svf2, svf3, svf4

# -------------------------
# SurfDeform (same deformation logic, single MRI encoder)
# -------------------------
class SurfDeform(nn.Module):
    def __init__(
        self,
        C_in=1,
        C_hid=(8, 16, 32, 64, 128, 128),
        inshape=(184, 224, 184),
        sigma=1.0,
        gn_groups=8,
        dropout=0.0,
    ):
        super().__init__()
        self.inshape = tuple(int(value) for value in inshape)

        if len(self.inshape) != 3:
            raise ValueError(f"inshape must be a 3-tuple/list (D,H,W), got {self.inshape}")

        if any(value <= 0 or value % 8 != 0 for value in self.inshape):
            raise ValueError(
                "SurfDeform inshape must contain positive dimensions divisible by 8, "
                f"got {self.inshape}"
            )

        self.munet = MUNetV2(
            C_in=C_in,
            C_hid=C_hid,
            gn_groups=gn_groups,
            dropout=dropout,
        )

        self.gaussian = GaussianFilter(
            C=3,
            sigma=sigma,
            truncate=2.0,
            padding_mode="replicate",
        )
        self._grid_cache = {}

    def forward(
        self,
        vert: torch.Tensor,
        vol: torch.Tensor,
        n_steps: int,
    ):
        if vert.ndim != 3 or int(vert.shape[-1]) != 3:
            raise ValueError(
                f"vert must have shape (B,V,3), got {tuple(vert.shape)}"
            )

        if vol.ndim != 5:
            raise ValueError(
                f"vol must have shape (B,C,D,H,W), got {tuple(vol.shape)}"
            )

        if int(vert.shape[0]) != int(vol.shape[0]):
            raise ValueError(
                f"vert/vol batch mismatch: {int(vert.shape[0])} vs {int(vol.shape[0])}"
            )

        if int(vol.shape[1]) != 1:
            raise ValueError(
                f"MRI-only SurfDeform expects one input channel, got {int(vol.shape[1])}"
            )

        n_steps = int(n_steps)
        if n_steps < 0:
            raise ValueError(
                f"n_steps must be >= 0, got {n_steps}"
            )

        if tuple(vol.shape[2:]) != self.inshape:
            raise ValueError(f"Input vol shape {tuple(vol.shape[2:])} != inshape {self.inshape}")

        svfs = self.munet(vol)

        for svf in svfs:
            svf = self.gaussian(svf)
            phi = self.integrate(svf, n_steps)

            deform = self.interpolate(
                vert[:, :, None, None],
                phi,
            )[..., 0, 0].permute(0, 2, 1)

            vert = vert + deform

        return vert


    def _get_base_grid(self, shape, ref: torch.Tensor) -> torch.Tensor:
        D, H, W = [int(v) for v in shape]
        key = (D, H, W, ref.device.type, ref.device.index, ref.dtype)

        grid = self._grid_cache.get(key, None)
        if grid is None:
            zz, yy, xx = torch.meshgrid(
                torch.arange(D, device=ref.device, dtype=ref.dtype),
                torch.arange(H, device=ref.device, dtype=ref.dtype),
                torch.arange(W, device=ref.device, dtype=ref.dtype),
                indexing="ij",
            )
            grid = torch.stack((zz, yy, xx), dim=0)[None]  # (1,3,D,H,W), IJK/DHW
            self._grid_cache[key] = grid

        return grid


    def transform(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        base = self._get_base_grid(flow.shape[2:], flow)
        coord = (base + flow).permute(0, 2, 3, 4, 1)  # (B,D,H,W,3), IJK
        return self.interpolate(coord, src)


    def integrate(self, svf: torch.Tensor, n_steps: int = 7) -> torch.Tensor:
        n_steps = int(n_steps)
        if n_steps < 0:
            raise ValueError(
                f"n_steps must be >= 0, got {n_steps}"
            )

        if svf.ndim != 5 or int(svf.shape[1]) != 3:
            raise ValueError(
                f"svf must have shape (B,3,D,H,W), got {tuple(svf.shape)}"
            )

        flow = svf / float(2 ** n_steps)
        for _ in range(n_steps):
            flow = flow + self.transform(flow, flow)
        return flow

    def interpolate(self, coord_ijk: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """
        coord_ijk: (..., 3) in IJK / DHW voxel coordinates of src
        src: (B, C, D, H, W)
        """
        D, H, W = src.shape[2:]

        d = 2.0 * coord_ijk[..., 0] / max(D - 1, 1) - 1.0
        h = 2.0 * coord_ijk[..., 1] / max(H - 1, 1) - 1.0
        w = 2.0 * coord_ijk[..., 2] / max(W - 1, 1) - 1.0

        # grid_sample expects XYZ = WHD for 5D input
        grid_xyz = torch.stack((w, h, d), dim=-1)

        return F.grid_sample(
            src,
            grid_xyz,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
