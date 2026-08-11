"""Model definitions for super-resolution experiments."""

from .ecbsr import ECBSR, fuse_ecbsr
from .fsrcnn import FSRCNN
from .mobile_srnet import MobileSRNet, fuse_mobile_srnet
from .sepres_v2 import PlainBodyBlock, SepResV2, fuse_sepres_v2

__all__ = [
    "FSRCNN",
    "MobileSRNet",
    "ECBSR",
    "SepResV2",
    "PlainBodyBlock",
    "fuse_ecbsr",
    "fuse_mobile_srnet",
    "fuse_sepres_v2",
]

# Optional local-only Dual/Plain probes (gitignored on public remote).
try:
    from .dual_plain_sr import DualStreamSR, PlainSR, fuse_dual_stream_sr

    __all__ += ["DualStreamSR", "PlainSR", "fuse_dual_stream_sr"]
except ImportError:  # pragma: no cover
    pass
