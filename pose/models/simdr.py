import torch
from torch import Tensor, nn

from .backbones import HRNet


class SimDR(nn.Module):
    def __init__(
        self,
        backbone: str = "w32",
        num_joints: int = 17,
        image_size: tuple = (256, 192),
    ):
        super().__init__()
        self.backbone = HRNet(backbone)
        self.final_layer = nn.Conv2d(self.backbone.all_channels[0], num_joints, 1)
        self.mlp_head_x = nn.Linear(3072, int(image_size[1] * 2.0))
        self.mlp_head_y = nn.Linear(3072, int(image_size[0] * 2.0))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(
                torch.load(pretrained, map_location="cpu"), strict=False
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.backbone(x)
        out = self.final_layer(out).flatten(2)
        pred_x = self.mlp_head_x(out)
        pred_y = self.mlp_head_y(out)
        return pred_x, pred_y


if __name__ == "__main__":
    model = SimDR("w32")
    model.load_state_dict(
        torch.load(
            "checkpoints/pretrained/simdr_hrnet_w32_256x192.pth", map_location="cpu"
        )
    )
    x = torch.randn(4, 3, 256, 192)
    px, py = model(x)
    print(px.shape, py.shape)
