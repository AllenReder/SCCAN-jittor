import jittor as jt
from jittor import nn

class ASPP(nn.Module):
    def __init__(self, out_channels=256):
        super(ASPP, self).__init__()
        self.layer6_0 = nn.Sequential(
            nn.Conv(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
        )
        self.layer6_1 = nn.Sequential(
            nn.Conv(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
        )
        self.layer6_2 = nn.Sequential(
            nn.Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=True),
            nn.ReLU(),
        )
        self.layer6_3 = nn.Sequential(
            nn.Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=True),
            nn.ReLU(),
        )
        self.layer6_4 = nn.Sequential(
            nn.Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18, bias=True),
            nn.ReLU(),
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv):
                jt.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm):
                m.weight.data = jt.init.constant_(m.weight.shape, 1.0)
                m.bias.data = jt.init.constant_(m.bias.shape, 0.0)
                
    def execute(self, x):
        feature_size = x.shape[-2:]
        global_feature = jt.mean(x, dims=(2, 3), keepdims=True)

        global_feature = self.layer6_0(global_feature)

        global_feature = global_feature.repeat([1, 1, feature_size[0], feature_size[1]])
        out = jt.concat(
            [global_feature, self.layer6_1(x), self.layer6_2(x), self.layer6_3(x), self.layer6_4(x)], dim=1)
        return out