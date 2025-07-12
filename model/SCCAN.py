import jittor as jt
from jittor import nn
import jittor.nn as F

from model.ASPP import ASPP
from model.backbone_utils import Backbone
from model.swin_sccan import SwinTransformer
from model.loss import WeightedDiceLoss


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = mask.mean(dims=(2,3), keepdims=True) * feat_h * feat_w + 0.0005
    supp_feat_mean = (supp_feat.sum(dims=(2,3), keepdims=True) / (area - 0.0005))
    return supp_feat_mean


class OneModel(nn.Module):
    def __init__(self, args):
        super(OneModel, self).__init__()
        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.vgg = args.vgg
        self.dataset = args.data_set
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.criterion_dice = WeightedDiceLoss()
        self.print_freq = args.print_freq / 2
        self.pretrained = True
        self.classes = 2

        assert self.layers in [50, 101, 152]
        self.backbone = Backbone('resnet{}'.format(self.layers), train_backbone=False,
                                 return_interm_layers=True, dilation=[False, True, True])

        # Meta Learner
        reduce_dim = 256
        self.low_fea_id = args.low_fea[-1]
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )

        embed_dim = reduce_dim
        self.init_merge_query = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + 1, embed_dim, kernel_size=1, padding=0, bias=False)
        )
        self.init_merge_supp = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + 1, embed_dim, kernel_size=1, padding=0, bias=False)
        )

        # swin transformer
        depths = (8,)
        num_heads = (8,)
        window_size = 8
        mlp_ratio = 1.
        self.window_size = window_size
        pretrain_img_size = 64
        self.transformer = SwinTransformer(pretrain_img_size=pretrain_img_size, embed_dim=embed_dim,
                                           depths=depths, num_heads=num_heads, window_size=window_size,
                                           mlp_ratio=mlp_ratio, out_indices=tuple(range(len(depths))))

        scale = 0
        for i in range(len(depths)):
            scale += 2 ** i
        self.ASPP_meta = ASPP(scale * embed_dim)
        self.res1_meta = nn.Sequential(
            nn.Conv2d(scale * embed_dim * 5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.res2_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        )
        self.cls_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
        )
        self.relu = nn.ReLU()

    def get_optim(self, model, args, LR):
        optimizer = jt.optim.SGD(
            [
                {'params': model.down_query.parameters()},
                {'params': model.down_supp.parameters()},
                {'params': model.init_merge_query.parameters()},
                {'params': model.init_merge_supp.parameters()},
                {'params': model.ASPP_meta.parameters()},
                {'params': model.res1_meta.parameters()},
                {'params': model.res2_meta.parameters()},
                {'params': model.cls_meta.parameters()},
            ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer_swin = jt.optim.AdamW(
            [
                {"params": [p for n, p in model.named_parameters() if "transformer" in n and p.requires_grad]}
            ], lr=6e-5
        )
        return optimizer, optimizer_swin

    def freeze_modules(self, model):
        for param in model.backbone.parameters():
            param.requires_grad = False

    def generate_prior(self, query_feat_high, final_supp_list, mask_list, fts_size):
        bsize, ch_sz, sp_sz, _ = query_feat_high.size()
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size()[2]
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat = tmp_supp_feat
            q = query_feat_high.flatten(2).transpose(-2, -1)
            s = tmp_supp_feat.flatten(2).transpose(-2, -1)

            tmp_query = q
            tmp_query = tmp_query.permute(0, 2, 1)  # [bs, c, h*w]
            tmp_query_norm = jt.norm(tmp_query, dim=1, keepdims=True)

            tmp_supp = s
            tmp_supp_norm = jt.norm(tmp_supp, dim=2, keepdims=True)

            similarity = jt.bmm(tmp_supp, tmp_query) / (jt.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
            similarity = similarity.permute(0, 2, 1)
            similarity = F.softmax(similarity, dim=-1)
            similarity = jt.bmm(similarity, tmp_mask.flatten(2).transpose(-2, -1)).squeeze(-1)
            similarity = (similarity - similarity.min(dim=1, keepdims=True)[0]) / (
                    similarity.max(dim=1, keepdims=True)[0] - similarity.min(dim=1, keepdims=True)[0] + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=fts_size, mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)
        corr_query_mask = jt.concat(corr_query_mask_list, dim=1)
        corr_query_mask = (corr_query_mask).mean(dim=1, keepdims=True)
        return corr_query_mask

    # que_img, sup_img, sup_mask, que_mask(meta), cat_idx(meta)
    def execute(self, x, s_x, s_y, y_m, cat_idx=None):
        x_size = x.size()
        bs = x_size[0]
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)  # (473 - 1) / 8 * 8 + 1 = 60
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)  # 60

        # Interpolation size for pascal/coco
        size = (64, 64)

        # ========================================
        # Feature Extraction - Query/Support
        # ========================================
        # Query/Support Feature
        with jt.no_grad():
            qry_bcb_fts = self.backbone(x)
            supp_bcb_fts = self.backbone(s_x.view(-1, 3, x_size[2], x_size[3]))

        query_feat_high = qry_bcb_fts['3']

        query_feat = jt.concat([qry_bcb_fts['1'], qry_bcb_fts['2']], dim=1)
        query_feat = self.down_query(query_feat)
        query_feat = F.interpolate(query_feat, size=size, mode='bilinear', align_corners=True)
        fts_size = query_feat.size()[-2:]
        supp_feat = jt.concat([supp_bcb_fts['1'], supp_bcb_fts['2']], dim=1)
        supp_feat = self.down_supp(supp_feat)
        supp_feat = F.interpolate(supp_feat, size=size, mode='bilinear', align_corners=True)

        mask_list = []
        supp_pro_list = []
        supp_feat_list = []
        final_supp_list = []
        supp_feat_mid = supp_feat.view(bs, self.shot, -1, fts_size[0], fts_size[1])
        supp_bcb_fts['3'] = F.interpolate(supp_bcb_fts['3'], size=size, mode='bilinear', align_corners=True)
        supp_feat_high = supp_bcb_fts['3'].view(bs, self.shot, -1, fts_size[0], fts_size[1])
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask = F.interpolate(mask, size=fts_size, mode='bilinear', align_corners=True)
            mask_list.append(mask)
            final_supp_list.append(supp_feat_high[:, i, :, :, :])
            supp_feat_list.append((supp_feat_mid[:, i, :, :, :] * mask).unsqueeze(-1))
            supp_pro = Weighted_GAP(supp_feat_mid[:, i, :, :, :], mask)
            supp_pro_list.append(supp_pro)

        # Support features/prototypes/masks
        supp_mask = jt.concat(mask_list, dim=1).mean(dim=1, keepdims=True)  # bs, 1, 60, 60
        supp_feat = jt.concat(supp_feat_list, dim=-1).mean(-1)  # bs, 256, 60, 60
        supp_pro = jt.concat(supp_pro_list, dim=2).mean(dim=2, keepdims=True)  # bs, 256, 1, 1
        supp_pro = supp_pro.expand(query_feat.shape)  # bs, 256, 60, 60

        # Prior Similarity Mask
        corr_query_mask = self.generate_prior(query_feat_high, final_supp_list, mask_list, fts_size)

        # ========================================
        # Cross Swin Transformer
        # ========================================
        # Adapt query/support features with support prototype
        query_cat = jt.concat([query_feat, supp_pro, corr_query_mask], dim=1)  # bs, 512, 60, 60
        query_feat = self.init_merge_query(query_cat)  # bs, 256, 60, 60
        supp_cat = jt.concat([supp_feat, supp_pro, supp_mask], dim=1)  # bs, 512, 60, 60
        supp_feat = self.init_merge_supp(supp_cat)  # bs, 256, 60, 60

        # Swin transformer (cross)
        query_feat_list = []
        query_feat_list.extend(self.transformer(query_feat, supp_feat, supp_mask))
        fused_query_feat = []
        for idx, qry_feat in enumerate(query_feat_list):
            fused_query_feat.append(
                self.relu(
                    F.interpolate(qry_feat, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)
                )
            )
        merge_feat = jt.concat(fused_query_feat, dim=1)

        # ========================================
        # Meta Output
        # ========================================
        query_meta = self.ASPP_meta(merge_feat)
        query_meta = self.res1_meta(query_meta)
        query_meta = self.res2_meta(query_meta) + query_meta
        meta_out = self.cls_meta(query_meta)

        # Interpolate
        if self.zoom_factor != 1:
            meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)

        # ========================================
        # Loss
        # ========================================
        if self.training:
            main_loss = self.criterion_dice(meta_out, y_m.long())
            aux_loss1 = jt.zeros_like(main_loss)
            aux_loss2 = jt.zeros_like(main_loss)
            return meta_out.argmax(dim=1)[0], main_loss, aux_loss1, aux_loss2
        else:
            return meta_out


if __name__ == "__main__":
    import argparse
    from util import config

    def get_parser():
        parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation')
        parser.add_argument('--arch', type=str, default='SCCAN')
        parser.add_argument('--viz', action='store_true', default=False)
        parser.add_argument('--config', type=str, default='config/pascal/pascal_split0_resnet50.yaml',
                            help='config file')  # coco/coco_split0_resnet50.yaml
        parser.add_argument('--opts', help='see config/pascal/pascal_split0_resnet50.yaml for all options', default=None,
                            nargs=argparse.REMAINDER)
        args = parser.parse_args()
        assert args.config is not None
        cfg = config.load_cfg_from_cfg_file(args.config)
        cfg = config.merge_cfg_from_args(cfg, args)
        if args.opts is not None:
            cfg = config.merge_cfg_from_list(cfg, args.opts)
        return cfg
    
    args = get_parser()
    model = OneModel(args)
    model.train()

    # 输出每个参数是否是训练状态
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")
    exit()

    # s_x: torch.Size([1, 1, 3, 473, 473]), s_y: torch.Size([1, 1, 473, 473]), x: torch.Size([1, 3, 473, 473]), y_m: torch.Size([1, 473, 473]), cat_idx: [tensor([2])]
    s_x = jt.randn(1, 1, 3, 473, 473)
    s_y = jt.randn(1, 1, 473, 473)
    x = jt.randn(1, 3, 473, 473)
    y_m = jt.randn(1, 473, 473)
    output = model(x, s_x, s_y, y_m)
    print(output[0])
    print(output[1])
    print(output[2])
    print(output[3])


    # optimizer
    # optimizer, optimizer_swin = model.get_optim(model, args, LR=args.base_lr)
    # print("优化器管理的参数名:")
    # for idx, param_group in enumerate(optimizer.param_groups):
    #     print(f"参数组 {idx}:")
    #     params_set = set(id(p) for p in param_group['params'])
    #     for name, param in model.named_parameters():
    #         if id(param) in params_set:
    #             print(f"  {name}")

    # # 对于optimizer_swin
    # print("\nSwin优化器管理的参数名:")
    # for idx, param_group in enumerate(optimizer_swin.param_groups):
    #     print(f"参数组 {idx}:")
    #     params_set = set(id(p) for p in param_group['params'])
    #     for name, param in model.named_parameters():
    #         if id(param) in params_set:
    #             print(f"  {name}")