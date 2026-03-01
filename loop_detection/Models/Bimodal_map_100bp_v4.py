from Models.uformer_utils import *


# Downsample Block
class Downsample1D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=4, stride=2, padding=1):
        super(Downsample1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.conv(x)
        out = out.transpose(1, 2)
        return out


class Upsample1D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=4, stride=2, padding=1, output_padding=0):
        super(Upsample1D, self).__init__()
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.conv_transpose(x)
        out = out.transpose(1, 2)
        return out

class Pooling1DEncoder(nn.Module):
    def __init__(self, embed_dim=64, dropout=0.0):
        super(Pooling1DEncoder, self).__init__()
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.pool2 = nn.MaxPool1d(5)
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm1d(embed_dim)
        self.pool3 = nn.MaxPool1d(5)
        self.dropout3 = nn.Dropout(dropout)

        self.conv4 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm1d(embed_dim)

    def forward(self, seq):
        # (batch, embed_dim, 5000)
        x = rearrange(seq, 'b l c -> b c l')

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = F.relu(self.bn4(self.conv4(x)))

        down = rearrange(x, 'b c l -> b l c')

        return down


class Downsampling1DTransformer(nn.Module):
    def __init__(self, embed_dim=64, dropout=0.0, num_heads=4, depth=2):
        super(Downsampling1DTransformer, self).__init__()

        self.pooling_encoder = Pooling1DEncoder(embed_dim=embed_dim, dropout=dropout)
        forward_layer_0 = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation='relu'
        )
        self.seq_encoder_0 = nn.TransformerEncoder(forward_layer_0, num_layers=depth)
        self.dowsample_0 = Downsample1D(embed_dim, embed_dim * 2)

        forward_layer_1 = nn.TransformerEncoderLayer(
            d_model=embed_dim * 2,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu'
        )
        self.seq_encoder_1 = nn.TransformerEncoder(forward_layer_1, num_layers=depth)
        self.dowsample_1 = Downsample1D(embed_dim * 2, embed_dim * 4)

        forward_layer_2 = nn.TransformerEncoderLayer(
            d_model=embed_dim * 4,
            nhead=num_heads,
            dim_feedforward=embed_dim * 8,
            dropout=dropout,
            activation='relu'
        )
        self.seq_encoder_2 = nn.TransformerEncoder(forward_layer_2, num_layers=depth)
        self.dowsample_2 = Downsample1D(embed_dim * 4, embed_dim * 8)

        forward_layer_bottleneck = nn.TransformerEncoderLayer(
            d_model=embed_dim * 8,
            nhead=num_heads,
            dim_feedforward=embed_dim * 16,
            dropout=dropout,
            activation='relu'
        )
        self.bottleneck = nn.TransformerEncoder(forward_layer_bottleneck, num_layers=depth)


    def forward(self, seq):
        pool_seq = self.pooling_encoder(seq)
        encode0 = self.seq_encoder_0(pool_seq)
        down0 = self.dowsample_0(encode0)

        encode1 = self.seq_encoder_1(down0)
        down1 = self.dowsample_1(encode1)

        encode2 = self.seq_encoder_2(down1)
        down2 = self.dowsample_2(encode2)

        bottle = self.bottleneck(down2)

        return bottle, (encode2, encode1, encode0)


class Upsampling1DTransformer(nn.Module):
    def __init__(self, embed_dim=64, dropout=0.0, num_heads=4, depth=2):
        super(Upsampling1DTransformer, self).__init__()
        self.upsample_3 = Upsample1D(embed_dim * 8, embed_dim * 4, output_padding=1)
        backward_layer_3 = nn.TransformerEncoderLayer(
            d_model=embed_dim * 8,
            nhead=num_heads,
            dim_feedforward=embed_dim * 16,
            dropout=dropout,
            activation='relu'
        )
        self.seq_decoder_3 = nn.TransformerEncoder(backward_layer_3, num_layers=depth)

        self.upsample_4 = Upsample1D(embed_dim * 8, embed_dim * 2)
        backward_layer_4 = nn.TransformerEncoderLayer(
            d_model=embed_dim * 4,
            nhead=num_heads,
            dim_feedforward=embed_dim * 8,
            dropout=dropout,
            activation='relu'
        )
        self.seq_decoder_4 = nn.TransformerEncoder(backward_layer_4, num_layers=depth)

        self.upsample_5 = Upsample1D(embed_dim * 4, embed_dim)
        backward_layer_5 = nn.TransformerEncoderLayer(
            d_model=embed_dim * 2,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu'
        )
        self.seq_decoder_5 = nn.TransformerEncoder(backward_layer_5, num_layers=depth)

    def forward(self, fusion_embed, encode0, encode1, encode2):
        up3 = self.upsample_3(fusion_embed)
        decode3 = self.seq_decoder_3(torch.cat([up3, encode2], dim=-1))

        up4 = self.upsample_4(decode3)
        decode4 = self.seq_decoder_4(torch.cat([up4, encode1], dim=-1))

        up5 = self.upsample_5(decode4)
        output = self.seq_decoder_5(torch.cat([up5, encode0], dim=-1))

        return output


class ImageToPatches(nn.Module):
    def __init__(self, patch_size):
        super(ImageToPatches, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        # x is in shape (B, C, H, W)
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, C * self.patch_size * self.patch_size)
        return x


class PatchesToImage(nn.Module):
    def __init__(self, patch_size, output_size=(1, 50 , 50)):
        super(PatchesToImage, self).__init__()
        self.patch_size = patch_size
        self.output_size = output_size  # (C, H, W)
        self.C, self.H, self.W = output_size

    def forward(self, x):
        # x is in shape (B, 100, 25)
        B, num_patches, patch_vector_size = x.shape
        assert num_patches == (self.H // self.patch_size) * (self.W // self.patch_size), "Number of patches does not match."
        assert patch_vector_size == self.C * self.patch_size * self.patch_size, "Patch vector size does not match."

        # Reshape back to (B, L, C, ph, pw)
        x = x.view(B, num_patches, self.C, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # We need to reconstruct the original dimensions from patches
        # (B, C, num_patches, patch_height, patch_width)
        # Calculate number of patches along height and width
        num_patches_height = self.H // self.patch_size
        num_patches_width = self.W // self.patch_size

        # Fold patches back to image
        x = x.view(B, self.C, num_patches_height, num_patches_width, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, self.C, self.H, self.W)
        return x

class Downsampling2DTransformer(nn.Module):
    def __init__(self, num_heads=4, patch_size=2, embed_dim=64, in_dim=2, dropout=0.0,
                 depths=2, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, token_mlp='leff',
                 token_projection='linear', dowsample=Downsample, mlp_ratio=2,use_repo=True):
        super(Downsampling2DTransformer, self).__init__()
        self.encoderlayer_0 = BasicTransformerLayer(dim=embed_dim,
                                                    depth=depths,
                                                    num_heads=num_heads,
                                                    mlp_ratio=mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=dropout, attn_drop=dropout,
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=False,
                                                    token_projection=token_projection, token_mlp=token_mlp, use_rope=use_repo)
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2)
        self.encoderlayer_1 = BasicTransformerLayer(dim=embed_dim * 2,
                                                    depth=depths,
                                                    num_heads=num_heads,
                                                    mlp_ratio=mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=dropout, attn_drop=dropout,
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=False,
                                                    token_projection=token_projection, token_mlp=token_mlp, use_rope=use_repo)
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4)

        self.encoderlayer_2 = BasicTransformerLayer(dim=embed_dim * 4,
                                                    depth=depths,
                                                    num_heads=num_heads,
                                                    mlp_ratio=mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=dropout, attn_drop=dropout,
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=False,
                                                    token_projection=token_projection, token_mlp=token_mlp, use_rope=use_repo)
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8)

        self.encoderlayer_3 = BasicTransformerLayer(dim=embed_dim * 8,
                                                    depth=depths,
                                                    num_heads=num_heads,
                                                    mlp_ratio=mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=dropout, attn_drop=dropout,
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=False,
                                                    token_projection=token_projection, token_mlp=token_mlp, use_rope=use_repo)

    def forward(self, x, mask=None):
        conv0 = self.encoderlayer_0(x, mask=mask)
        pool0 = self.dowsample_0(conv0)

        conv1 = self.encoderlayer_1(pool0, mask=mask)
        pool1 = self.dowsample_1(conv1)

        conv2 = self.encoderlayer_2(pool1)
        pool2 = self.dowsample_2(conv2)

        map_embed = self.encoderlayer_3(pool2)
        return map_embed


class UformerGraphFuse(nn.Module):
    def __init__(self, img_size=50, num_heads=4, patch_size=2, embed_dim=64, in_dim=2, dropout=0.0,
                 depths=2, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, token_mlp='leff',
                 token_projection='linear', mlp_ratio=2, mode='finetune', loss_mode='translation', modality='infer_map', use_repo=False):
        super().__init__()
        """
            loss_mode: contras, orth, gauss, all, geometric, translation
            mode: pretrain, finetune
            modality: epi, infer_map
        """
        self.modality = modality
        self.mode = mode
        self.loss_mode = loss_mode
        print(f'num depths: {depths}, embed_dim: {embed_dim}, mode: {mode}, loss_mode: {loss_mode}, modality: {modality}')

        self.seq_input_projection = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.seq_downsampleTrans = Downsampling1DTransformer(embed_dim=embed_dim, dropout=dropout, num_heads=num_heads)

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.to_patches = ImageToPatches(patch_size)
        self.patch_embedding = nn.Linear(patch_size * patch_size * 1, embed_dim)
        self.map_downsampleTrans = Downsampling2DTransformer(num_heads=num_heads, patch_size=patch_size,
                                                             embed_dim=embed_dim, in_dim=in_dim, dropout=dropout,
                                                             depths=depths, token_mlp=token_mlp,
                                                             mlp_ratio=mlp_ratio, use_repo=use_repo)

        self.seq_upsampleTrans = Upsampling1DTransformer(embed_dim=embed_dim, dropout=dropout, num_heads=num_heads)


        if self.loss_mode == 'contras' or self.loss_mode == 'geometric':
            classifier_dim = embed_dim * 2
        else:
            classifier_dim = embed_dim * 2

        if self.mode != 'pretrain':
            self.classifier = nn.Sequential(
                nn.Linear(classifier_dim, 32),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

            self.fusion_block = BasicTransformerLayer(dim=embed_dim * 8,
                                                      depth=depths,
                                                      num_heads=num_heads,
                                                      mlp_ratio=mlp_ratio,
                                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                      drop=dropout, attn_drop=dropout,
                                                      norm_layer=norm_layer,
                                                      use_checkpoint=False,
                                                      token_projection=token_projection, token_mlp='mlp',
                                                      use_rope=use_repo)

        self.apply(self._init_weights)

        self.temp = nn.Parameter(0.07*torch.ones([]))
        self.epi_share_proj = nn.Linear(embed_dim * 8, embed_dim * 4)
        self.map_share_proj = nn.Linear(embed_dim * 8, embed_dim * 4)
        self.epi_independent_proj = nn.Linear(embed_dim * 8, embed_dim * 4)
        self.map_independent_proj = nn.Linear(embed_dim * 8, embed_dim * 4)
        self.mlp_e2m = nn.Linear(embed_dim * 8, embed_dim * 8)
        self.mlp_m2e = nn.Linear(embed_dim * 8, embed_dim * 8)

        self.adaptive_pool_epi = nn.AdaptiveAvgPool1d(9)
        self.adaptive_pool_map = nn.AdaptiveAvgPool1d(12)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def contrastive_loss(self, epi, map):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        # epi_mean_embed = torch.mean(epi, dim=1)
        # map_mean_embed = torch.mean(map, dim=1)
        sim_o2r = torch.matmul(epi, map.permute(1, 0)) / self.temp
        sim_r2d = torch.matmul(map, epi.permute(1, 0)) / self.temp

        with torch.no_grad():
            sim_targets = torch.zeros(sim_o2r.size()).to(epi.device)
            sim_targets.fill_diagonal_(1)

        loss_o2r = -torch.sum(F.log_softmax(sim_o2r, dim=1) * sim_targets, dim=1).mean()
        loss_r2o = -torch.sum(F.log_softmax(sim_r2d, dim=1) * sim_targets, dim=1).mean()
        loss_contras = (loss_o2r + loss_r2o) / 2
        return loss_contras

    def orthogonal_loss(self, original_v, independent_v, original_t, independent_t):
        original_v = original_v.unsqueeze(1)  # (batch, 1, emb_dim)
        independent_v = independent_v.unsqueeze(2)  # (batch, emb_dim, 1)
        original_t = original_t.unsqueeze(1)  # (batch, 1, emb_dim)
        independent_t = independent_t.unsqueeze(2)  # (batch, emb_dim, 1)

        # Compute dot products using batch matrix multiplication
        dot_product_v = torch.bmm(original_v, independent_v)  # (batch, 1, 1)
        dot_product_t = torch.bmm(original_t, independent_t)  # (batch, 1, 1)

        # Flatten the results to (batch,) and square each element
        dot_product_v = dot_product_v.squeeze() ** 2  # (batch,)
        dot_product_t = dot_product_t.squeeze() ** 2  # (batch,)

        # Compute the mean of squared dot products for both modalities
        total_loss = (dot_product_v + dot_product_t).mean()
        return total_loss

    def gaussian_loss(self, features_v, features_t, t=2):
        # 计算特征之间的差的平方
        diffs_v = features_v.unsqueeze(1) - features_v.unsqueeze(0)  # (N, N, D)
        diffs_t = features_t.unsqueeze(1) - features_t.unsqueeze(0)  # (N, N, D)

        # 计算高斯势能
        norms_v = torch.sum(diffs_v ** 2, dim=2)  # (N, N)
        norms_t = torch.sum(diffs_t ** 2, dim=2)  # (N, N)
        gaussian_potential_v = torch.exp(-t * norms_v)
        gaussian_potential_t = torch.exp(-t * norms_t)

        # 计算损失
        loss = -torch.log(torch.mean(gaussian_potential_v + gaussian_potential_t))
        return loss

    def geometric_loss(self, features_v, features_t):
        # 计算跨模态特征对的内积
        dot_vt = torch.matmul(features_v, features_t.t())  # (N, N)
        dot_tv = dot_vt.t()  # (N, N)

        # 计算同模态特征对的内积
        dot_vv = torch.matmul(features_v, features_v.t())  # (N, N)
        dot_tt = torch.matmul(features_t, features_t.t())  # (N, N)

        # 计算几何一致性损失
        consistency_vt = (dot_vt - dot_tv) ** 2  # 跨模态一致性
        consistency_vv_tt = (dot_vv - dot_tt) ** 2  # 同模态一致性

        # 损失是所有差异的平均值
        loss = torch.mean(consistency_vt + consistency_vv_tt)
        return loss

    def translation_loss(self, epi_share_feat, epi_independent_feat, map_share_feat, map_independent_feat):
        epi_concat = torch.cat([epi_share_feat, epi_independent_feat], dim=-1)
        map_concat = torch.cat([map_share_feat, map_independent_feat], dim=-1)
        epi_emd_map = self.mlp_e2m(epi_concat)
        map_emd_epi = self.mlp_m2e(map_concat)
        epi_emd_map = epi_emd_map / torch.norm(epi_emd_map, dim=1, keepdim=True)
        map_emd_epi = map_emd_epi / torch.norm(map_emd_epi, dim=1, keepdim=True)
        e2m_loss = torch.mean(torch.norm(epi_emd_map - map_concat / torch.norm(map_concat, dim=1, keepdim=True), dim=1))
        m2e_loss = torch.mean(torch.norm(map_emd_epi - epi_concat / torch.norm(epi_concat, dim=1, keepdim=True), dim=1))
        loss = (e2m_loss + m2e_loss) / 2
        return loss

    def translation_loss_seq(self, epi_concat, map_concat):
        epi_concat_tmp = epi_concat.permute(0, 2, 1)  # 调整维度以适应池化层(batch, features, seq_len)
        map_concat_tmp = map_concat.permute(0, 2, 1)  # 同上

        epi_concat_tmp = self.adaptive_pool_epi(epi_concat_tmp)  # 调整epi的长度
        map_concat_tmp = self.adaptive_pool_map(map_concat_tmp)  # 调整map的长度

        epi_concat_tmp = epi_concat_tmp.permute(0, 2, 1)  # 恢复原始维度
        map_concat_tmp = map_concat_tmp.permute(0, 2, 1)

        epi_emd_map = self.mlp_e2m(epi_concat_tmp)
        map_emd_epi = self.mlp_m2e(map_concat_tmp)

        epi_emd_map = epi_emd_map / torch.norm(epi_emd_map, dim=2, keepdim=True)
        map_emd_epi = map_emd_epi / torch.norm(map_emd_epi, dim=2, keepdim=True)
        e2m_loss = torch.mean(torch.norm(epi_emd_map - map_concat / torch.norm(map_concat, dim=2, keepdim=True), dim=2))
        m2e_loss = torch.mean(torch.norm(map_emd_epi - epi_concat / torch.norm(epi_concat, dim=2, keepdim=True), dim=2))
        loss = (e2m_loss + m2e_loss) / 2
        return loss

    def all_loss(self, epi, map):
        epi_share_feat = F.normalize(self.epi_share_proj(epi), dim=-1)
        map_share_feat = F.normalize(self.map_share_proj(map), dim=-1)

        epi_independent_feat = F.normalize(self.epi_independent_proj(epi), dim=-1)
        map_independent_feat = F.normalize(self.map_independent_proj(map), dim=-1)

        epi_concat_embed = torch.cat([epi_share_feat, epi_independent_feat], dim=-1)
        map_concat_embed = torch.cat([map_share_feat, map_independent_feat], dim=-1)


        epi_share_mean = torch.mean(epi_share_feat, dim=1)
        map_share_mean = torch.mean(map_share_feat, dim=1)
        epi_independent_mean = torch.mean(epi_independent_feat, dim=1)
        map_independent_mean = torch.mean(map_independent_feat, dim=1)


        contra_loss = self.contrastive_loss(epi_share_mean, map_share_mean)

        if self.loss_mode == 'contras':
            all_loss = contra_loss
        elif self.loss_mode == 'orth':
            orth_loss = self.orthogonal_loss(epi_share_mean, epi_independent_mean, map_share_mean, map_independent_mean)
            all_loss = contra_loss + orth_loss
        elif self.loss_mode == 'gauss':
            gauss_loss = self.gaussian_loss(epi_independent_mean, map_independent_mean)
            all_loss = contra_loss + gauss_loss
        elif self.loss_mode == 'geometric':
            geome_loss = self.geometric_loss(epi_share_mean, map_share_mean)
            all_loss = contra_loss + geome_loss
        elif self.loss_mode == 'translation':
            # trans_loss = self.translation_loss(epi_share_mean, epi_independent_mean, map_share_mean, map_independent_mean)
            trans_loss = self.translation_loss_seq(epi_concat_embed, map_concat_embed)
            orth_loss = self.orthogonal_loss(epi_share_mean, epi_independent_mean, map_share_mean, map_independent_mean)
            all_loss = contra_loss + trans_loss + orth_loss
        else:
            orth_loss = self.orthogonal_loss(epi_share_mean, epi_independent_mean, map_share_mean, map_independent_mean)
            gauss_loss = self.gaussian_loss(epi_independent_mean, map_independent_mean)
            all_loss = contra_loss + orth_loss + gauss_loss
        return all_loss, (epi_concat_embed, map_concat_embed)

    def infer_missing_modality(self, epi):
        # with torch.no_grad():
        epi_share_feat = F.normalize(self.epi_share_proj(epi), dim=-1)
        epi_independent_feat = F.normalize(self.epi_independent_proj(epi), dim=-1)

        epi_concat_embed = torch.cat([epi_share_feat, epi_independent_feat], dim=-1)

        epi_concat_tmp = epi_concat_embed.permute(0, 2, 1)  # 调整维度以适应池化层(batch, features, seq_len)
        epi_concat_tmp = self.adaptive_pool_epi(epi_concat_tmp)  # 调整epi的长度
        epi_concat_tmp = epi_concat_tmp.permute(0, 2, 1)  # 恢复原始维度

        map_concat_embed_infer = self.mlp_e2m(epi_concat_tmp)
        return epi_concat_embed, map_concat_embed_infer

    def forward(self, graph_data):
        """

        Args:
            map_data:
            graph_data: (batch_size, 100, 2)
            seq_data: (batch_size, 100, 13)
            mask:

        Returns:

        """

        input = graph_data
        input = self.seq_input_projection(input)
        epi_embed, (encode2, encode1, encode0) = self.seq_downsampleTrans(input)
        if self.modality == 'infer_map':
            epi_concat_embed, map_concat_embed = self.infer_missing_modality(epi_embed)
            fusion_embed = self.fusion_block(epi_concat_embed, map_concat_embed)
        else:
            fusion_embed = epi_embed

        decoder_output = self.seq_upsampleTrans(fusion_embed, encode0, encode1, encode2)
        output = self.output_head(decoder_output)
        output = self.classifier(output)
        output = rearrange(output, 'b h w c -> b c h w')

        return output

    def output_head(self, decoder_output, bins=50):
        x1 = decoder_output[:, :50, :]
        x2 = decoder_output[:, 50:, :]
        x1 = torch.tile(x1.unsqueeze(1),(1,bins,1,1))
        x2 = torch.tile(x2.unsqueeze(2), (1, 1, bins, 1))

        mean_out=(x1+x2)/2

        dot_out=x1*x2

        # (batch_size, 50, 50, embed_dim)
        return mean_out + dot_out
