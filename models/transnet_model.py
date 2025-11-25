# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GCNConv,  GATv2Conv, TopKPooling, SAGPooling, global_max_pool, InstanceNorm
# from torch.nn import TransformerEncoder, TransformerEncoderLayer

# class SpaceTempGoG_detr_dad(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, atten_feat_dim=2304, num_classes=2):
#         super(SpaceTempGoG_detr_dad, self).__init__()

#         self.num_heads = 4
#         self.input_dim = input_dim
#         self.embedding_dim = embedding_dim

#         # -----------------------
#         # Object graph features
#         # -----------------------
#         self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
#         self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
#         self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
#         self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

#         # -----------------------
#         # Spatial and temporal graph transformers
#         # -----------------------
#         self.gc1_spatial = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         self.gc1_temporal = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # -----------------------
#         # Cross-graph attention
#         # -----------------------
#         self.gc_cross = TransformerConv(
#             in_channels=(embedding_dim // 2 * self.num_heads) * 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc_cross_norm = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # Graph pooling
#         self.pool = SAGPooling(embedding_dim // 2 * self.num_heads, ratio=0.8)

#         # -----------------------
#         # I3D + attention features -> Transformer
#         # -----------------------
#         self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)
#         self.atten_fc = nn.Linear(atten_feat_dim, embedding_dim * 2)

#         encoder_layer = TransformerEncoderLayer(
#             d_model=embedding_dim * 2,
#             nhead=4,
#             batch_first=True
#         )
#         self.temporal_transformer_img = TransformerEncoder(encoder_layer, num_layers=2)
#         self.temporal_transformer_atten = TransformerEncoder(encoder_layer, num_layers=2)

#         # -----------------------
#         # Transformer for g_embed
#         # -----------------------
#         self.g_fc = nn.Linear(embedding_dim // 2 * self.num_heads, embedding_dim * 2)
#         self.temporal_transformer_g = TransformerEncoder(encoder_layer, num_layers=2)

#         # -----------------------
#         # Frame-level graph embedding for I3D and attention
#         # -----------------------
#         self.gc2_i3d = TransformerConv(
#             in_channels=embedding_dim * 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_atten = TransformerConv(
#             in_channels=embedding_dim * 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_norm_i3d = InstanceNorm(embedding_dim // 2 * self.num_heads)
#         self.gc2_norm_atten = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # -----------------------
#         # Classification
#         # -----------------------
#         concat_dim = (embedding_dim * 2) + (embedding_dim // 2 * self.num_heads) + (embedding_dim // 2 * self.num_heads)
#         self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
#         self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, edge_index, img_feat, video_adj_list, atten_feat, edge_embeddings,
#                 temporal_adj_list, temporal_edge_w, batch_vec):

#         # -----------------------
#         # Object graph processing
#         # -----------------------
#         x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
#         x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
#         x = torch.cat((x_feat, x_label), 1)

#         # Spatial graph
#         edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x.dtype).to(x.device)
#         n_embed_spatial = self.relu(self.gc1_norm1(self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)))

#         # Temporal graph
#         edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x.dtype).to(x.device)
#         n_embed_temporal = self.relu(self.gc1_norm2(self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal)))

#         # Cross-graph attention
#         n_embed_cross = torch.cat((n_embed_spatial, n_embed_temporal), dim=-1)
#         n_embed_cross = self.relu(self.gc_cross_norm(self.gc_cross(n_embed_cross, edge_index)))

#         # Graph pooling
#         n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed_cross, edge_index, None, batch_vec)
#         g_embed = global_max_pool(n_embed, batch_vec)

#         # -----------------------
#         # g_embed transformer
#         # -----------------------
#         g_embed_trans = self.g_fc(g_embed).unsqueeze(0)
#         g_embed_trans = self.temporal_transformer_g(g_embed_trans).squeeze(0)

#         # -----------------------
#         # I3D feature processing
#         # -----------------------
#         frame_embed_img = self.relu(self.gc2_norm_i3d(self.gc2_i3d(
#             self.temporal_transformer_img(self.img_fc(img_feat).unsqueeze(0)).squeeze(0), video_adj_list.clone())))

#         # -----------------------
#         # Attention feature processing
#         # -----------------------
#         frame_embed_atten = self.relu(self.gc2_norm_atten(self.gc2_atten(
#             self.temporal_transformer_atten(self.atten_fc(atten_feat).unsqueeze(0)).squeeze(0), video_adj_list.clone())))

#         # -----------------------
#         # Concatenate all features
#         # -----------------------
#         frame_embed_ = torch.cat((g_embed_trans, frame_embed_img, frame_embed_atten), dim=1)

#         # -----------------------
#         # Classification
#         # -----------------------
#         frame_embed_ = self.relu(self.classify_fc1(frame_embed_))
#         logits_mc = self.classify_fc2(frame_embed_)
#         probs_mc = self.softmax(logits_mc)

#         return logits_mc, probs_mc


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import (
#     TransformerConv,
#     SAGPooling,
#     global_max_pool,
#     InstanceNorm
# )
# from torch.nn import TransformerEncoder, TransformerEncoderLayer

# class SpaceTempGoG_detr_dad(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
#         super(SpaceTempGoG_detr_dad, self).__init__()

#         self.num_heads = 4
#         self.input_dim = input_dim
#         self.embedding_dim = embedding_dim

#         # -----------------------
#         # Object graph features
#         # -----------------------
#         self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
#         self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
#         self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
#         self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

#         # -----------------------
#         # Spatial and temporal graph transformers
#         # -----------------------
#         self.gc1_spatial = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         self.gc1_temporal = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # Graph pooling
#         self.pool = SAGPooling(embedding_dim * self.num_heads, ratio=0.8)

#         # -----------------------
#         # I3D features -> Transformer (causal)
#         # -----------------------
#         self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)
#         encoder_layer = TransformerEncoderLayer(
#             d_model=embedding_dim * 2,
#             nhead=4,
#             batch_first=True
#         )
#         self.temporal_transformer = TransformerEncoder(encoder_layer, num_layers=2)

#         encoder_layer_fusion = TransformerEncoderLayer(
#             d_model=embedding_dim * 2,
#             nhead=4,
#             batch_first=True,
#             dropout=0.1
#         )
#         self.temporal_fusion_transformer = TransformerEncoder(encoder_layer_fusion, num_layers=2)

#         # -----------------------
#         # Frame-level graph encoding
#         # -----------------------
#         self.gc2_sg = TransformerConv(
#             in_channels=embedding_dim * self.num_heads,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         self.gc2_i3d = TransformerConv(
#             in_channels=embedding_dim * 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # -----------------------
#         # Classification
#         # -----------------------
#         concat_dim = (embedding_dim // 2 * self.num_heads) + \
#                      (embedding_dim // 2 * self.num_heads) + \
#                      (embedding_dim * 2)
#         self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
#         self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings,
#                 temporal_adj_list, temporal_edge_w, batch_vec):

#         # -----------------------
#         # Object graph processing
#         # -----------------------
#         x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
#         x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
#         x = torch.cat((x_feat, x_label), 1)

#         # Spatial graph
#         edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x.dtype).to(x.device)
#         n_embed_spatial = self.relu(self.gc1_norm1(
#             self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)
#         ))

#         # Temporal graph (causal)
#         edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x.dtype).to(x.device)
#         n_embed_temporal = self.relu(self.gc1_norm2(
#             self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal)
#         ))

#         # Concat + pooling
#         n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
#         n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
#         g_embed = global_max_pool(n_embed, batch_vec)

#         # -----------------------
#         # I3D feature processing (causal)
#         # -----------------------
#         img_feat_proj = self.img_fc(img_feat).unsqueeze(0)  # (1, seq_len, d_model)
#         img_feat_orig = self.temporal_transformer(img_feat_proj, is_causal=True).squeeze(0)
#         img_feat_fusion = self.temporal_fusion_transformer(img_feat_proj, is_causal=True).squeeze(0)

#         # -----------------------
#         # Frame-level embeddings
#         # -----------------------
#         frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
#         frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat_orig, video_adj_list)))

#         # Concatenate all features
#         frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img, img_feat_fusion), 1)

#         # -----------------------
#         # Classification
#         # -----------------------
#         frame_embed_ = self.relu(self.classify_fc1(frame_embed_))
#         logits_mc = self.classify_fc2(frame_embed_)
#         probs_mc = self.softmax(logits_mc)

#         return logits_mc, probs_mc

import torch
import torch.nn as nn
from torch_geometric.nn import (
    TransformerConv,
    SAGPooling,
    global_max_pool,
    InstanceNorm
)
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# class SpaceTempGoG_detr_dad(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, atten_feat_dim=2304, num_classes=2):
#         super(SpaceTempGoG_detr_dad, self).__init__()

#         self.num_heads = 4
#         self.input_dim = input_dim
#         self.embedding_dim = embedding_dim

#         # -----------------------
#         # Object graph features
#         # -----------------------
#         self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
#         self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
#         self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
#         self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

#         # -----------------------
#         # Spatial and temporal graph transformers
#         # -----------------------
#         self.gc1_spatial = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         self.gc1_temporal = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # Graph pooling
#         self.pool = SAGPooling(embedding_dim * self.num_heads, ratio=0.8)

#         # -----------------------
#         # I3D and Attention-SlowFast features -> Transformers
#         # -----------------------
#         self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)
#         self.atten_fc = nn.Linear(atten_feat_dim, embedding_dim * 2)

#         encoder_layer_img = TransformerEncoderLayer(
#             d_model=embedding_dim * 2, nhead=self.num_heads, batch_first=True
#         )
#         encoder_layer_atten = TransformerEncoderLayer(
#             d_model=embedding_dim * 2, nhead=self.num_heads, batch_first=True
#         )

#         self.temporal_transformer_img = TransformerEncoder(encoder_layer_img, num_layers=2)
#         self.temporal_transformer_atten = TransformerEncoder(encoder_layer_atten, num_layers=2)

#         # -----------------------
#         # LSTMs (num_layers=1, hidden_size = input_size)
#         # -----------------------
#         self.temporal_lstm_graph = nn.LSTM(
#             input_size=embedding_dim * self.num_heads,
#             hidden_size=embedding_dim * self.num_heads,
#             num_layers=1,
#             batch_first=True
#         )
#         self.temporal_lstm_img = nn.LSTM(
#             input_size=embedding_dim * 2,
#             hidden_size=embedding_dim * 2,
#             num_layers=1,
#             batch_first=True
#         )
#         self.temporal_lstm_atten = nn.LSTM(
#             input_size=embedding_dim * 2,
#             hidden_size=embedding_dim * 2,
#             num_layers=1,
#             batch_first=True
#         )

#         # -----------------------
#         # Classification
#         # -----------------------
#         concat_dim = embedding_dim * self.num_heads + embedding_dim * 2 + embedding_dim * 2
#         self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
#         self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, edge_index, img_feat, video_adj_list, atten_feat,
#                 edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):

#         # -----------------------
#         # Object graph processing
#         # -----------------------
#         x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
#         x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
#         x = torch.cat((x_feat, x_label), 1)

#         # Spatial graph
#         edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x)
#         n_embed_spatial = self.relu(self.gc1_norm1(
#             self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)
#         ))

#         # Temporal graph
#         edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x)
#         n_embed_temporal = self.relu(self.gc1_norm2(
#             self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal)
#         ))

#         # Concat + pooling
#         n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
#         n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
#         g_embed = global_max_pool(n_embed, batch_vec)  # (num_nodes, feat_dim)

#         # -----------------------
#         # LSTM over graph pooled features
#         # -----------------------
#         g_embed_seq = g_embed.unsqueeze(0)  # Add sequence dimension: (1, num_nodes, feat_dim)
#         g_embed_seq, _ = self.temporal_lstm_graph(g_embed_seq)
#         lstm_out_graph = g_embed_seq.squeeze(0)  # Back to (num_nodes, feat_dim)

#         # -----------------------
#         # I3D feature processing
#         # -----------------------
#         img_feat_proj = self.img_fc(img_feat)
#         img_feat_trans = self.temporal_transformer_img(img_feat_proj)
#         img_feat_seq = img_feat_trans.unsqueeze(0)  # (1, num_nodes, feat_dim)
#         img_feat_seq, _ = self.temporal_lstm_img(img_feat_seq)
#         lstm_out_img = img_feat_seq.squeeze(0)

#         # -----------------------
#         # Attention SlowFast feature processing
#         # -----------------------
#         atten_feat_proj = self.atten_fc(atten_feat)
#         atten_feat_trans = self.temporal_transformer_atten(atten_feat_proj)
#         atten_feat_seq = atten_feat_trans.unsqueeze(0)  # (1, num_nodes, feat_dim)
#         atten_feat_seq, _ = self.temporal_lstm_atten(atten_feat_seq)
#         lstm_out_atten = atten_feat_seq.squeeze(0)

#         # -----------------------
#         # Concatenate all LSTM outputs
#         # -----------------------
#         fused_feat = torch.cat((lstm_out_graph, lstm_out_img, lstm_out_atten), dim=1)

#         # -----------------------

#         # -----------------------
#         # Classification
#         # -----------------------
#         fused_feat = self.relu(self.classify_fc1(fused_feat))
#         logits_mc = self.classify_fc2(fused_feat)
#         probs_mc = self.softmax(logits_mc)

#         # If you compute min_pred from probs or logits, inspect them:
#         print("logits_mc.shape:", logits_mc.shape, "min/max logits:", logits_mc.min().item(), logits_mc.max().item())
#         print("probs_mc.shape:", probs_mc.shape, "min/max probs:", probs_mc.min().item(), probs_mc.max().item())

#         return logits_mc, probs_mc




# class Trasnet(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, atten_feat_dim=2304, num_classes=2):
#         super(Trasnet, self).__init__()

#         self.num_heads = 4
#         self.input_dim = input_dim
#         self.embedding_dim = embedding_dim

#         # -----------------------
#         # Object graph features
#         # -----------------------
#         self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
#         self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
#         self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
#         self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

#         # -----------------------
#         # Spatial and temporal graph transformers
#         # -----------------------
#         self.gc1_spatial = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         self.gc1_temporal = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # Graph pooling
#         self.pool = SAGPooling(embedding_dim * self.num_heads, ratio=0.8)

#         # -----------------------
#         # I3D and Attention-SlowFast features -> Transformers
#         # -----------------------
#         self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)
#         self.atten_fc = nn.Linear(atten_feat_dim, embedding_dim * 2)

#         encoder_layer_img = TransformerEncoderLayer(
#             d_model=embedding_dim * 2, nhead=self.num_heads, batch_first=True
#         )
#         encoder_layer_atten = TransformerEncoderLayer(
#             d_model=embedding_dim * 2, nhead=self.num_heads, batch_first=True
#         )

#         self.temporal_transformer_img = TransformerEncoder(encoder_layer_img, num_layers=2)
#         self.temporal_transformer_atten = TransformerEncoder(encoder_layer_atten, num_layers=2)

#         # -----------------------
#         # LSTMs (num_layers=1, hidden_size = input_size)
#         # -----------------------
#         self.temporal_lstm_graph = nn.LSTM(
#             input_size=embedding_dim * self.num_heads,
#             hidden_size=embedding_dim * self.num_heads,
#             num_layers=1,
#             batch_first=True
#         )
#         self.temporal_lstm_img = nn.LSTM(
#             input_size=embedding_dim * 2,
#             hidden_size=embedding_dim * 2,
#             num_layers=1,
#             batch_first=True
#         )
#         self.temporal_lstm_atten = nn.LSTM(
#             input_size=embedding_dim * 2,
#             hidden_size=embedding_dim * 2,
#             num_layers=1,
#             batch_first=True
#         )

#         # -----------------------
#         # Classification
#         # -----------------------
#         concat_dim = embedding_dim * self.num_heads + embedding_dim * 2 + embedding_dim * 2
#         self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
#         self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, edge_index, img_feat, video_adj_list, atten_feat,
#                 edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):
    
#         # -----------------------
#         # Helper function: sanitize activations
#         # -----------------------
#         def sanitize(tensor, name):
#             # if torch.isnan(tensor).any() or torch.isinf(tensor).any():
#             #     # print(f"[⚠️ Sanitizing {name}] min={tensor.min().item()}, max={tensor.max().item()}")
#             if torch.isnan(tensor).any() or torch.isinf(tensor).any():
#                 valid_mask = torch.isfinite(tensor)
#                 if valid_mask.any():  # Only compute if there are valid numbers
#                     finite_min = tensor[valid_mask].min().item()
#                     finite_max = tensor[valid_mask].max().item()
#                     print(f"[⚠️ Sanitizing {name}] finite_min={finite_min:.4f}, finite_max={finite_max:.4f}")
#                 else:
#                     print(f"[⚠️ Sanitizing {name}] all values are NaN or Inf!")
#             tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e3, neginf=-1e3)
#             tensor = torch.clamp(tensor, -1e3, 1e3)
#             return tensor

#         # img_feat = torch.nan_to_num(img_feat, nan=0.0, posinf=1e6, neginf=-1e6)  # optional
#         # min_val = img_feat.amin(dim=-1, keepdim=True)
#         # max_val = img_feat.amax(dim=-1, keepdim=True)
#         # img_feat = (img_feat - min_val) / (max_val - min_val + 1e-6)

#         # -----------------------
#         # Object graph processing
#         # -----------------------
#         x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
#         # x_feat = sanitize(x_feat, "x_feat")
    
#         x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
#         # x_label = sanitize(x_label, "x_label")
    
#         x = torch.cat((x_feat, x_label), 1)
#         # x = sanitize(x, "x (concatenated)")
    
#         # -----------------------
#         # Spatial graph
#         # -----------------------
#         edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x)
#         n_embed_spatial = self.relu(self.gc1_norm1(
#             self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)
#         ))
#         # n_embed_spatial = sanitize(n_embed_spatial, "n_embed_spatial")
    
#         # -----------------------
#         # Temporal graph
#         # -----------------------
#         edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x)
#         n_embed_temporal = self.relu(self.gc1_norm2(
#             self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal)
#         ))
#         # n_embed_temporal = sanitize(n_embed_temporal, "n_embed_temporal")
    
#         # -----------------------
#         # Concat + pooling
#         # -----------------------
#         n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
#         # n_embed = sanitize(n_embed, "n_embed before pooling")
    
#         n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
#         # n_embed = sanitize(n_embed, "n_embed after pooling")
    
#         g_embed = global_max_pool(n_embed, batch_vec)
#         # g_embed = sanitize(g_embed, "g_embed")
    
#         # -----------------------
#         # LSTM over graph pooled features
#         # -----------------------
#         g_embed_seq = g_embed.unsqueeze(0)
#         g_embed_seq, _ = self.temporal_lstm_graph(g_embed_seq)
#         lstm_out_graph = g_embed_seq.squeeze(0)
#         # lstm_out_graph = sanitize(lstm_out_graph, "lstm_out_graph")
    
#         # -----------------------
#         # I3D feature processing
#         # -----------------------
#         img_feat_proj = self.img_fc(img_feat)
#         img_feat_proj = sanitize(img_feat_proj, "img_feat_proj")
    
#         img_feat_trans = self.temporal_transformer_img(img_feat_proj)
#         img_feat_trans = sanitize(img_feat_trans, "img_feat_trans")
    
#         img_feat_seq = img_feat_trans.unsqueeze(0)
#         img_feat_seq, _ = self.temporal_lstm_img(img_feat_seq)
#         lstm_out_img = img_feat_seq.squeeze(0)
#         lstm_out_img = sanitize(lstm_out_img, "lstm_out_img")
    
#         # -----------------------
#         # Attention SlowFast feature processing
#         # -----------------------
#         atten_feat_proj = self.atten_fc(atten_feat)
#         atten_feat_proj = sanitize(atten_feat_proj, "atten_feat_proj")
    
#         atten_feat_trans = self.temporal_transformer_atten(atten_feat_proj)
#         atten_feat_trans = sanitize(atten_feat_trans, "atten_feat_trans")
    
#         atten_feat_seq = atten_feat_trans.unsqueeze(0)
#         atten_feat_seq, _ = self.temporal_lstm_atten(atten_feat_seq)
#         lstm_out_atten = atten_feat_seq.squeeze(0)
#         lstm_out_atten = sanitize(lstm_out_atten, "lstm_out_atten")
    
#         # -----------------------
#         # Concatenate all LSTM outputs
#         # -----------------------
#         fused_feat = torch.cat((lstm_out_graph, lstm_out_img, lstm_out_atten), dim=1)
#         fused_feat = sanitize(fused_feat, "fused_feat before classification")
    
#         # -----------------------
#         # Classification
#         # -----------------------
#         fused_feat = self.relu(self.classify_fc1(fused_feat))
#         # fused_feat = sanitize(fused_feat, "fused_feat after fc1")
    
#         logits_mc = self.classify_fc2(fused_feat)
#         # logits_mc = sanitize(logits_mc, "logits_mc")
    
#         probs_mc = self.softmax(logits_mc)
#         # probs_mc = sanitize(probs_mc, "probs_mc")

    
#         return logits_mc, probs_mc

## Trans and 2 LSTM - CVPR - DAD 72
import torch
import torch.nn as nn
from torch_geometric.nn import (
    TransformerConv,
    SAGPooling,
    global_max_pool,
    InstanceNorm
)
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Trans_LSTM(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
        super(Trans_LSTM, self).__init__()

        self.num_heads = 4
        self.graph_heads = 4             # old setup - 4
        self.encoder_layers = 2          # old setup - 2
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # -----------------------
        # Object graph features
        # -----------------------
        self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
        self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
        self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
        self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

        # -----------------------
        # Spatial and temporal graph transformers
        # -----------------------
        self.gc1_spatial = TransformerConv(
            in_channels=embedding_dim * 2 + embedding_dim // 2,
            out_channels=embedding_dim // 2,
            heads=self.graph_heads,
            edge_dim=1,
            beta=True
        )
        self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.graph_heads)

        self.gc1_temporal = TransformerConv(
            in_channels=embedding_dim * 2 + embedding_dim // 2,
            out_channels=embedding_dim // 2,
            heads=self.graph_heads,
            edge_dim=1,
            beta=True
        )
        self.gc1_norm2 = InstanceNorm(embedding_dim // 2 * self.graph_heads)

        # Graph pooling
        self.pool = SAGPooling(embedding_dim * self.graph_heads, ratio=0.8)
        # self.pool = TopKPooling(embedding_dim * self.num_heads, ratio=0.8)

        # -----------------------
        # I3D features -> Transformer
        # -----------------------
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

        encoder_layer_img = TransformerEncoderLayer(
            d_model=embedding_dim * 2, nhead=self.num_heads, batch_first=True
        )
        self.temporal_transformer_img = TransformerEncoder(encoder_layer_img, num_layers=self.encoder_layers)

        # -----------------------
        # LSTMs (num_layers=1, hidden_size = input_size)
        # -----------------------
        self.temporal_lstm_graph = nn.LSTM(
            input_size=embedding_dim * self.graph_heads,
            hidden_size=embedding_dim * self.graph_heads,
            num_layers=1,
            batch_first=True
        )
        self.temporal_lstm_img = nn.LSTM(
            input_size=embedding_dim * 2,
            hidden_size=embedding_dim * 2,
            num_layers=1,
            batch_first=True
        )

        # -----------------------
        # Classification
        # -----------------------
        concat_dim = embedding_dim * self.graph_heads + embedding_dim * 2
        self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
        self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, video_adj_list,                        # att_feat,
                edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):

        # -----------------------
        # Helper function
        # -----------------------
        def sanitize(tensor, name):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                valid_mask = torch.isfinite(tensor)
                if valid_mask.any():
                    finite_min = tensor[valid_mask].min().item()
                    finite_max = tensor[valid_mask].max().item()
                    print(f"[⚠️ Sanitizing {name}] finite_min={finite_min:.4f}, finite_max={finite_max:.4f}")
                else:
                    print(f"[⚠️ Sanitizing {name}] all values are NaN or Inf!")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e3, neginf=-1e3)
            tensor = torch.clamp(tensor, -1e3, 1e3)
            return tensor

        # -----------------------
        # Object graph processing
        # -----------------------
        x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
        x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
        x = torch.cat((x_feat, x_label), 1)

        # -----------------------
        # Spatial graph
        # -----------------------
        edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x)
        n_embed_spatial = self.relu(self.gc1_norm1(
            self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)
        ))

        # -----------------------
        # Temporal graph
        # -----------------------
        edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x)
        n_embed_temporal = self.relu(self.gc1_norm2(
            self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal)
        ))

        # -----------------------
        # Concat + pooling
        # -----------------------
        n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
        n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
        g_embed = global_max_pool(n_embed, batch_vec)

        # -----------------------
        # LSTM over graph pooled features
        # -----------------------
        g_embed_seq = g_embed.unsqueeze(0)
        g_embed_seq, _ = self.temporal_lstm_graph(g_embed_seq)
        lstm_out_graph = g_embed_seq.squeeze(0)

        # -----------------------
        # I3D feature processing
        # -----------------------
        img_feat_proj = self.img_fc(img_feat)
        img_feat_proj = sanitize(img_feat_proj, "img_feat_proj")

        img_feat_trans = self.temporal_transformer_img(img_feat_proj, is_causal=True)
        img_feat_trans = sanitize(img_feat_trans, "img_feat_trans")

        img_feat_seq = img_feat_trans.unsqueeze(0)
        img_feat_seq, _ = self.temporal_lstm_img(img_feat_seq)
        lstm_out_img = img_feat_seq.squeeze(0)
        lstm_out_img = sanitize(lstm_out_img, "lstm_out_img")

        # # Add batch dimension for Transformer
        # img_feat_proj = img_feat_proj.unsqueeze(0)                       # [1, seq_len, embed_dim]
        # img_feat_trans = self.temporal_transformer_img(img_feat_proj, is_causal=True)
        # img_feat_trans = sanitize(img_feat_trans, "img_feat_trans")
        
        # # Apply LSTM next
        # img_feat_seq, _ = self.temporal_lstm_img(img_feat_trans)          # [1, seq_len, embed_dim]
        # lstm_out_img = img_feat_seq.squeeze(0)                            # [seq_len, embed_dim]
        # lstm_out_img = sanitize(lstm_out_img, "lstm_out_img")       

        # -----------------------
        # Concatenate all LSTM outputs
        # -----------------------
        fused_feat = torch.cat((lstm_out_graph, lstm_out_img), dim=1)
        fused_feat = sanitize(fused_feat, "fused_feat before classification")

        # -----------------------
        # Classification
        # -----------------------
        fused_feat = self.relu(self.classify_fc1(fused_feat))
        logits_mc = self.classify_fc2(fused_feat)
        probs_mc = self.softmax(logits_mc)

        return logits_mc, probs_mc

# Same as above but the temporal object graph removed - DAD 70
class Trans_LSTM_Sans_Obj_Temp_Graph(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
        super(Trans_LSTM_Sans_Obj_Temp_Graph, self).__init__()

        self.num_heads = 4
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # -----------------------
        # Object graph features
        # -----------------------
        self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
        self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
        self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
        self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

        # -----------------------
        # Spatial graph transformer
        # -----------------------
        self.gc1_spatial = TransformerConv(
            in_channels=embedding_dim * 2 + embedding_dim // 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads,
            edge_dim=1,
            beta=True
        )
        self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

        # Graph pooling
        self.pool = SAGPooling(embedding_dim // 2 * self.num_heads, ratio=0.8)

        # -----------------------
        # I3D features -> Transformer
        # -----------------------
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

        encoder_layer_img = TransformerEncoderLayer(
            d_model=embedding_dim * 2, nhead=self.num_heads, batch_first=True
        )
        self.temporal_transformer_img = TransformerEncoder(encoder_layer_img, num_layers=2)

        # -----------------------
        # LSTMs
        # -----------------------
        self.temporal_lstm_graph = nn.LSTM(
            input_size=embedding_dim // 2 * self.num_heads,
            hidden_size=embedding_dim // 2 * self.num_heads,
            num_layers=1,
            batch_first=True
        )
        self.temporal_lstm_img = nn.LSTM(
            input_size=embedding_dim * 2,
            hidden_size=embedding_dim * 2,
            num_layers=1,
            batch_first=True
        )

        # -----------------------
        # Classification
        # -----------------------
        concat_dim = (embedding_dim // 2 * self.num_heads) + (embedding_dim * 2)
        self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
        self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    # def forward(self, x, edge_index, img_feat, video_adj_list, att_feat,
    #             edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):
    def forward(self, x, edge_index, img_feat, video_adj_list,
                edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):

        # -----------------------
        # Helper function
        # -----------------------
        def sanitize(tensor, name):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                valid_mask = torch.isfinite(tensor)
                if valid_mask.any():
                    finite_min = tensor[valid_mask].min().item()
                    finite_max = tensor[valid_mask].max().item()
                    print(f"[⚠️ Sanitizing {name}] finite_min={finite_min:.4f}, finite_max={finite_max:.4f}")
                else:
                    print(f"[⚠️ Sanitizing {name}] all values are NaN or Inf!")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e3, neginf=-1e3)
            tensor = torch.clamp(tensor, -1e3, 1e3)
            return tensor

        # -----------------------
        # Object graph processing
        # -----------------------
        x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
        x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
        x = torch.cat((x_feat, x_label), 1)

        # -----------------------
        # Spatial graph
        # -----------------------
        edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x)
        n_embed_spatial = self.relu(self.gc1_norm1(
            self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)
        ))

        # -----------------------
        # Pooling
        # -----------------------
        n_embed, edge_index, _, batch_vec, _, _ = self.pool(
            n_embed_spatial, edge_index, None, batch_vec
        )
        g_embed = global_max_pool(n_embed, batch_vec)

        # -----------------------
        # LSTM over graph pooled features
        # -----------------------
        g_embed_seq = g_embed.unsqueeze(0)
        g_embed_seq, _ = self.temporal_lstm_graph(g_embed_seq)
        lstm_out_graph = g_embed_seq.squeeze(0)

        # -----------------------
        # I3D feature processing
        # -----------------------
        img_feat_proj = self.img_fc(img_feat)
        img_feat_proj = sanitize(img_feat_proj, "img_feat_proj")

        img_feat_trans = self.temporal_transformer_img(img_feat_proj, is_causal=True)
        img_feat_trans = sanitize(img_feat_trans, "img_feat_trans")

        img_feat_seq = img_feat_trans.unsqueeze(0)
        img_feat_seq, _ = self.temporal_lstm_img(img_feat_seq)
        lstm_out_img = img_feat_seq.squeeze(0)
        lstm_out_img = sanitize(lstm_out_img, "lstm_out_img")

        # -----------------------
        # Concatenate all LSTM outputs
        # -----------------------
        fused_feat = torch.cat((lstm_out_graph, lstm_out_img), dim=1)
        fused_feat = sanitize(fused_feat, "fused_feat before classification")

        # -----------------------
        # Classification
        # -----------------------
        fused_feat = self.relu(self.classify_fc1(fused_feat))
        logits_mc = self.classify_fc2(fused_feat)
        probs_mc = self.softmax(logits_mc)

        return logits_mc, probs_mc

# same as above but transformer added for object graph - 53.53
class Trans_LSTM_Sans_Obj_Temp_Graph_Trans_Obj(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
        super(Trans_LSTM_Sans_Obj_Temp_Graph_Trans_Obj, self).__init__()

        self.num_heads = 4
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # -----------------------
        # Object graph features
        # -----------------------
        self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
        self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
        self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
        self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

        # -----------------------
        # Spatial graph transformer
        # -----------------------
        self.gc1_spatial = TransformerConv(
            in_channels=embedding_dim * 2 + embedding_dim // 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads,
            edge_dim=1,
            beta=True
        )
        self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

        # Graph pooling
        self.pool = SAGPooling(embedding_dim // 2 * self.num_heads, ratio=0.8)

        # -----------------------
        # Transformer for Graph Embeddings
        # -----------------------
        encoder_layer_graph = TransformerEncoderLayer(
            d_model=embedding_dim // 2 * self.num_heads,
            nhead=self.num_heads,
            batch_first=True
        )
        self.temporal_transformer_graph = TransformerEncoder(
            encoder_layer_graph, num_layers=2
        )

        # -----------------------
        # I3D features -> Transformer
        # -----------------------
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

        encoder_layer_img = TransformerEncoderLayer(
            d_model=embedding_dim * 2, nhead=self.num_heads, batch_first=True
        )
        self.temporal_transformer_img = TransformerEncoder(
            encoder_layer_img, num_layers=2
        )

        # -----------------------
        # LSTMs
        # -----------------------
        self.temporal_lstm_graph = nn.LSTM(
            input_size=embedding_dim // 2 * self.num_heads,
            hidden_size=embedding_dim // 2 * self.num_heads,
            num_layers=1,
            batch_first=True
        )
        self.temporal_lstm_img = nn.LSTM(
            input_size=embedding_dim * 2,
            hidden_size=embedding_dim * 2,
            num_layers=1,
            batch_first=True
        )

        # -----------------------
        # Classification
        # -----------------------
        concat_dim = (embedding_dim // 2 * self.num_heads) + (embedding_dim * 2)
        self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
        self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, video_adj_list, att_feat,
                edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):

        # -----------------------
        # Helper: sanitize
        # -----------------------
        def sanitize(tensor, name):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                valid_mask = torch.isfinite(tensor)
                if valid_mask.any():
                    finite_min = tensor[valid_mask].min().item()
                    finite_max = tensor[valid_mask].max().item()
                    print(f"[⚠️ Sanitizing {name}] finite_min={finite_min:.4f}, finite_max={finite_max:.4f}")
                else:
                    print(f"[⚠️ Sanitizing {name}] all values are NaN or Inf!")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e3, neginf=-1e3)
            tensor = torch.clamp(tensor, -1e3, 1e3)
            return tensor

        # -----------------------
        # Object graph processing
        # -----------------------
        x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
        x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
        x = torch.cat((x_feat, x_label), 1)

        # -----------------------
        # Spatial graph
        # -----------------------
        edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x)
        n_embed_spatial = self.relu(self.gc1_norm1(
            self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)
        ))

        # -----------------------
        # Pooling
        # -----------------------
        n_embed, edge_index, _, batch_vec, _, _ = self.pool(
            n_embed_spatial, edge_index, None, batch_vec
        )
        g_embed = global_max_pool(n_embed, batch_vec)
        g_embed = sanitize(g_embed, "g_embed before transformer")

        # -----------------------
        # Transformer on Graph Embeddings
        # -----------------------
        g_embed_seq = g_embed.unsqueeze(0)  # (1, N, dim)
        g_embed_trans = self.temporal_transformer_graph(g_embed_seq, is_causal=True)
        g_embed_trans = sanitize(g_embed_trans, "g_embed_trans")

        # -----------------------
        # LSTM over transformed graph features
        # -----------------------
        g_embed_seq, _ = self.temporal_lstm_graph(g_embed_trans)
        lstm_out_graph = g_embed_seq.squeeze(0)

        # -----------------------
        # I3D feature processing
        # -----------------------
        img_feat_proj = self.img_fc(img_feat)
        img_feat_proj = sanitize(img_feat_proj, "img_feat_proj")

        img_feat_trans = self.temporal_transformer_img(img_feat_proj, is_causal=True)
        img_feat_trans = sanitize(img_feat_trans, "img_feat_trans")

        img_feat_seq = img_feat_trans.unsqueeze(0)
        img_feat_seq, _ = self.temporal_lstm_img(img_feat_seq)
        lstm_out_img = img_feat_seq.squeeze(0)
        lstm_out_img = sanitize(lstm_out_img, "lstm_out_img")

        # -----------------------
        # Concatenate all LSTM outputs
        # -----------------------
        fused_feat = torch.cat((lstm_out_graph, lstm_out_img), dim=1)
        fused_feat = sanitize(fused_feat, "fused_feat before classification")

        # -----------------------
        # Classification
        # -----------------------
        fused_feat = self.relu(self.classify_fc1(fused_feat))
        logits_mc = self.classify_fc2(fused_feat)
        probs_mc = self.softmax(logits_mc)

        return logits_mc, probs_mc


# import torch
# import torch.nn as nn
# from torch_geometric.nn import (
#     TransformerConv,
#     SAGPooling,
#     global_max_pool,
#     InstanceNorm
# )
# from torch.nn import TransformerEncoder, TransformerEncoderLayer


# class Trasnet(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
#         super(Trasnet, self).__init__()

#         self.num_heads = 4
#         self.input_dim = input_dim
#         self.embedding_dim = embedding_dim

#         # -----------------------
#         # Object graph features
#         # -----------------------
#         self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
#         self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
#         self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
#         self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

#         # -----------------------
#         # Spatial and temporal graph transformers
#         # -----------------------
#         self.gc1_spatial = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         self.gc1_temporal = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # Graph pooling
#         self.pool = SAGPooling(embedding_dim * self.num_heads, ratio=0.8)

#         # -----------------------
#         # I3D feature encoder
#         # -----------------------
#         self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)
#         encoder_layer_img = TransformerEncoderLayer(
#             d_model=embedding_dim * 2, nhead=self.num_heads, batch_first=True
#         )
#         self.temporal_transformer_img = TransformerEncoder(encoder_layer_img, num_layers=2)

#         # -----------------------
#         # Graph feature encoder
#         # -----------------------
#         encoder_layer_graph = TransformerEncoderLayer(
#             d_model=embedding_dim * self.num_heads, nhead=self.num_heads, batch_first=True
#         )
#         self.temporal_transformer_graph = TransformerEncoder(encoder_layer_graph, num_layers=2)

#         # -----------------------
#         # Classification
#         # -----------------------
#         concat_dim = embedding_dim * self.num_heads + embedding_dim * 2
#         self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
#         self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, edge_index, img_feat, video_adj_list, atten_feat,
#                 edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):
#         # -----------------------
#         # Helper function
#         # -----------------------
#         def sanitize(tensor, name):
#             if torch.isnan(tensor).any() or torch.isinf(tensor).any():
#                 valid_mask = torch.isfinite(tensor)
#                 if valid_mask.any():
#                     finite_min = tensor[valid_mask].min().item()
#                     finite_max = tensor[valid_mask].max().item()
#                     print(f"[⚠️ Sanitizing {name}] finite_min={finite_min:.4f}, finite_max={finite_max:.4f}")
#                 else:
#                     print(f"[⚠️ Sanitizing {name}] all values are NaN or Inf!")
#             tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e3, neginf=-1e3)
#             tensor = torch.clamp(tensor, -1e3, 1e3)
#             return tensor

#         # -----------------------
#         # Object graph processing
#         # -----------------------
#         x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
#         x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
#         x = torch.cat((x_feat, x_label), 1)

#         # -----------------------
#         # Spatial graph
#         # -----------------------
#         edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x)
#         n_embed_spatial = self.relu(
#             self.gc1_norm1(self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial))
#         )

#         # -----------------------
#         # Temporal graph
#         # -----------------------
#         edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x)
#         n_embed_temporal = self.relu(
#             self.gc1_norm2(self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal))
#         )

#         # -----------------------
#         # Concat + pooling
#         # -----------------------
#         n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
#         n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
#         g_embed = global_max_pool(n_embed, batch_vec)

#         # -----------------------
#         # Transformer over graph pooled features
#         # -----------------------
#         g_embed_seq = g_embed.unsqueeze(0)  # (1, B, D)
#         g_embed_enc = self.temporal_transformer_graph(g_embed_seq)
#         g_embed_enc = g_embed_enc.squeeze(0)
#         g_embed_enc = sanitize(g_embed_enc, "g_embed_enc")

#         # -----------------------
#         # Transformer over image features
#         # -----------------------
#         img_feat_proj = self.img_fc(img_feat)
#         img_feat_proj = sanitize(img_feat_proj, "img_feat_proj")

#         img_feat_enc = self.temporal_transformer_img(img_feat_proj.unsqueeze(0))
#         img_feat_enc = img_feat_enc.squeeze(0)
#         img_feat_enc = sanitize(img_feat_enc, "img_feat_enc")

#         # -----------------------
#         # Concatenate both transformer outputs
#         # -----------------------
#         fused_feat = torch.cat((g_embed_enc, img_feat_enc), dim=1)
#         fused_feat = sanitize(fused_feat, "fused_feat before classification")

#         # -----------------------
#         # Classification
#         # -----------------------
#         fused_feat = self.relu(self.classify_fc1(fused_feat))
#         logits_mc = self.classify_fc2(fused_feat)
#         probs_mc = self.softmax(logits_mc)

#         return logits_mc, probs_mc

## Trans then multi then combine three output - now obj_feat also added
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, InstanceNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention

class Trans_Obj_Net(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, obj_feat_dim=512, num_classes=2):
        super(Trans_Obj_Net, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = 4

        # -----------------------
        # Image feature projection
        # -----------------------
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

        # -----------------------
        # Object feature projection (obj_feat size = 512)
        # -----------------------
        self.obj_fc = nn.Linear(obj_feat_dim, embedding_dim * 2)
        self.obj_lstm = nn.LSTM(embedding_dim * 2, embedding_dim * 2, batch_first=True)

        # -----------------------
        # Single causal TransformerEncoder
        # -----------------------
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim * 2,
            nhead=self.num_heads,
            batch_first=True
        )
        self.temporal_transformer = TransformerEncoder(encoder_layer, num_layers=2)

        # -----------------------
        # Optional fusion layer to capture complementary patterns
        # -----------------------
        self.fusion_fc = nn.Linear(embedding_dim * 2, embedding_dim * 2)

        # -----------------------
        # Multihead attention branch (self-attention)
        # -----------------------
        self.img_attn = MultiheadAttention(
            embed_dim=embedding_dim * 2,
            num_heads=self.num_heads,
            batch_first=True
        )

        # -----------------------
        # Graph TransformerConv branches
        # -----------------------
        self.gc_orig = TransformerConv(
            in_channels=embedding_dim * 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads
        )
        self.gc_attn = TransformerConv(
            in_channels=embedding_dim * 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads
        )
        self.norm_orig = InstanceNorm(embedding_dim // 2 * self.num_heads)
        self.norm_attn = InstanceNorm(embedding_dim // 2 * self.num_heads)

        # -----------------------
        # Classification
        # -----------------------
        # Adding obj_feat output to concat_dim
        concat_dim = (embedding_dim // 2 * self.num_heads) * 2 + embedding_dim * 2 + embedding_dim * 2
        self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
        self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, video_adj_list, obj_feat, edge_embeddings=None,
                temporal_adj_list=None, temporal_edge_w=None, batch_vec=None):
        """
        img_feat: (seq_len, img_feat_dim)
        obj_feat: (seq_len, 512)
        video_adj_list: graph edges for TransformerConv
        """

        # -----------------------
        # Image feature projection
        # -----------------------
        img_feat_proj = self.img_fc(img_feat).unsqueeze(0)  # (1, seq_len, d_model)

        # -----------------------
        # Single causal Transformer
        # -----------------------
        img_feat_trans = self.temporal_transformer(img_feat_proj, is_causal=True)
        img_feat_trans = self.fusion_fc(img_feat_trans.squeeze(0))  # fusion after transformer

        # -----------------------
        # Multihead attention branch
        # -----------------------
        img_feat_attn, _ = self.img_attn(
            img_feat_trans.unsqueeze(0),  # Q
            img_feat_trans.unsqueeze(0),  # K
            img_feat_trans.unsqueeze(0),  # V
            is_causal=True
        )
        img_feat_attn = img_feat_attn.squeeze(0)

        # -----------------------
        # Graph TransformerConv
        # -----------------------
        frame_embed_orig = self.relu(self.norm_orig(self.gc_orig(img_feat_trans, video_adj_list)))
        frame_embed_attn = self.relu(self.norm_attn(self.gc_attn(img_feat_attn, video_adj_list)))

        # -----------------------
        # Object feature processing
        # -----------------------
        obj_feat_proj = self.obj_fc(obj_feat).unsqueeze(0)  # (1, seq_len, d_model)
        obj_feat_lstm, _ = self.obj_lstm(obj_feat_proj)
        obj_feat_lstm = obj_feat_lstm.squeeze(0)

        # -----------------------
        # Concatenate all features
        # -----------------------
        frame_embed_ = torch.cat((frame_embed_orig, frame_embed_attn, img_feat_trans, obj_feat_lstm), dim=1)

        # -----------------------
        # Classification
        # -----------------------
        frame_embed_ = self.relu(self.classify_fc1(frame_embed_))
        logits = self.classify_fc2(frame_embed_)
        probs = self.softmax(logits)

        return logits, probs


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, SAGPooling, global_max_pool, InstanceNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# -----------------------------
# Gated Feature Fusion
# -----------------------------
class GatedFusion(nn.Module):
    def __init__(self, graph_dim, img_dim, fusion_dim):
        super().__init__()
        self.fc_graph = nn.Linear(graph_dim, fusion_dim)
        self.fc_img = nn.Linear(img_dim, fusion_dim)
        self.gate = nn.Linear(fusion_dim, 2)  # 2 weights for graph & img

    def forward(self, graph_feat, img_feat):
        g = self.fc_graph(graph_feat)
        i = self.fc_img(img_feat)
        combined = g + i

        weights = torch.sigmoid(self.gate(combined))
        g_weighted = g * weights[:, 0].unsqueeze(1)
        i_weighted = i * weights[:, 1].unsqueeze(1)

        fused = g_weighted + i_weighted
        return fused

# -----------------------------
# Main Model
# -----------------------------
class TransGated(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
        super(TransGated, self).__init__()

        self.num_heads = 4
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # -----------------------
        # Object graph features
        # -----------------------
        self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
        self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
        self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
        self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

        # -----------------------
        # Spatial and temporal graph transformers
        # -----------------------
        self.gc1_spatial = TransformerConv(
            in_channels=embedding_dim * 2 + embedding_dim // 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads,
            edge_dim=1,
            beta=True
        )
        self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

        self.gc1_temporal = TransformerConv(
            in_channels=embedding_dim * 2 + embedding_dim // 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads,
            edge_dim=1,
            beta=True
        )
        self.gc1_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

        # Graph pooling
        self.pool = SAGPooling(embedding_dim * self.num_heads, ratio=0.8)

        # -----------------------
        # I3D features -> Transformer
        # -----------------------
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)
        encoder_layer_img = TransformerEncoderLayer(
            d_model=embedding_dim * 2, nhead=self.num_heads, batch_first=True
        )
        self.temporal_transformer_img = TransformerEncoder(encoder_layer_img, num_layers=2)

        # -----------------------
        # LSTM for graph pooled features
        # -----------------------
        self.temporal_lstm_graph = nn.LSTM(
            input_size=embedding_dim * self.num_heads,
            hidden_size=embedding_dim * self.num_heads,
            num_layers=1,
            batch_first=True
        )

        # -----------------------
        # Gated fusion
        # -----------------------
        concat_dim = embedding_dim * self.num_heads + embedding_dim * 2
        self.gated_fusion = GatedFusion(graph_dim=embedding_dim * self.num_heads,
                                        img_dim=embedding_dim * 2,
                                        fusion_dim=concat_dim)

        # -----------------------
        # Classification
        # -----------------------
        self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
        self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    # -----------------------
    # Forward
    # -----------------------
    def forward(self, x, edge_index, img_feat, video_adj_list, att_feat,
                edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):

        # -----------------------
        # Helper function
        # -----------------------
        def sanitize(tensor, name):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                valid_mask = torch.isfinite(tensor)
                if valid_mask.any():
                    finite_min = tensor[valid_mask].min().item()
                    finite_max = tensor[valid_mask].max().item()
                    print(f"[⚠️ Sanitizing {name}] finite_min={finite_min:.4f}, finite_max={finite_max:.4f}")
                else:
                    print(f"[⚠️ Sanitizing {name}] all values are NaN or Inf!")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e3, neginf=-1e3)
            tensor = torch.clamp(tensor, -1e3, 1e3)
            return tensor

        # -----------------------
        # Object graph processing
        # -----------------------
        x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
        x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
        x = torch.cat((x_feat, x_label), 1)

        # -----------------------
        # Spatial graph
        # -----------------------
        edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x)
        n_embed_spatial = self.relu(self.gc1_norm1(
            self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)
        ))

        # -----------------------
        # Temporal graph
        # -----------------------
        edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x)
        n_embed_temporal = self.relu(self.gc1_norm2(
            self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal)
        ))

        # -----------------------
        # Concat + pooling
        # -----------------------
        n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
        n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
        g_embed = global_max_pool(n_embed, batch_vec)

        # -----------------------
        # LSTM over graph pooled features
        # -----------------------
        g_embed_seq = g_embed.unsqueeze(1)  # seq_len=1
        g_embed_seq, _ = self.temporal_lstm_graph(g_embed_seq)
        lstm_out_graph = g_embed_seq.squeeze(1)

        # -----------------------
        # I3D feature processing (Transformer only)
        # -----------------------
        img_feat_proj = self.img_fc(img_feat)
        img_feat_proj = sanitize(img_feat_proj, "img_feat_proj")
        lstm_out_img = self.temporal_transformer_img(img_feat_proj, is_causal=True)
        lstm_out_img = sanitize(lstm_out_img, "img_feat_trans")

        # -----------------------
        # Feature fusion
        # -----------------------
        fused_feat = self.gated_fusion(lstm_out_graph, lstm_out_img)
        fused_feat = sanitize(fused_feat, "fused_feat before classification")

        # -----------------------
        # Classification
        # -----------------------
        fused_feat = self.relu(self.classify_fc1(fused_feat))
        logits_mc = self.classify_fc2(fused_feat)
        probs_mc = self.softmax(logits_mc)

        return logits_mc, probs_mc


from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class Trans_Encode_Deode(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2, num_heads=4):
        super(Trans_Encode_Deode, self).__init__()

        self.num_heads = num_heads
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # -----------------------
        # Object graph features
        # -----------------------
        self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
        self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
        self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
        self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

        # -----------------------
        # Graph Transformers
        # -----------------------
        self.gc1_spatial = TransformerConv(
            in_channels=embedding_dim * 2 + embedding_dim // 2,
            out_channels=embedding_dim // 2,
            heads=num_heads,
            edge_dim=1,
            beta=True
        )
        self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * num_heads)

        self.gc1_temporal = TransformerConv(
            in_channels=embedding_dim * 2 + embedding_dim // 2,
            out_channels=embedding_dim // 2,
            heads=num_heads,
            edge_dim=1,
            beta=True
        )
        self.gc1_norm2 = InstanceNorm(embedding_dim // 2 * num_heads)

        self.pool = SAGPooling(embedding_dim * num_heads, ratio=0.8)

        # -----------------------
        # I3D features -> Transformer Encoder + Decoder
        # -----------------------
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

        encoder_layer_img = TransformerEncoderLayer(
            d_model=embedding_dim * 2, nhead=num_heads, batch_first=True
        )
        self.img_encoder = TransformerEncoder(encoder_layer_img, num_layers=2)

        decoder_layer_img = TransformerDecoderLayer(
            d_model=embedding_dim * 2, nhead=num_heads, batch_first=True
        )
        self.img_decoder = TransformerDecoder(decoder_layer_img, num_layers=2)

        # Learnable query for decoder
        self.decoder_query = nn.Parameter(torch.randn(1, 1, embedding_dim * 2))

        # -----------------------
        # LSTM for graph features
        # -----------------------
        self.temporal_lstm_graph = nn.LSTM(
            input_size=embedding_dim * num_heads,
            hidden_size=embedding_dim * num_heads,
            num_layers=1,
            batch_first=True
        )

        # -----------------------
        # Classification
        # -----------------------
        concat_dim = embedding_dim * num_heads + embedding_dim * 2
        self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
        self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, video_adj_list, att_feat,
                edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):

        # -----------------------
        # Helper sanitization
        # -----------------------
        def sanitize(tensor, name):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                valid_mask = torch.isfinite(tensor)
                if valid_mask.any():
                    finite_min = tensor[valid_mask].min().item()
                    finite_max = tensor[valid_mask].max().item()
                    print(f"[⚠️ Sanitizing {name}] finite_min={finite_min:.4f}, finite_max={finite_max:.4f}")
                else:
                    print(f"[⚠️ Sanitizing {name}] all values are NaN or Inf!")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e3, neginf=-1e3)
            tensor = torch.clamp(tensor, -1e3, 1e3)
            return tensor

        # -----------------------
        # Object graph processing
        # -----------------------
        x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
        x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
        x = torch.cat((x_feat, x_label), 1)

        # -----------------------
        # Spatial graph
        # -----------------------
        edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x)
        n_embed_spatial = self.relu(self.gc1_norm1(
            self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)
        ))

        # -----------------------
        # Temporal graph
        # -----------------------
        edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x)
        n_embed_temporal = self.relu(self.gc1_norm2(
            self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal)
        ))

        # -----------------------
        # Concat + pooling
        # -----------------------
        n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
        n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
        g_embed = global_max_pool(n_embed, batch_vec)

        # -----------------------
        # LSTM over graph pooled features
        # -----------------------
        g_embed_seq = g_embed.unsqueeze(0)
        g_embed_seq, _ = self.temporal_lstm_graph(g_embed_seq)
        graph_out = g_embed_seq.squeeze(0)

        # -----------------------
        # I3D feature processing: Encoder + Decoder
        # -----------------------
        img_feat_proj = self.img_fc(img_feat)
        img_feat_proj = sanitize(img_feat_proj, "img_feat_proj")

        img_feat_proj = img_feat_proj.unsqueeze(1)

        encoder_out = self.img_encoder(img_feat_proj, is_causal=True)
        batch_size = encoder_out.size(0)
        # Repeat decoder query for batch
        decoder_query = self.decoder_query.repeat(batch_size, 1, 1)
        decoder_out = self.img_decoder(tgt=decoder_query, memory=encoder_out)
        img_out = decoder_out.squeeze(1)
        img_out = sanitize(img_out, "img_out")

        # -----------------------
        # Concatenate graph + image features
        # -----------------------
        fused_feat = torch.cat((graph_out, img_out), dim=1)
        fused_feat = sanitize(fused_feat, "fused_feat before classification")

        # -----------------------
        # Classification
        # -----------------------
        fused_feat = self.relu(self.classify_fc1(fused_feat))
        logits_mc = self.classify_fc2(fused_feat)
        probs_mc = self.softmax(logits_mc)

        return logits_mc, probs_mc


# 
class Trans_LSTM_Sans_Img_LSTM(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
        super(Trans_LSTM_Sans_Img_LSTM, self).__init__()

        self.num_heads = 4
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # -----------------------
        # Object graph features
        # -----------------------
        self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
        self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
        self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
        self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

        # -----------------------
        # Spatial and temporal graph transformers
        # -----------------------
        self.gc1_spatial = TransformerConv(
            in_channels=embedding_dim * 2 + embedding_dim // 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads,
            edge_dim=1,
            beta=True
        )
        self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

        self.gc1_temporal = TransformerConv(
            in_channels=embedding_dim * 2 + embedding_dim // 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads,
            edge_dim=1,
            beta=True
        )
        self.gc1_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

        # Graph pooling
        self.pool = SAGPooling(embedding_dim * self.num_heads, ratio=0.8)

        # -----------------------
        # I3D features -> Transformer
        # -----------------------
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)
        encoder_layer_img = TransformerEncoderLayer(
            d_model=embedding_dim * 2, nhead=self.num_heads, batch_first=True
        )
        self.temporal_transformer_img = TransformerEncoder(encoder_layer_img, num_layers=2)

        # -----------------------
        # LSTM over graph features only
        # -----------------------
        self.temporal_lstm_graph = nn.LSTM(
            input_size=embedding_dim * self.num_heads,
            hidden_size=embedding_dim * self.num_heads,
            num_layers=1,
            batch_first=True
        )

        # -----------------------
        # Classification
        # -----------------------
        concat_dim = embedding_dim * self.num_heads + embedding_dim * 2  # graph + img transformer
        self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
        self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, video_adj_list,                   # att_feat,
                edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):

        # -----------------------
        # Helper function
        # -----------------------
        def sanitize(tensor, name):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                valid_mask = torch.isfinite(tensor)
                if valid_mask.any():
                    finite_min = tensor[valid_mask].min().item()
                    finite_max = tensor[valid_mask].max().item()
                    print(f"[⚠️ Sanitizing {name}] finite_min={finite_min:.4f}, finite_max={finite_max:.4f}")
                else:
                    print(f"[⚠️ Sanitizing {name}] all values are NaN or Inf!")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e3, neginf=-1e3)
            tensor = torch.clamp(tensor, -1e3, 1e3)
            return tensor

        # -----------------------
        # Object graph processing
        # -----------------------
        x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
        x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
        x = torch.cat((x_feat, x_label), 1)

        # -----------------------
        # Spatial graph
        # -----------------------
        edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x)
        n_embed_spatial = self.relu(self.gc1_norm1(
            self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)
        ))

        # -----------------------
        # Temporal graph
        # -----------------------
        edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x)
        n_embed_temporal = self.relu(self.gc1_norm2(
            self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal)
        ))

        # -----------------------
        # Concat + pooling
        # -----------------------
        n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
        n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
        g_embed = global_max_pool(n_embed, batch_vec)

        # -----------------------
        # LSTM over graph pooled features
        # -----------------------
        g_embed_seq = g_embed.unsqueeze(0)
        g_embed_seq, _ = self.temporal_lstm_graph(g_embed_seq)
        lstm_out_graph = g_embed_seq.squeeze(0)

        # -----------------------
        # I3D feature processing (direct concat, no LSTM, no pooling)
        # -----------------------
        img_feat_proj = self.img_fc(img_feat)
        img_feat_proj = sanitize(img_feat_proj, "img_feat_proj")

        # img_feat_trans = img_feat_proj.unsqueeze(0)
        # img_feat_trans = self.temporal_transformer_img(img_feat_trans, is_causal=True)
        img_feat_trans = self.temporal_transformer_img(img_feat_proj, is_causal=True)
        img_feat_trans = sanitize(img_feat_trans, "img_feat_trans")

        # Directly use transformer output (squeeze batch dimension if needed)
        # img_feat_trans = img_feat_trans.squeeze(0)
        # img_feat_trans = sanitize(img_feat_trans, "img_feat_final")

        # -----------------------
        # Concatenate graph and image features
        # -----------------------
        fused_feat = torch.cat((lstm_out_graph, img_feat_trans), dim=1)
        fused_feat = sanitize(fused_feat, "fused_feat before classification")

        # -----------------------
        # Classification
        # -----------------------
        fused_feat = self.relu(self.classify_fc1(fused_feat))
        logits_mc = self.classify_fc2(fused_feat)
        probs_mc = self.softmax(logits_mc)

        return logits_mc, probs_mc


class LSTM_Only(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
        super(LSTM_Only, self).__init__()

        self.num_heads = 4
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # -----------------------
        # Object graph features
        # -----------------------
        self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
        self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
        self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
        self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

        # -----------------------
        # Spatial and temporal graph transformers
        # -----------------------
        self.gc1_spatial = TransformerConv(
            in_channels=embedding_dim * 2 + embedding_dim // 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads,
            edge_dim=1,
            beta=True
        )
        self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

        self.gc1_temporal = TransformerConv(
            in_channels=embedding_dim * 2 + embedding_dim // 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads,
            edge_dim=1,
            beta=True
        )
        self.gc1_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

        # Graph pooling
        self.pool = SAGPooling(embedding_dim * self.num_heads, ratio=0.8)

        # -----------------------
        # I3D features -> LSTM only
        # -----------------------
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)
        self.temporal_lstm_img = nn.LSTM(
            input_size=embedding_dim * 2,
            hidden_size=embedding_dim * 2,
            num_layers=1,
            batch_first=True
        )

        # -----------------------
        # LSTM over graph pooled features
        # -----------------------
        self.temporal_lstm_graph = nn.LSTM(
            input_size=embedding_dim * self.num_heads,
            hidden_size=embedding_dim * self.num_heads,
            num_layers=1,
            batch_first=True
        )

        # -----------------------
        # Classification
        # -----------------------
        concat_dim = embedding_dim * self.num_heads + embedding_dim * 2
        self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
        self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, video_adj_list,                 # att_feat,
                edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):

        # -----------------------
        # Helper function
        # -----------------------
        def sanitize(tensor, name):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                valid_mask = torch.isfinite(tensor)
                if valid_mask.any():
                    finite_min = tensor[valid_mask].min().item()
                    finite_max = tensor[valid_mask].max().item()
                    print(f"[⚠️ Sanitizing {name}] finite_min={finite_min:.4f}, finite_max={finite_max:.4f}")
                else:
                    print(f"[⚠️ Sanitizing {name}] all values are NaN or Inf!")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e3, neginf=-1e3)
            tensor = torch.clamp(tensor, -1e3, 1e3)
            return tensor

        # -----------------------
        # Object graph processing
        # -----------------------
        x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
        x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
        x = torch.cat((x_feat, x_label), 1)

        # -----------------------
        # Spatial graph
        # -----------------------
        edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x)
        n_embed_spatial = self.relu(self.gc1_norm1(
            self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)
        ))

        # -----------------------
        # Temporal graph
        # -----------------------
        edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x)
        n_embed_temporal = self.relu(self.gc1_norm2(
            self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal)
        ))

        # -----------------------
        # Concat + pooling
        # -----------------------
        n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
        n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
        g_embed = global_max_pool(n_embed, batch_vec)

        # -----------------------
        # LSTM over graph pooled features
        # -----------------------
        g_embed_seq = g_embed.unsqueeze(0)
        g_embed_seq, _ = self.temporal_lstm_graph(g_embed_seq)
        lstm_out_graph = g_embed_seq.squeeze(0)

        # -----------------------
        # I3D feature processing -> LSTM only
        # -----------------------
        img_feat_proj = self.img_fc(img_feat)
        img_feat_proj = sanitize(img_feat_proj, "img_feat_proj")

        img_feat_seq = img_feat_proj.unsqueeze(0)  # seq_len=1 if necessary
        img_feat_seq, _ = self.temporal_lstm_img(img_feat_seq)
        lstm_out_img = img_feat_seq.squeeze(0)
        lstm_out_img = sanitize(lstm_out_img, "lstm_out_img")

        # -----------------------
        # Concatenate all LSTM outputs
        # -----------------------
        fused_feat = torch.cat((lstm_out_graph, lstm_out_img), dim=1)
        fused_feat = sanitize(fused_feat, "fused_feat before classification")

        # -----------------------
        # Classification
        # -----------------------
        fused_feat = self.relu(self.classify_fc1(fused_feat))
        logits_mc = self.classify_fc2(fused_feat)
        probs_mc = self.softmax(logits_mc)

        return logits_mc, probs_mc


import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, InstanceNorm, SAGPooling, global_max_pool

class LSTM_Trans(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
        super(LSTM_Trans, self).__init__()

        self.num_heads = 4
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # -----------------------
        # Object graph features
        # -----------------------
        self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
        self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
        self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
        self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

        # -----------------------
        # Spatial and temporal graph transformers
        # -----------------------
        self.gc1_spatial = TransformerConv(
            in_channels=embedding_dim * 2 + embedding_dim // 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads,
            edge_dim=1,
            beta=True
        )
        self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

        self.gc1_temporal = TransformerConv(
            in_channels=embedding_dim * 2 + embedding_dim // 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads,
            edge_dim=1,
            beta=True
        )
        self.gc1_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

        # Graph pooling
        self.pool = SAGPooling(embedding_dim * self.num_heads, ratio=0.8)

        # -----------------------
        # I3D features -> LSTM -> Transformer
        # -----------------------
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

        # LSTM first
        self.temporal_lstm_img = nn.LSTM(
            input_size=embedding_dim * 2,
            hidden_size=embedding_dim * 2,
            num_layers=1,
            batch_first=True
        )

        # Transformer after LSTM
        encoder_layer_img = nn.TransformerEncoderLayer(
            d_model=embedding_dim * 2, nhead=self.num_heads, batch_first=True
        )
        self.temporal_transformer_img = nn.TransformerEncoder(encoder_layer_img, num_layers=2)

        # -----------------------
        # Graph LSTM
        # -----------------------
        self.temporal_lstm_graph = nn.LSTM(
            input_size=embedding_dim * self.num_heads,
            hidden_size=embedding_dim * self.num_heads,
            num_layers=1,
            batch_first=True
        )

        # -----------------------
        # Classification
        # -----------------------
        concat_dim = embedding_dim * self.num_heads + embedding_dim * 2
        self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
        self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, video_adj_list,
                edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):

        # -----------------------
        # Helper function
        # -----------------------
        def sanitize(tensor, name):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                valid_mask = torch.isfinite(tensor)
                if valid_mask.any():
                    finite_min = tensor[valid_mask].min().item()
                    finite_max = tensor[valid_mask].max().item()
                    print(f"[⚠️ Sanitizing {name}] finite_min={finite_min:.4f}, finite_max={finite_max:.4f}")
                else:
                    print(f"[⚠️ Sanitizing {name}] all values are NaN or Inf!")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e3, neginf=-1e3)
            tensor = torch.clamp(tensor, -1e3, 1e3)
            return tensor

        # -----------------------
        # Object graph processing
        # -----------------------
        x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
        x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
        x = torch.cat((x_feat, x_label), 1)

        # -----------------------
        # Spatial graph
        # -----------------------
        edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x)
        n_embed_spatial = self.relu(self.gc1_norm1(
            self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)
        ))

        # -----------------------
        # Temporal graph
        # -----------------------
        edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x)
        n_embed_temporal = self.relu(self.gc1_norm2(
            self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal)
        ))

        # -----------------------
        # Concat + pooling
        # -----------------------
        n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
        n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
        g_embed = global_max_pool(n_embed, batch_vec)

        # -----------------------
        # LSTM over graph pooled features
        # -----------------------
        g_embed_seq = g_embed.unsqueeze(0)
        g_embed_seq, _ = self.temporal_lstm_graph(g_embed_seq)
        lstm_out_graph = g_embed_seq.squeeze(0)

        # -----------------------
        # I3D feature processing (LSTM → Transformer)
        # -----------------------
        img_feat_proj = self.img_fc(img_feat)
        img_feat_proj = sanitize(img_feat_proj, "img_feat_proj")

        # Apply LSTM first
        img_feat_seq = img_feat_proj.unsqueeze(0)
        img_feat_seq, _ = self.temporal_lstm_img(img_feat_seq)
        lstm_out_img = img_feat_seq.squeeze(0)
        lstm_out_img = sanitize(lstm_out_img, "lstm_out_img")

        # Then Transformer
        # print("lstm_out_img shape: ", lstm_out_img.shape)
        img_feat_trans = self.temporal_transformer_img(lstm_out_img, is_causal=True)
        img_feat_trans = sanitize(img_feat_trans, "img_feat_trans")

        # -----------------------
        # Concatenate all LSTM outputs
        # -----------------------
        fused_feat = torch.cat((lstm_out_graph, img_feat_trans), dim=1)
        fused_feat = sanitize(fused_feat, "fused_feat before classification")

        # -----------------------
        # Classification
        # -----------------------
        fused_feat = self.relu(self.classify_fc1(fused_feat))
        logits_mc = self.classify_fc2(fused_feat)
        probs_mc = self.softmax(logits_mc)

        return logits_mc, probs_mc


# Graph(Graph)
class SpaceTempGoG_detr_dota(nn.Module):

	def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
		super(SpaceTempGoG_detr_dota, self).__init__()

		self.num_heads = 1
		self.input_dim = input_dim

		# process the object graph features
		self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
		self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
		self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
		self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

		# GNN for encoding the object-level graph
		self.gc1_spatial = GCNConv(embedding_dim * 2 + embedding_dim // 2, embedding_dim // 2)
		self.gc1_norm1 = InstanceNorm(embedding_dim // 2)
		self.gc1_temporal = GCNConv(embedding_dim * 2 + embedding_dim // 2, embedding_dim // 2)
		self.gc1_norm2 = InstanceNorm(embedding_dim // 2)
		self.pool = TopKPooling(embedding_dim, ratio=0.8)

		# I3D features
		self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

		self.gc2_sg = GATv2Conv(embedding_dim, embedding_dim // 2, heads=self.num_heads)  # +
		self.gc2_norm1 = InstanceNorm((embedding_dim // 2) * self.num_heads)
		self.gc2_i3d = GATv2Conv(embedding_dim * 2, embedding_dim // 2, heads=self.num_heads)
		self.gc2_norm2 = InstanceNorm((embedding_dim // 2) * self.num_heads)

		self.classify_fc1 = nn.Linear(embedding_dim, embedding_dim // 2)
		self.classify_fc2 = nn.Linear(embedding_dim // 2, num_classes)

		self.relu = nn.LeakyReLU(0.2)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, temporal_edge_w,
				batch_vec):
		"""
		Inputs:
		x - object-level graph nodes' feature matrix
		edge_index - spatial graph connectivity for object-level graph
		img_feat - frame I3D features
		video_adj_list - Graph connectivity for frame-level graph
		edge_embeddings - Edge features for the object-level graph
		temporal_adj_list - temporal graph connectivity for object-level graph
		temporal_wdge_w - edge weights for frame-level graph
		batch_vec - vector for graph pooling the object-level graph

		Returns:
		logits_mc - Final logits
		probs_mc - Final probabilities
		"""

		# process object graph features
		x_feat = self.x_fc(x[:, :self.input_dim])
		x_feat = self.relu(self.x_bn1(x_feat))
		x_label = self.obj_l_fc(x[:, self.input_dim:])
		x_label = self.relu(self.obj_l_bn1(x_label))
		x = torch.cat((x_feat, x_label), 1)

		# Get graph embedding for ibject-level graph
		n_embed_spatial = self.relu(self.gc1_norm1(self.gc1_spatial(x, edge_index, edge_weight=edge_embeddings[:, -1])))
		n_embed_temporal = self.relu(self.gc1_norm2(self.gc1_temporal(x, temporal_adj_list, temporal_edge_w)))
		n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
		n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
		g_embed = global_max_pool(n_embed, batch_vec)

		# Process I3D feature
		img_feat = self.img_fc(img_feat)

		# Get frame embedding for all nodes in frame-level graph
		frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
		frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat, video_adj_list)))
		frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img), 1)
		frame_embed_sg = self.relu(self.classify_fc1(frame_embed_))
		logits_mc = self.classify_fc2(frame_embed_sg)
		probs_mc = self.softmax(logits_mc)

		return logits_mc, probs_mc


# STAGNet - DoTA DADA
# from torch_geometric.nn import (
#     GATv2Conv, 
#     TopKPooling,
#     SAGPooling,
#     global_max_pool, 
#     global_mean_pool,
#     InstanceNorm
# )
# from torch.nn import Sequential, Linear, BatchNorm1d, LeakyReLU, Dropout, GRU, MultiheadAttention

# class SpaceTempGoG_detr_dota(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
#         super(SpaceTempGoG_detr_dota, self).__init__()

#         self.num_heads = 1
#         self.input_dim = input_dim
#         self.embedding_dim = embedding_dim

#         # process the object graph features
#         self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
#         self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
#         self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
#         self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)
        
#         # Improved GNN for encoding the object-level graph
#         # self.gc1_spatial = GATv2Conv(
#         #     embedding_dim * 2 + embedding_dim // 2, 
#         #     embedding_dim // 2, 
#         #     heads=self.num_heads,
#         #     edge_dim=1  # Using temporal_edge_w as edge features
#         # )
#         # GNN for encoding the object-level graph
#         self.gc1_spatial = GCNConv(embedding_dim * 2 + embedding_dim // 2, embedding_dim // 2)
#         self.gc1_norm1 = InstanceNorm(embedding_dim // 2)
        
#         # Improved temporal graph convolution
#         self.gc1_temporal = GATv2Conv(
#             embedding_dim * 2 + embedding_dim // 2, 
#             embedding_dim // 2, 
#             heads=self.num_heads,
#             edge_dim=1  # Using temporal_edge_w as edge features
#         )
#         # self.gc1_temporal = GCNConv(embedding_dim * 2 + embedding_dim // 2, embedding_dim // 2)
#         self.gc1_norm2 = InstanceNorm(embedding_dim // 2)  # Removed *num_heads since we're using 1 head
        
#         # self.pool = TopKPooling(embedding_dim, ratio=0.8)
#         self.pool = SAGPooling(embedding_dim, ratio=0.8)

#         # I3D features with temporal processing
#         self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)
        
#         # # Added GRU for temporal sequence processing
#         # self.temporal_gru = nn.GRU(
#         #     input_size=embedding_dim * 2,
#         #     hidden_size=embedding_dim * 2,  # Changed to match input size
#         #     num_layers=1,
#         #     batch_first=True
#         # )

#         # Added LSTM for temporal sequence processing
#         self.temporal_lstm = nn.LSTM(
#             input_size=embedding_dim * 2,
#             hidden_size=embedding_dim * 2,  # Changed to match input size
#             num_layers=1,
#             batch_first=True
#         )

#         # Fixed dimension mismatches in these layers
#         self.gc2_sg = GATv2Conv(
#             embedding_dim,  # Input from g_embed
#             embedding_dim // 2, 
#             heads=self.num_heads
#         )
#         self.gc2_norm1 = InstanceNorm(embedding_dim // 2)
        
#         self.gc2_i3d = GATv2Conv(
#             embedding_dim * 2,  # Input from GRU output
#             embedding_dim // 2, 
#             heads=self.num_heads
#         )
#         self.gc2_norm2 = InstanceNorm(embedding_dim // 2)

#         self.classify_fc1 = nn.Linear(embedding_dim, embedding_dim // 2)
#         self.classify_fc2 = nn.Linear(embedding_dim // 2, num_classes)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):
#         # process object graph features
#         x_feat = self.x_fc(x[:, :self.input_dim])
#         x_feat = self.relu(self.x_bn1(x_feat))
#         x_label = self.obj_l_fc(x[:, self.input_dim:])
#         x_label = self.relu(self.obj_l_bn1(x_label))
#         x = torch.cat((x_feat, x_label), 1)

#         # Old Get graph embedding for object-level graph
#         n_embed_spatial = self.relu(self.gc1_norm1(self.gc1_spatial(x, edge_index, edge_weight=edge_embeddings[:, -1])))
        
#         # Improved Get graph embedding for object-level graph
#         # n_embed_spatial = self.relu(self.gc1_norm1(
#         #     self.gc1_spatial(x, edge_index, edge_attr=edge_embeddings[:, -1].unsqueeze(1))
#         # ))
        
#         # Old temporal processing
#         # n_embed_temporal = self.relu(self.gc1_norm2(self.gc1_temporal(x, temporal_adj_list, temporal_edge_w)))
        
#         # Improved temporal processing
#         n_embed_temporal = self.relu(self.gc1_norm2(
#             self.gc1_temporal(x, temporal_adj_list, edge_attr=temporal_edge_w.unsqueeze(1))
#         ))
        
#         n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
#         n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
#         g_embed = global_max_pool(n_embed, batch_vec)

#         # Process I3D feature with temporal modeling
#         img_feat = self.img_fc(img_feat)
#         # print("After img_fc:", img_feat.shape)
        
#         # GRU processing - reshape for temporal dimension
#         # img_feat = img_feat.unsqueeze(0)  # Add sequence dimension (1, num_nodes, features)
#         # img_feat, _ = self.temporal_gru(img_feat)
#         # img_feat = img_feat.squeeze(0)  # Back to (num_nodes, features)

# 		# LSTM processing - reshape for temporal dimension
#         img_feat = img_feat.unsqueeze(0)  # Add sequence dimension (1, num_nodes, features)
#         img_feat, (_, _) = self.temporal_lstm(img_feat)  # Extract only output, discard hidden and cell state
#         img_feat = img_feat.squeeze(0)  # Back to (num_nodes, features)

#         # Get frame embedding for all nodes in frame-level graph
#         frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
#         frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat, video_adj_list)))
#         frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img), 1)
#         frame_embed_sg = self.relu(self.classify_fc1(frame_embed_))
#         logits_mc = self.classify_fc2(frame_embed_sg)
#         probs_mc = self.softmax(logits_mc)

#         return logits_mc, probs_mc
