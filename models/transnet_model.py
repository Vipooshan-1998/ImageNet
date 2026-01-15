# Trans then multi then combine three output
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, InstanceNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention

class Img_Trans_Multi_Graph_Net(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
        super(Img_Trans_Multi_Graph_Net, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = 4

        # -----------------------
        # Image feature projection
        # -----------------------
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

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
        concat_dim = (embedding_dim // 2 * self.num_heads) * 2 + embedding_dim * 2
        self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
        self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings=None,
                temporal_adj_list=None, temporal_edge_w=None, batch_vec=None):
        """
        img_feat: (seq_len, img_feat_dim)
        video_adj_list: graph edges for TransformerConv
        """
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
        # Image feature projection
        # -----------------------
        img_feat_proj = self.img_fc(img_feat).unsqueeze(0)  # (1, seq_len, d_model)

        # -----------------------
        # Single causal Transformer
        # -----------------------
        img_feat_trans = self.temporal_transformer(img_feat_proj, is_causal=True)
        img_feat_trans = sanitize(img_feat_trans, "img_feat_trans")
	
        img_feat_trans = self.fusion_fc(img_feat_trans.squeeze(0))  # fusion after transformer
        img_feat_trans = sanitize(img_feat_trans, "img_feat_trans fc")

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
        img_feat_attn = sanitize(img_feat_attn, "img_feat_attn")
					
        # -----------------------
        # Graph TransformerConv
        # -----------------------
        frame_embed_orig = self.relu(self.norm_orig(self.gc_orig(img_feat_trans, video_adj_list)))
        frame_embed_orig = sanitize(frame_embed_orig, "frame_embed_orig")
					
        frame_embed_attn = self.relu(self.norm_attn(self.gc_attn(img_feat_attn, video_adj_list)))
        frame_embed_attn = sanitize(frame_embed_attn, "frame_embed_attn")

        # -----------------------
        # Concatenate all features
        # -----------------------
        frame_embed_ = torch.cat((frame_embed_orig, frame_embed_attn, img_feat_trans), dim=1)

        # -----------------------
        # Classification
        # -----------------------
        frame_embed_ = self.relu(self.classify_fc1(frame_embed_))
        logits = self.classify_fc2(frame_embed_)
        probs = self.softmax(logits)

        return logits, probs

# Ablation 1 - Keeping only the Transformer
class Img_Trans_Net(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
        super(Img_Trans_Net, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = 4

        # -----------------------
        # Image feature projection
        # -----------------------
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

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
        # Classification
        # -----------------------
        # concat_dim = transformer output dimension
        concat_dim = embedding_dim * 2
        self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
        self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings=None,
                temporal_adj_list=None, temporal_edge_w=None, batch_vec=None):
        """
        img_feat: (seq_len, img_feat_dim)
        video_adj_list: graph edges for TransformerConv
        """
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
        # Image feature projection
        # -----------------------
        img_feat_proj = self.img_fc(img_feat).unsqueeze(0)  # (1, seq_len, d_model)

        # -----------------------
        # Single causal Transformer
        # -----------------------
        img_feat_trans = self.temporal_transformer(img_feat_proj, is_causal=True)
        img_feat_trans = sanitize(img_feat_trans, "img_feat_trans")

        img_feat_trans = self.fusion_fc(img_feat_trans.squeeze(0))  # fusion after transformer
        img_feat_trans = sanitize(img_feat_trans, "img_feat_trans fc")    

        # -----------------------
        # Classification
        # -----------------------
        frame_embed_ = self.relu(self.classify_fc1(img_feat_trans))
        logits = self.classify_fc2(frame_embed_)
        probs = self.softmax(logits)

        return logits, probs


class Img_Trans_Net_sans_fc2(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
        super(Img_Trans_Net_sans_fc2, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = 4

        # -----------------------
        # Image feature projection
        # -----------------------
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

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
        # Classification
        # -----------------------
        # concat_dim = transformer output dimension
        concat_dim = embedding_dim * 2
        self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
        self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings=None,
                temporal_adj_list=None, temporal_edge_w=None, batch_vec=None):
        """
        img_feat: (seq_len, img_feat_dim)
        video_adj_list: graph edges for TransformerConv
        """
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
        # Image feature projection
        # -----------------------
        img_feat_proj = self.img_fc(img_feat).unsqueeze(0)  # (1, seq_len, d_model)

        # -----------------------
        # Single causal Transformer
        # -----------------------
        img_feat_trans = self.temporal_transformer(img_feat_proj, is_causal=True)
        img_feat_trans = sanitize(img_feat_trans, "img_feat_trans")
        img_feat_trans = img_feat_trans.squeeze(0)
        # print("img_feat_trans.shape: ", img_feat_trans.shape)
 
        # -----------------------
        # Classification
        # -----------------------
        frame_embed_ = self.relu(self.classify_fc1(img_feat_trans))
        logits = self.classify_fc2(frame_embed_)
        probs = self.softmax(logits)

        return logits, probs

# Image Only
class Img_Only(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
        super(Img_Only, self).__init__()

        self.embedding_dim = embedding_dim

        # -----------------------
        # Classification
        # -----------------------
        self.classify_fc1 = nn.Linear(img_feat_dim, embedding_dim)
        self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings=None,
                temporal_adj_list=None, temporal_edge_w=None, batch_vec=None):
        """
        img_feat: (seq_len, img_feat_dim)
        video_adj_list: graph edges for TransformerConv
        """
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
        # Classification
        # -----------------------
        frame_embed_ = self.relu(self.classify_fc1(img_feat))
        logits = self.classify_fc2(frame_embed_)
        probs = self.softmax(logits)

        return logits, probs

# Trans and Graph
class Img_Trans_Graph_Net(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
        super(Img_Trans_Graph_Net, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = 4

        # -----------------------
        # Image feature projection
        # -----------------------
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

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
        # Graph TransformerConv branches
        # -----------------------
        self.gc_orig = TransformerConv(
            in_channels=embedding_dim * 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads
        )

        self.norm_orig = InstanceNorm(embedding_dim // 2 * self.num_heads)

        # -----------------------
        # Classification
        # -----------------------
        concat_dim = (embedding_dim // 2 * self.num_heads)
        self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
        self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings=None,
                temporal_adj_list=None, temporal_edge_w=None, batch_vec=None):
        """
        img_feat: (seq_len, img_feat_dim)
        video_adj_list: graph edges for TransformerConv
        """
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
        # Image feature projection
        # -----------------------
        img_feat_proj = self.img_fc(img_feat).unsqueeze(0)  # (1, seq_len, d_model)

        # -----------------------
        # Single causal Transformer
        # -----------------------
        img_feat_trans = self.temporal_transformer(img_feat_proj, is_causal=True)
        img_feat_trans = sanitize(img_feat_trans, "img_feat_trans")
        img_feat_trans = img_feat_trans.squeeze(0)
					
        # -----------------------
        # Graph TransformerConv
        # -----------------------
        frame_embed_orig = self.relu(self.norm_orig(self.gc_orig(img_feat_trans, video_adj_list)))
        frame_embed_orig = sanitize(frame_embed_orig, "frame_embed_orig")

        # -----------------------
        # Classification
        # -----------------------
        frame_embed_ = self.relu(self.classify_fc1(frame_embed_orig))
        logits = self.classify_fc2(frame_embed_)
        probs = self.softmax(logits)

        return logits, probs

## Ablation 2nd case - removing green trans keeping others
class Img_Graph_Multi_Graph_Net(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
        super(Img_Graph_Multi_Graph_Net, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = 4

        # -----------------------
        # Image feature projection
        # -----------------------
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

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
        concat_dim = (embedding_dim // 2 * self.num_heads) * 1 + embedding_dim * 2
        self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
        self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings=None,
                temporal_adj_list=None, temporal_edge_w=None, batch_vec=None):
        """
        img_feat: (seq_len, img_feat_dim)
        video_adj_list: graph edges for TransformerConv
        """
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
        # Image feature projection
        # -----------------------
        img_feat = self.img_fc(img_feat) #.unsqueeze(0)  # (1, seq_len, d_model)

        # -----------------------
        # Multihead attention branch
        # -----------------------
        img_feat_attn, _ = self.img_attn(
            img_feat.unsqueeze(0),  # Q
            img_feat.unsqueeze(0),  # K
            img_feat.unsqueeze(0),  # V
            is_causal=True
        )
        img_feat_attn = img_feat_attn.squeeze(0)
        img_feat_attn = sanitize(img_feat_attn, "img_feat_attn")
					
        # -----------------------
        # Graph TransformerConv
        # -----------------------
        frame_embed_orig = self.relu(self.norm_orig(self.gc_orig(img_feat, video_adj_list)))
        frame_embed_orig = sanitize(frame_embed_orig, "frame_embed_orig")
					
        frame_embed_attn = self.relu(self.norm_attn(self.gc_attn(img_feat_attn, video_adj_list)))
        frame_embed_attn = sanitize(frame_embed_attn, "frame_embed_attn")

        # -----------------------
        # Concatenate all features
        # -----------------------
        frame_embed_ = torch.cat((frame_embed_orig, frame_embed_attn), dim=1)

        # -----------------------
        # Classification
        # -----------------------
        frame_embed_ = self.relu(self.classify_fc1(frame_embed_))
        logits = self.classify_fc2(frame_embed_)
        probs = self.softmax(logits)

        return logits, probs

## Final Arch
class Img_Trans_Graph_Trans_Net(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
        super(Img_Trans_Graph_Trans_Net, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = 4

        # -----------------------
        # Image feature projection
        # -----------------------
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

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
        # Graph TransformerConv branches
        # -----------------------
        self.gc_orig = TransformerConv(
            in_channels=embedding_dim * 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads
        )

        self.norm_orig = InstanceNorm(embedding_dim // 2 * self.num_heads)

        # -----------------------
        # Classification
        # -----------------------
        concat_dim = (embedding_dim // 2 * self.num_heads) + embedding_dim * 2
        self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
        self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings=None,
                temporal_adj_list=None, temporal_edge_w=None, batch_vec=None):
        """
        img_feat: (seq_len, img_feat_dim)
        video_adj_list: graph edges for TransformerConv
        """
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
        # Image feature projection
        # -----------------------
        img_feat_proj = self.img_fc(img_feat).unsqueeze(0)  # (1, seq_len, d_model)

        # -----------------------
        # Single causal Transformer
        # -----------------------
        img_feat_trans = self.temporal_transformer(img_feat_proj, is_causal=True)
        img_feat_trans = sanitize(img_feat_trans, "img_feat_trans")
        img_feat_trans = img_feat_trans.squeeze(0)
					
        # -----------------------
        # Graph TransformerConv
        # -----------------------
        frame_embed_orig = self.relu(self.norm_orig(self.gc_orig(img_feat_trans, video_adj_list)))
        frame_embed_orig = sanitize(frame_embed_orig, "frame_embed_orig")
					
        # -----------------------
        # Concatenate all features
        # -----------------------
        frame_embed_ = torch.cat((frame_embed_orig, img_feat_trans), dim=1)

        # -----------------------
        # Classification
        # -----------------------
        frame_embed_ = self.relu(self.classify_fc1(frame_embed_))
        logits = self.classify_fc2(frame_embed_)
        probs = self.softmax(logits)

        return logits, probs


# STAGNet - DoTA DADA Nexar
from torch_geometric.nn import (
    GATv2Conv, 
    TopKPooling,
    SAGPooling,
    global_max_pool, 
    global_mean_pool,
    InstanceNorm
)
from torch.nn import Sequential, Linear, BatchNorm1d, LeakyReLU, Dropout, GRU, MultiheadAttention

class STAGNet(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
        super(STAGNet, self).__init__()

        self.num_heads = 1
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # process the object graph features
        self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
        self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
        self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
        self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)
        
        # Improved GNN for encoding the object-level graph
        self.gc1_spatial = GATv2Conv(
            embedding_dim * 2 + embedding_dim // 2, 
            embedding_dim // 2, 
            heads=self.num_heads,
            edge_dim=1  # Using temporal_edge_w as edge features
        )
        # GNN for encoding the object-level graph
        # self.gc1_spatial = GCNConv(embedding_dim * 2 + embedding_dim // 2, embedding_dim // 2)
        self.gc1_norm1 = InstanceNorm(embedding_dim // 2)
        
        # Improved temporal graph convolution
        self.gc1_temporal = GATv2Conv(
            embedding_dim * 2 + embedding_dim // 2, 
            embedding_dim // 2, 
            heads=self.num_heads,
            edge_dim=1  # Using temporal_edge_w as edge features
        )
        # self.gc1_temporal = GCNConv(embedding_dim * 2 + embedding_dim // 2, embedding_dim // 2)
        self.gc1_norm2 = InstanceNorm(embedding_dim // 2)  # Removed *num_heads since we're using 1 head
        
        # self.pool = TopKPooling(embedding_dim, ratio=0.8)
        self.pool = SAGPooling(embedding_dim, ratio=0.8)

        # I3D features with temporal processing
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)
        
        # # Added GRU for temporal sequence processing
        # self.temporal_gru = nn.GRU(
        #     input_size=embedding_dim * 2,
        #     hidden_size=embedding_dim * 2,  # Changed to match input size
        #     num_layers=1,
        #     batch_first=True
        # )

        # Added LSTM for temporal sequence processing
        self.temporal_lstm = nn.LSTM(
            input_size=embedding_dim * 2,
            hidden_size=embedding_dim * 2,  # Changed to match input size
            num_layers=1,
            batch_first=True
        )

        # Fixed dimension mismatches in these layers
        self.gc2_sg = GATv2Conv(
            embedding_dim,  # Input from g_embed
            embedding_dim // 2, 
            heads=self.num_heads
        )
        self.gc2_norm1 = InstanceNorm(embedding_dim // 2)
        
        self.gc2_i3d = GATv2Conv(
            embedding_dim * 2,  # Input from GRU output
            embedding_dim // 2, 
            heads=self.num_heads
        )
        self.gc2_norm2 = InstanceNorm(embedding_dim // 2)

        self.classify_fc1 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.classify_fc2 = nn.Linear(embedding_dim // 2, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):
        # process object graph features
        x_feat = self.x_fc(x[:, :self.input_dim])
        x_feat = self.relu(self.x_bn1(x_feat))
        x_label = self.obj_l_fc(x[:, self.input_dim:])
        x_label = self.relu(self.obj_l_bn1(x_label))
        x = torch.cat((x_feat, x_label), 1)

        # Old Get graph embedding for object-level graph
        # n_embed_spatial = self.relu(self.gc1_norm1(self.gc1_spatial(x, edge_index, edge_weight=edge_embeddings[:, -1])))
        
        # Improved Get graph embedding for object-level graph
        n_embed_spatial = self.relu(self.gc1_norm1(
            self.gc1_spatial(x, edge_index, edge_attr=edge_embeddings[:, -1].unsqueeze(1))
        ))
        
        # Old temporal processing
        # n_embed_temporal = self.relu(self.gc1_norm2(self.gc1_temporal(x, temporal_adj_list, temporal_edge_w)))
        
        # Improved temporal processing
        n_embed_temporal = self.relu(self.gc1_norm2(
            self.gc1_temporal(x, temporal_adj_list, edge_attr=temporal_edge_w.unsqueeze(1))
        ))
        
        n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
        n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
        g_embed = global_max_pool(n_embed, batch_vec)

        # Process I3D feature with temporal modeling
        img_feat = self.img_fc(img_feat)
        # print("After img_fc:", img_feat.shape)
        
        # GRU processing - reshape for temporal dimension
        # img_feat = img_feat.unsqueeze(0)  # Add sequence dimension (1, num_nodes, features)
        # img_feat, _ = self.temporal_gru(img_feat)
        # img_feat = img_feat.squeeze(0)  # Back to (num_nodes, features)

		# LSTM processing - reshape for temporal dimension
        img_feat = img_feat.unsqueeze(0)  # Add sequence dimension (1, num_nodes, features)
        img_feat, (_, _) = self.temporal_lstm(img_feat)  # Extract only output, discard hidden and cell state
        img_feat = img_feat.squeeze(0)  # Back to (num_nodes, features)

        # Get frame embedding for all nodes in frame-level graph
        frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
        frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat, video_adj_list)))
        frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img), 1)
        frame_embed_sg = self.relu(self.classify_fc1(frame_embed_))
        logits_mc = self.classify_fc2(frame_embed_sg)
        probs_mc = self.softmax(logits_mc)

        return logits_mc, probs_mc
