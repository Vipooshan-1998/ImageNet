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
