import torch, torch.nn as nn
from transformers import CLIPModel
class VideoNet(nn.Module):
    def __init__(self, nc=400):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        h = self.clip.config.projection_dim
        self.temporal = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=h, nhead=8, batch_first=True), num_layers=4)
        self.head = nn.Linear(h, nc)
    def forward(self, frames):
        B,T,C,H,W = frames.shape
        vis = self.clip.get_image_features(frames.view(B*T,C,H,W)).view(B,T,-1)
        return self.head(self.temporal(vis).mean(1))
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = VideoNet().to(device)
    print(f"VideoNet on {torch.cuda.get_device_name(0)}, {sum(p.numel() for p in m.parameters())/1e9:.2f}B params")
