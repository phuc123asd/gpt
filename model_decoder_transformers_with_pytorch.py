import pip
try:
  __import__("lightning")
except ImportError:
  pip.main(['install', "lightning"])  

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, IterableDataset

import lightning as L
import math

from datasets import load_dataset
from transformers import AutoTokenizer

from lightning.pytorch.callbacks import ModelCheckpoint
class PositionEncoding(nn.Module):
    def __init__(self, d_model=256, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()
        
        div_term = 1/torch.tensor(10000.0)**(embedding_index / d_model)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, word_embeddings):
        return word_embeddings + self.pe[:word_embeddings.size(1)].unsqueeze(0)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            att = att.masked_fill(mask, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)

class FeedForward(nn.Module):
    def __init__(self, d_model=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class GPT1Block(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x

class GPT1Mini(L.LightningModule):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_heads=8,
        n_layers=6,
        max_len=256,
        lr=3e-4
    ):
        super().__init__()
        self.save_hyperparameters()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionEncoding(d_model, max_len)

        self.blocks = nn.ModuleList([
            GPT1Block(d_model, n_heads)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # GPT-1 weight tying
        self.lm_head.weight = self.token_emb.weight

        # causal mask (cached)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_len, max_len)).bool()
        )

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_emb(idx)
        x = self.pos_emb(x)

        mask = self.causal_mask[:T, :T]
        mask = ~mask
        mask = mask.unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        return self.lm_head(x)

    # -------- Lightning parts --------

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
class VietVaultGPTDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer, block_size=128, max_tokens=2_000_000):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_tokens = max_tokens

    def __iter__(self):
        token_count = 0

        for sample in self.dataset:
            text = sample["markdown"].strip()
            if len(text) < 50:
                continue

            ids = self.tokenizer.encode(text)

            for i in range(0, len(ids) - self.block_size, self.block_size):
                x = ids[i:i+self.block_size]
                y = ids[i+1:i+self.block_size+1]

                yield torch.tensor(x), torch.tensor(y)

                token_count += self.block_size
                if token_count >= self.max_tokens:
                    return
def generate(model, tokenizer, prompt, max_new_tokens=80):
    model.eval()
    ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(ids)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
        ids = ids[:, -model.hparams.max_len:]

    return tokenizer.decode(ids[0])
if __name__ == "__main__":
    dataset = load_dataset(
        "nampdn-ai/vietvault",
        split="train",
        streaming=True
    )

    # for i, sample in enumerate(dataset):
    #     print(sample["markdown"])
    #     if i == 3:
    #         break
    tokenizer = AutoTokenizer.from_pretrained("NlpHUST/gpt2-vietnamese")
    tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = VietVaultGPTDataset(
        dataset,
        tokenizer,
        block_size=128,
        max_tokens=2_000_000  # TRAIN NHẸ LẦN ĐẦU
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
    )
    model = GPT1Mini(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=8,
        n_layers=6,
        max_len=256,
        lr=3e-4
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",          # thư mục lưu
        filename="gpt1mini-epoch{epoch}",# tên file
        save_top_k=-1,                  # LƯU TẤT CẢ epoch
        every_n_epochs=1,               # mỗi epoch lưu 1 lần
        save_weights_only=True          # chỉ lưu weight (nhẹ)
    )
    
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=30,
        precision="16-mixed",
        callbacks=[checkpoint_callback],
        log_every_n_steps=50,
        gradient_clip_val=1.0
    )

    trainer.fit(model, train_loader)