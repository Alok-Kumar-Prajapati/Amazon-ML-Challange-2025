# =========================================================================================
# Inference Script - Load LightGBM Model and Make Predictions on Test Data
# =========================================================================================

import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import logging
import gc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from transformers import AutoTokenizer, AutoModel, CLIPVisionModel
from sklearn.model_selection import KFold
import lightgbm as lgb
import joblib

# ----------------------------
# Config
# ----------------------------
class Config:
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    TEXT_MODEL_NAME = "microsoft/deberta-v3-base"
    IMG_MODEL_NAME = "openai/clip-vit-base-patch32"
    MAX_LENGTH = 128
    BATCH_SIZE = 8
    NUM_FOLDS = 5
    IMG_FOLDER_TEST = '../images/test'
    CHECKPOINT_DIR = 'checkpoints_god_tier'
    NUMERIC_COLS = ['value', 'weight_in_g', 'pack_size', 'title_length', 'num_bullets', 'avg_bullet_len']
    MULTI_LABEL_COLS = ['is_food', 'is_drink', 'is_cosmetic']
    TARGET_COL = 'price'
    NUM_WORKERS = 16
    PIN_MEMORY = True

CONFIG = Config()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# ----------------------------
# Dataset Class
# ----------------------------
class MultiModalDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, img_folder, img_transform, numeric_cols, multi_label_cols, target_col):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.img_folder = img_folder
        self.img_transform = img_transform
        self.numeric_cols = numeric_cols
        self.multi_label_cols = multi_label_cols
        self.target_col = target_col
        self.is_train = (self.target_col is not None) and (self.target_col in self.df.columns)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row.get('text_input', ''))
        text_enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        img = torch.zeros((3, 224, 224), dtype=torch.float)
        if self.img_folder:
            img_path = os.path.join(self.img_folder, f"{row['sample_id']}.jpg")
            if os.path.exists(img_path):
                try:
                    pil_img = Image.open(img_path).convert('RGB')
                    if self.img_transform:
                        img = self.img_transform(pil_img)
                except Exception:
                    img = torch.zeros((3, 224, 224), dtype=torch.float)
        numeric_values = row[self.numeric_cols].values.astype(np.float32)
        numeric = torch.tensor(numeric_values, dtype=torch.float)
        item = {
            'input_ids': text_enc['input_ids'].squeeze(0),
            'attention_mask': text_enc['attention_mask'].squeeze(0),
            'image': img,
            'numeric': numeric
        }
        if self.is_train:
            item['price_target'] = torch.tensor(np.log1p(row[self.target_col]), dtype=torch.float)
            category_values = row[self.multi_label_cols].values.astype(np.float32)
            item['category_target'] = torch.tensor(category_values, dtype=torch.float)
        return item

# ----------------------------
# Model Class
# ----------------------------
class GatedFusionBlock(nn.Module):
    def __init__(self, text_dim, img_dim, num_dim, hidden_dim=512, num_heads=4, dropout=0.1):
        super().__init__()
        self.proj_text = nn.Linear(text_dim, hidden_dim)
        self.proj_img = nn.Linear(img_dim, hidden_dim)
        self.proj_num = nn.Linear(num_dim, hidden_dim)
        self.gate_layer = nn.Sequential(nn.Linear(text_dim + img_dim + num_dim, 3), nn.Softmax(dim=-1))
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True, dim_feedforward=hidden_dim*2
        )
        self.fusion_transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, txt_emb, img_emb, num_emb):
        txt_proj, img_proj, num_proj = self.proj_text(txt_emb), self.proj_img(img_emb), self.proj_num(num_emb)
        gates = self.gate_layer(torch.cat([txt_emb, img_emb, num_emb], dim=-1))
        gate_text, gate_img, gate_num = gates.chunk(3, dim=-1)
        fused_emb = (txt_proj * gate_text) + (img_proj * gate_img) + (num_proj * gate_num)
        fused_emb = fused_emb.unsqueeze(1)
        fused_output = self.fusion_transformer(fused_emb)
        return self.final_norm(fused_output.squeeze(1))

class GodTierModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.img_model = CLIPVisionModel.from_pretrained(config.IMG_MODEL_NAME)
        self.text_model.config.use_cache = False
        self.img_model.config.use_cache = False
        
        text_dim = getattr(self.text_model.config, 'hidden_size', 768)
        img_dim = getattr(self.img_model.config, 'hidden_size', 768)
        numeric_hidden = 64
        self.num_mlp = nn.Sequential(nn.Linear(len(config.NUMERIC_COLS), 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, numeric_hidden), nn.ReLU())
        self.fusion_block = GatedFusionBlock(text_dim, img_dim, numeric_hidden, hidden_dim=512)
        fusion_dim = 512
        self.regression_head = nn.Sequential(nn.Linear(fusion_dim, 256), nn.ReLU(), nn.LayerNorm(256), nn.Dropout(0.3), nn.Linear(256, 1))
        self.classification_head = nn.Sequential(nn.Linear(fusion_dim, 256), nn.ReLU(), nn.LayerNorm(256), nn.Dropout(0.3), nn.Linear(256, len(config.MULTI_LABEL_COLS)))

    def forward(self, input_ids, attention_mask, image, numeric):
        txt_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        img_out = self.img_model(pixel_values=image).pooler_output
        num_out = self.num_mlp(numeric)
        fused_features = self.fusion_block(txt_out, img_out, num_out)
        price_pred = self.regression_head(fused_features).squeeze(-1)
        category_pred = self.classification_head(fused_features)
        return price_pred, category_pred

# ----------------------------
# Prediction Function
# ----------------------------
def predict(loader, model, config):
    model.to(config.DEVICE)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            price_pred, _ = model(
                input_ids=batch['input_ids'].to(config.DEVICE),
                attention_mask=batch['attention_mask'].to(config.DEVICE),
                image=batch['image'].to(config.DEVICE),
                numeric=batch['numeric'].to(config.DEVICE)
            )
            predictions.append(price_pred.detach().cpu().numpy())
    if len(predictions) == 0:
        return np.array([])
    return np.concatenate(predictions, axis=0)

# ----------------------------
# Main Inference Function
# ----------------------------
def main():
    logger.info("\n" + "="*70)
    logger.info("🚀 STARTING INFERENCE PIPELINE")
    logger.info("="*70 + "\n")
    
    # Load test data
    logger.info("📂 Loading test data...")
    test_df = pd.read_parquet("../processed_test.parquet")
    for col in CONFIG.MULTI_LABEL_COLS:
        if col not in test_df.columns:
            test_df[col] = 0
    logger.info(f"✅ Test data loaded: {len(test_df)} samples")
    
    # Load tokenizer and transforms
    logger.info("\n🔧 Loading tokenizer and transforms...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.TEXT_MODEL_NAME)
    clip_stats = {'mean': [0.48145466, 0.4578275, 0.40821073], 'std': [0.26862954, 0.26130258, 0.27577711]}
    val_transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(**clip_stats)])
    logger.info("✅ Tokenizer and transforms loaded")
    
    # Generate test predictions from all folds
    logger.info("\n" + "="*70)
    logger.info("🧠 GENERATING NEURAL NETWORK PREDICTIONS FROM ALL FOLDS")
    logger.info("="*70)
    test_preds_list = []
    
    for fold in range(CONFIG.NUM_FOLDS):
        logger.info(f"\n📦 Processing Fold {fold}...")
        model_path = os.path.join(CONFIG.CHECKPOINT_DIR, f'fold_{fold}', 'best_model.pt')
        
        if not os.path.exists(model_path):
            logger.warning(f"⚠️  Model not found for fold {fold} at {model_path}")
            continue
        
        logger.info(f"   Loading model from: {model_path}")
        model = GodTierModel(CONFIG)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        # Create test dataset and dataloader
        test_ds = MultiModalDataset(test_df, tokenizer, CONFIG.MAX_LENGTH, CONFIG.IMG_FOLDER_TEST, val_transform, 
                                     CONFIG.NUMERIC_COLS, CONFIG.MULTI_LABEL_COLS, target_col=None)
        test_loader = DataLoader(test_ds, batch_size=CONFIG.BATCH_SIZE * 2, shuffle=False, num_workers=CONFIG.NUM_WORKERS)
        
        # Make predictions
        fold_preds = predict(test_loader, model, CONFIG)
        test_preds_list.append(fold_preds)
        logger.info(f"   ✅ Fold {fold} predictions generated: shape {fold_preds.shape}")
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Average predictions across folds
    logger.info("\n📊 Averaging predictions across all folds...")
    if len(test_preds_list) == 0:
        logger.error("❌ No predictions generated!")
        return
    
    avg_test_preds = np.mean(test_preds_list, axis=0)
    logger.info(f"✅ Averaged predictions shape: {avg_test_preds.shape}")
    
    # Load LightGBM stacking model
    logger.info("\n" + "="*70)
    logger.info("⚙️  LOADING LIGHTGBM STACKING MODEL")
    logger.info("="*70)
    lgb_model_path = os.path.join(CONFIG.CHECKPOINT_DIR, 'stacking_lightgbm_model.pkl')
    
    if not os.path.exists(lgb_model_path):
        logger.error(f"❌ LightGBM model not found at {lgb_model_path}")
        return
    
    logger.info(f"📦 Loading model from: {lgb_model_path}")
    stacking_model = joblib.load(lgb_model_path)
    logger.info("✅ LightGBM model loaded successfully!")
    
    # Prepare features for LightGBM
    logger.info("\n🔄 Preparing features for stacking model...")
    X_test = np.hstack([avg_test_preds.reshape(-1, 1), test_df[CONFIG.NUMERIC_COLS].values])
    logger.info(f"✅ Test features prepared: shape {X_test.shape}")
    
    # Make final predictions
    logger.info("\n🎯 Making final predictions with stacking model...")
    final_preds_log = stacking_model.predict(X_test)
    final_preds = np.expm1(final_preds_log)
    final_preds[final_preds < 0] = 0.01
    logger.info(f"✅ Final predictions generated: {len(final_preds)} samples")
    
    # Create submission dataframe
    logger.info("\n💾 Creating submission file...")
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'].values,
        'price': final_preds
    })
    
    # Save to CSV
    submission_path = 'test_out.csv'
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"✅ Submission saved to: {submission_path}")
    
    # Print summary statistics
    logger.info("\n" + "="*70)
    logger.info("📈 SUBMISSION SUMMARY")
    logger.info("="*70)
    logger.info(f"Total predictions: {len(submission_df)}")
    logger.info(f"Min price: {submission_df['price'].min():.6f}")
    logger.info(f"Max price: {submission_df['price'].max():.6f}")
    logger.info(f"Mean price: {submission_df['price'].mean():.6f}")
    logger.info(f"Median price: {submission_df['price'].median():.6f}")
    logger.info(f"Std price: {submission_df['price'].std():.6f}")
    logger.info("="*70)
    
    # Print first few rows
    logger.info("\n📋 First 20 predictions:")
    logger.info("\n" + submission_df.head(20).to_string(index=False))
    
    logger.info("\n" + "="*70)
    logger.info("✅ INFERENCE COMPLETED SUCCESSFULLY!")
    logger.info("="*70 + "\n")

if __name__ == "__main__":
    main()