import os
import random
import math
import time
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

from transformers import AutoTokenizer, AutoModel, CLIPVisionModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import KFold
import lightgbm as lgb

# ----------------------------
# Config & Seed
# ----------------------------
class Config:
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---> NEW FEATURE: Set to True to resume training from the last checkpoint <---
    RESUME_TRAINING = True

    # Your original configuration - OPTIMIZED FOR GPU EFFICIENCY
    TEXT_MODEL_NAME = "microsoft/deberta-v3-base"
    IMG_MODEL_NAME = "openai/clip-vit-base-patch32"
    MAX_LENGTH = 128
    BATCH_SIZE = 8  # Larger batch size for better GPU utilization
    GRAD_ACCUM_STEPS = 2  # Less accumulation - process more data per step
    EPOCHS = 10
    NUM_FOLDS = 5
    EARLY_STOP_PATIENCE = 5
    IMG_FOLDER_TRAIN = '../images/train'
    IMG_FOLDER_TEST = '../images/test'
    CHECKPOINT_DIR = 'checkpoints_god_tier'
    LOG_FILE = 'training_god_tier.log'
    NUMERIC_COLS = ['value', 'weight_in_g', 'pack_size', 'title_length', 'num_bullets', 'avg_bullet_len']
    MULTI_LABEL_COLS = ['is_food', 'is_drink', 'is_cosmetic']
    TARGET_COL = 'price'
    LOSS_WEIGHT_ALPHA = 0.7
    LR = 2e-5
    WEIGHT_DECAY = 1e-2
    USE_AMP = False  # Disable mixed precision - overhead > benefit
    CLEAR_CACHE_STEPS = 50  # Clear cache less frequently
    NUM_WORKERS = 16  # Single worker - multiple workers have IPC overhead
    PIN_MEMORY = True

CONFIG = Config()

logging.basicConfig(filename=CONFIG.LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

def seed_everything(seed=CONFIG.SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything()

# ---> SMAPE Calculation Function <---
def calculate_smape(y_true, y_pred):
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / (denominator + 1e-8)) * 100

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

def get_optimizer(model, config):
    return torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

def train_one_fold(fold, train_loader, val_loader, model, optimizer, scheduler, config, start_epoch=0, best_val_loss=float('inf')):
    criterion_reg, criterion_cls = nn.SmoothL1Loss(), nn.BCEWithLogitsLoss()
    model.to(config.DEVICE)
    
    patience_counter = 0
    fold_ckpt_dir = os.path.join(config.CHECKPOINT_DIR, f'fold_{fold}')
    os.makedirs(fold_ckpt_dir, exist_ok=True)
    
    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        total_loss = 0.0
        all_train_preds_log, all_train_targets_log = [], []
        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Fold {fold} Epoch {epoch+1}")
        
        epoch_start_time = time.time()
        for step, batch in pbar:
            price_pred, cat_pred = model(
                input_ids=batch['input_ids'].to(config.DEVICE),
                attention_mask=batch['attention_mask'].to(config.DEVICE),
                image=batch['image'].to(config.DEVICE),
                numeric=batch['numeric'].to(config.DEVICE)
            )
            loss_price = criterion_reg(price_pred, batch['price_target'].to(config.DEVICE))
            loss_cat = criterion_cls(cat_pred, batch['category_target'].to(config.DEVICE))
            loss = (config.LOSS_WEIGHT_ALPHA * loss_price) + ((1 - config.LOSS_WEIGHT_ALPHA) * loss_cat)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
            
            # Collect predictions for SMAPE calculation during training
            all_train_preds_log.append(price_pred.detach().cpu().numpy())
            all_train_targets_log.append(batch['price_target'].detach().cpu().numpy())
            
            if (step + 1) % config.CLEAR_CACHE_STEPS == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            total_loss += loss.item()
            
            # Calculate running SMAPE for training
            if len(all_train_preds_log) > 0:
                train_preds_actual = np.expm1(np.concatenate(all_train_preds_log))
                train_targets_actual = np.expm1(np.concatenate(all_train_targets_log))
                train_smape = calculate_smape(train_targets_actual, train_preds_actual)
            else:
                train_smape = 0.0
            
            pbar.set_postfix(loss=total_loss / (step + 1), smape=f"{train_smape:.4f}%", lr=scheduler.get_last_lr()[0] if scheduler else config.LR)
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        model.eval()
        val_loss = 0.0
        all_val_preds_log, all_val_targets_log = [], []
        with torch.no_grad():
            for batch in val_loader:
                price_pred, cat_pred = model(
                    input_ids=batch['input_ids'].to(config.DEVICE),
                    attention_mask=batch['attention_mask'].to(config.DEVICE),
                    image=batch['image'].to(config.DEVICE),
                    numeric=batch['numeric'].to(config.DEVICE)
                )
                loss_price = criterion_reg(price_pred, batch['price_target'].to(config.DEVICE))
                loss_cat = criterion_cls(cat_pred, batch['category_target'].to(config.DEVICE))
                loss = (config.LOSS_WEIGHT_ALPHA * loss_price) + ((1 - config.LOSS_WEIGHT_ALPHA) * loss_cat)
                val_loss += loss.item()
                all_val_preds_log.append(price_pred.detach().cpu().numpy())
                all_val_targets_log.append(batch['price_target'].detach().cpu().numpy())

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_preds_actual = np.expm1(np.concatenate(all_val_preds_log))
        val_targets_actual = np.expm1(np.concatenate(all_val_targets_log))
        val_smape = calculate_smape(val_targets_actual, val_preds_actual)
        
        # Calculate final training SMAPE for epoch
        if len(all_train_preds_log) > 0:
            train_preds_final = np.expm1(np.concatenate(all_train_preds_log))
            train_targets_final = np.expm1(np.concatenate(all_train_targets_log))
            train_smape_final = calculate_smape(train_targets_final, train_preds_final)
        else:
            train_smape_final = 0.0
        
        logger.info(f"Fold {fold} | Epoch {epoch+1} | Train Loss: {total_loss / max(1, len(train_loader)):.4f} | Train SMAPE: {train_smape_final:.4f}% | Val Loss: {avg_val_loss:.4f} | Val SMAPE: {val_smape:.4f}%")

        # Save the "bookmark" checkpoint at the end of every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_loss': best_val_loss
        }, os.path.join(fold_ckpt_dir, 'last_epoch.pt'))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(fold_ckpt_dir, 'best_model.pt'))
            logger.info(f"-> Val Loss improved, saved best model for fold {fold}")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                logger.info("Early stopping triggered.")
                break

def predict(loader, model, config):
    model.to(config.DEVICE)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            autocast_device = 'cuda' if (config.DEVICE == 'cuda') else 'cpu'
            with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=config.USE_AMP):
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

def print_gpu_stats():
    """Print GPU utilization statistics"""
    if torch.cuda.is_available():
        logger.info("\n" + "="*70)
        logger.info("  GPU DIAGNOSTICS")
        logger.info("="*70)
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"Allocated Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        logger.info(f"Reserved Memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        logger.info(f"Available Memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB")
        logger.info("="*70 + "\n")

def main():
    os.makedirs(CONFIG.CHECKPOINT_DIR, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print_gpu_stats()
    
    train_df = pd.read_parquet("../processed_train.parquet")
    test_df = pd.read_parquet("../processed_test.parquet")
    for col in CONFIG.MULTI_LABEL_COLS:
        if col not in test_df.columns:
            test_df[col] = 0

    tokenizer = AutoTokenizer.from_pretrained(CONFIG.TEXT_MODEL_NAME)
    clip_stats = {'mean': [0.48145466, 0.4578275, 0.40821073], 'std': [0.26862954, 0.26130258, 0.27577711]}
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(**clip_stats)])
    val_transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(**clip_stats)])

    oof_preds, test_preds_list = np.zeros(len(train_df)), []
    kf = KFold(n_splits=CONFIG.NUM_FOLDS, shuffle=True, random_state=CONFIG.SEED)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        logger.info(f"\n===== FOLD {fold} =====")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        train_data, val_data = train_df.iloc[train_idx].reset_index(drop=True), train_df.iloc[val_idx].reset_index(drop=True)
        train_ds = MultiModalDataset(train_data, tokenizer, CONFIG.MAX_LENGTH, CONFIG.IMG_FOLDER_TRAIN, train_transform, CONFIG.NUMERIC_COLS, CONFIG.MULTI_LABEL_COLS, CONFIG.TARGET_COL)
        val_ds = MultiModalDataset(val_data, tokenizer, CONFIG.MAX_LENGTH, CONFIG.IMG_FOLDER_TRAIN, val_transform, CONFIG.NUMERIC_COLS, CONFIG.MULTI_LABEL_COLS, CONFIG.TARGET_COL)
        
        train_loader = DataLoader(
            train_ds, 
            batch_size=CONFIG.BATCH_SIZE, 
            shuffle=True, 
            num_workers=CONFIG.NUM_WORKERS, 
            pin_memory=CONFIG.PIN_MEMORY, 
            drop_last=True,
            persistent_workers=False
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=CONFIG.BATCH_SIZE, 
            shuffle=False, 
            num_workers=CONFIG.NUM_WORKERS, 
            pin_memory=CONFIG.PIN_MEMORY
        )
        
        model = GodTierModel(CONFIG)
        optimizer = get_optimizer(model, CONFIG)
        
        logger.info("\n" + "="*70)
        logger.info("🔧 MODEL INITIALIZATION")
        logger.info("="*70)
        logger.info(f"Model: GodTierModel")
        logger.info(f"Text Model: {CONFIG.TEXT_MODEL_NAME}")
        logger.info(f"Image Model: {CONFIG.IMG_MODEL_NAME}")
        logger.info(f"Device: {CONFIG.DEVICE}")
        logger.info(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info("="*70)
        
        start_epoch = 0
        best_val_loss = float('inf')
        
        if CONFIG.RESUME_TRAINING:
            ckpt_path = os.path.join(CONFIG.CHECKPOINT_DIR, f'fold_{fold}', 'last_epoch.pt')
            if os.path.exists(ckpt_path):
                logger.info("\n" + "="*70)
                logger.info(" MODEL CHECKPOINT LOADED SUCCESSFULLY!")
                logger.info("="*70)
                checkpoint = torch.load(ckpt_path, map_location=CONFIG.DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(CONFIG.DEVICE)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Move optimizer state to correct device
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(CONFIG.DEVICE)
                
                start_epoch = checkpoint.get('epoch', -1) + 1
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                logger.info(f" Fold: {fold}")
                logger.info(f" Resuming from epoch: {start_epoch}")
                logger.info(f" Best validation loss (so far): {best_val_loss:.6f}")
                logger.info(f"  Model state dict loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
                logger.info(f"  Optimizer state dict restored and moved to {CONFIG.DEVICE}")
                logger.info(f" Checkpoint location: {ckpt_path}")
                logger.info("="*70)
            else:
                logger.info("\n" + "="*70)
                logger.warning("  NO CHECKPOINT FOUND FOR THIS FOLD - Starting fresh!")
                logger.info(f" Starting from epoch: 0")
                logger.info(f" Fold: {fold}")
                logger.info("="*70)
        else:
            logger.info("\n" + "="*70)
            logger.info(" FRESH TRAINING MODE")
            logger.info(f" Starting from epoch: 0")
            logger.info(f" Fold: {fold}")
            logger.info("="*70)
        
        num_training_steps = math.ceil(len(train_loader) / CONFIG.GRAD_ACCUM_STEPS) * CONFIG.EPOCHS
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(0.1 * num_training_steps)), num_training_steps=max(1, num_training_steps)) if num_training_steps > 0 else None
        
        # If resuming, fast-forward the scheduler to the correct step
        if CONFIG.RESUME_TRAINING and start_epoch > 0 and scheduler is not None:
            steps_to_fast_forward = (start_epoch) * math.ceil(len(train_loader) / CONFIG.GRAD_ACCUM_STEPS)
            logger.info(f"Fast-forwarding scheduler by {steps_to_fast_forward} steps...")
            for _ in range(steps_to_fast_forward):
                scheduler.step()

        train_one_fold(fold, train_loader, val_loader, model, optimizer, scheduler, CONFIG, start_epoch=start_epoch, best_val_loss=best_val_loss)

        model_path = os.path.join(CONFIG.CHECKPOINT_DIR, f'fold_{fold}', 'best_model.pt')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        else:
            logger.warning(f"No 'best_model.pt' found for fold {fold}; using last epoch weights.")
        
        oof_preds[val_idx] = predict(val_loader, model, CONFIG)
        test_ds = MultiModalDataset(test_df, tokenizer, CONFIG.MAX_LENGTH, CONFIG.IMG_FOLDER_TEST, val_transform, CONFIG.NUMERIC_COLS, CONFIG.MULTI_LABEL_COLS, target_col=None)
        test_loader = DataLoader(test_ds, batch_size=CONFIG.BATCH_SIZE * 2, shuffle=False, num_workers=CONFIG.NUM_WORKERS)
        test_preds_list.append(predict(test_loader, model, CONFIG))
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if len(test_preds_list) == 0:
        avg_test_preds = np.zeros(len(test_df))
    else:
        avg_test_preds = np.mean(test_preds_list, axis=0)

    logger.info("\n===== Training Stacking LightGBM Model =====")
    X_stack_train = np.hstack([oof_preds.reshape(-1, 1), train_df[CONFIG.NUMERIC_COLS].values])
    X_stack_test = np.hstack([avg_test_preds.reshape(-1, 1), test_df[CONFIG.NUMERIC_COLS].values])
    y_stack_train = np.log1p(train_df[CONFIG.TARGET_COL].values)
    lgb_params = {'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 1000, 'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'lambda_l1': 0.1, 'num_leaves': 16, 'verbose': -1, 'n_jobs': -1, 'seed': CONFIG.SEED}
    stacking_model = lgb.LGBMRegressor(**lgb_params)
    stacking_model.fit(X_stack_train, y_stack_train)
    
    # Save the LightGBM stacking model
    lgb_model_path = os.path.join(CONFIG.CHECKPOINT_DIR, 'stacking_lightgbm_model.pkl')
    import joblib
    joblib.dump(stacking_model, lgb_model_path)
    logger.info("\n" + "="*70)
    logger.info(" LightGBM Stacking Model Saved Successfully!")
    logger.info("="*70)
    logger.info(f" Model saved at: {lgb_model_path}")
    logger.info(f" Model type: LGBMRegressor")
    logger.info(f" Number of estimators: {stacking_model.n_estimators}")
    logger.info(f" Learning rate: {stacking_model.learning_rate}")
    logger.info("="*70 + "\n")
    
    # Predict on train data using final stacked model
    train_preds_log = stacking_model.predict(X_stack_train)
    train_preds = np.expm1(train_preds_log)
    train_preds[train_preds < 0] = 0.01
    train_targets_actual = train_df[CONFIG.TARGET_COL].values
    train_smape = calculate_smape(train_targets_actual, train_preds)
    
    logger.info("\n" + "="*70)
    logger.info(" FINAL MODEL PERFORMANCE ON TRAINING DATA")
    logger.info("="*70)
    logger.info(f"Train SMAPE: {train_smape:.4f}%")
    logger.info("="*70 + "\n")
    
    # Predict on test data using final stacked model
    final_preds_log = stacking_model.predict(X_stack_test)
    final_preds = np.expm1(final_preds_log)
    final_preds[final_preds < 0] = 0.01
    submission_df = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': final_preds})
    submission_df.to_csv('test_out.csv', index=False)
    logger.info("\n Finished: test_out.csv saved!")
    logger.info(submission_df.head())

if __name__ == "__main__":
    main()
