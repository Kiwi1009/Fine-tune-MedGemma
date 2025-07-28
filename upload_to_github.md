# 🚀 上傳到 GitHub 指南

## 方法 1: 使用 GitHub Desktop (最簡單)

### 步驟 1: 安裝 GitHub Desktop
1. 前往 https://desktop.github.com/
2. 下載並安裝 GitHub Desktop
3. 使用您的 GitHub 帳戶登入

### 步驟 2: 添加倉庫
1. 打開 GitHub Desktop
2. 點擊 "Add an existing repository from your hard drive"
3. 選擇專案目錄：`C:\Users\User\Documents\GitHub\Fine-tune-MedGemma`
4. 點擊 "Add Repository"

### 步驟 3: 發布倉庫
1. 點擊 "Publish repository"
2. 設置倉庫名稱：`Fine-tune-MedGemma`
3. 描述：`MedGemma 27B Chinese fine-tuning project with GPU optimization`
4. 選擇 Public 或 Private
5. 點擊 "Publish Repository"

## 方法 2: 使用 GitHub 網頁界面

### 步驟 1: 創建新倉庫
1. 前往 https://github.com/new
2. 設置倉庫名稱：`Fine-tune-MedGemma`
3. 描述：`MedGemma 27B Chinese fine-tuning project with GPU optimization`
4. 設為 Public
5. **不要**勾選 "Add a README file"
6. 點擊 "Create repository"

### 步驟 2: 推送代碼
創建倉庫後，在終端中運行：

```bash
# 添加遠程倉庫
git remote add origin https://github.com/kiwi1009/Fine-tune-MedGemma.git

# 推送代碼
git push -u origin main
```

## 方法 3: 使用命令行 (需要 Personal Access Token)

### 步驟 1: 創建 Personal Access Token
1. 前往 https://github.com/settings/tokens
2. 點擊 "Generate new token (classic)"
3. 選擇 "repo" 權限
4. 複製生成的 token

### 步驟 2: 設置認證
```bash
# 設置認證助手
git config --global credential.helper store

# 添加遠程倉庫
git remote add origin https://github.com/kiwi1009/Fine-tune-MedGemma.git

# 推送代碼 (會提示輸入用戶名和 token)
git push -u origin main
```

## 方法 4: 使用 GitHub CLI

### 步驟 1: 安裝 GitHub CLI
```bash
# 下載並安裝 GitHub CLI
# 前往 https://cli.github.com/
```

### 步驟 2: 認證並創建倉庫
```bash
# 認證
gh auth login

# 創建倉庫
gh repo create Fine-tune-MedGemma --public --source=. --remote=origin --push
```

## 📋 專案文件清單

確保以下文件都已準備好：

- ✅ `README_中文.md` - 詳細中文說明
- ✅ `快速開始指南.md` - 快速開始指南
- ✅ `finetune_medgemma_27b_中文.ipynb` - Jupyter Notebook
- ✅ `train_medgemma_27b.py` - Python 腳本
- ✅ `config.json` - 配置文件
- ✅ `requirements.txt` - 依賴套件
- ✅ `medquad.csv` - 數據集 (如果有的話)

## 🔧 故障排除

### 問題 1: 權限被拒絕
**解決方案**：
- 確認您已登入正確的 GitHub 帳戶
- 檢查 Personal Access Token 是否有正確權限

### 問題 2: 倉庫不存在
**解決方案**：
- 先在 GitHub 上創建倉庫
- 確認倉庫名稱拼寫正確

### 問題 3: 認證失敗
**解決方案**：
- 重新生成 Personal Access Token
- 使用 GitHub Desktop 作為替代方案

## 📞 需要幫助？

如果遇到問題，可以：
1. 查看 GitHub 的官方文檔
2. 使用 GitHub Desktop 作為最簡單的解決方案
3. 聯繫 GitHub 支援

---

**祝您上傳順利！** 🚀 