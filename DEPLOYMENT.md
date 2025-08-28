# Deployment Guide: Howie CLI v2.2

## 🎯 Version 2.2 Features

Howie CLI v2.2 introduces a comprehensive search workflow with enhanced logging and multi-model support for maximum accuracy in fantasy football analysis.

## 🚀 Deployment Strategy: GitHub + Data Migration

### **Step 1: Push Enhanced Code to GitHub**

```bash
# From your current computer
cd /Users/trevor.gurgick/projectHowie

# Initialize git if not already done
git init
git add .
git commit -m "Howie CLI v2.2: Comprehensive Search Workflow

- Implemented Plan → Search → Verify → Reflect workflow
- Added multi-source validation with quality scoring
- Enhanced logging with complete event tracking
- Improved tool execution and error handling
- Real-time data verification and smart fallbacks
- Cost optimization and performance improvements"

# Push to GitHub
git remote add origin https://github.com/yourusername/projectHowie.git
git branch -M main
git push -u origin main
```

### **Step 2: Export Your Database Files**

```bash
# Export your databases to a portable archive
python migrate_data.py export --source /path/to/your/old/data --output howie_databases.tar.gz

# This creates an archive with all your .db files
# File will be named something like: howie_data_20241225_143022.tar.gz
```

### **Step 3: Transfer Data (Choose One Method)**

#### Option A: Manual Transfer
```bash
# Copy the archive to your new computer via:
scp howie_data_*.tar.gz user@newcomputer:~/
# Or use USB drive, cloud storage, etc.
```

#### Option B: Cloud Storage
```bash
# Upload to cloud storage
cp howie_data_*.tar.gz ~/Dropbox/
cp howie_data_*.tar.gz ~/GoogleDrive/
cp howie_data_*.tar.gz ~/OneDrive/
```

#### Option C: Git LFS (for version control of databases)
```bash
# Install Git LFS
git lfs install

# Track database files
git lfs track "*.db" "*.sqlite" "*.tar.gz"
git add .gitattributes

# Add and commit databases
git add howie_data_*.tar.gz
git commit -m "Add database archive"
git push
```

### **Step 4: Set Up on New Computer**

```bash
# Clone the repository
git clone https://github.com/yourusername/projectHowie.git
cd projectHowie

# Install the enhanced version
./install.sh

# Import your database files
python migrate_data.py import --archive /path/to/howie_data_*.tar.gz

# Test everything works
howie ask "Show database info"
```

## 📋 Complete Deployment Checklist

### **On Source Computer (where you developed):**

- [ ] Review and update .gitignore (already done)
- [ ] Commit all enhanced code changes
- [ ] Push to GitHub repository
- [ ] Export database files: `python migrate_data.py export --source ./data`
- [ ] Transfer database archive to new computer
- [ ] Document any custom configurations

### **On Target Computer (new installation):**

- [ ] Clone repository from GitHub
- [ ] Set up Python environment: `python3 -m venv howie-env && source howie-env/bin/activate`
- [ ] Install: `./install.sh`
- [ ] Import databases: `python migrate_data.py import --archive howie_data_*.tar.gz`
- [ ] Set up API keys:
  ```bash
  export OPENAI_API_KEY="sk-..."
  export ANTHROPIC_API_KEY="sk-ant-..."
  export PERPLEXITY_API_KEY="pplx-..."
  ```
- [ ] Test installation: `howie --help && howie ask "Hello"`
- [ ] Verify database access: `howie ask "Show database info"`
- [ ] Configure models: `howie configure`

## 🔄 Ongoing Synchronization Options

### **Option 1: Manual Periodic Sync**
```bash
# On source computer (after updates)
python migrate_data.py export --source ./data
# Transfer archive to other computers
# On target computers
python migrate_data.py import --archive latest_howie_data.tar.gz
```

### **Option 2: Cloud Storage Sync**
```bash
# Set up automated sync
mkdir ~/CloudSync/HowieData
python migrate_data.py export --source ./data --output ~/CloudSync/HowieData/howie_latest.tar.gz
# Use cloud storage client to sync folder
```

### **Option 3: Shared Network Storage**
```bash
# If you have network storage
ln -s /network/shared/howie_data ./data
# All computers access same data location
```

### **Option 4: Git LFS for Everything**
```bash
# Track databases in git
git lfs track "data/*.db"
git add data/*.db
git commit -m "Update databases"
git push

# On other computers
git pull
git lfs pull
```

## 🛠️ Migration Commands Reference

```bash
# Export databases from old location
python migrate_data.py export --source /old/path/data --output backup.tar.gz

# Import to new location
python migrate_data.py import --archive backup.tar.gz --target ./

# Verify databases work
python migrate_data.py verify --path ./data

# Full migration from old installation
python migrate_data.py migrate --old-location /old/howie --new-location ./
```

## 🔐 Security Considerations

### **What NOT to commit to GitHub:**
- API keys (handled by .gitignore)
- Personal configurations with sensitive data
- User session data
- Large database files (use separate transfer)

### **What TO commit to GitHub:**
- All enhanced code
- Documentation
- Example configurations (without API keys)
- Installation scripts

## 🚀 Quick Commands Summary

```bash
# On source computer
git add . && git commit -m "Enhanced Howie" && git push
python migrate_data.py export --source ./data

# Transfer archive file to new computer

# On target computer  
git clone https://github.com/yourusername/projectHowie.git
cd projectHowie
./install.sh
python migrate_data.py import --archive howie_data_*.tar.gz
howie ask "Test installation"
```

## 💡 Pro Tips

1. **Version Control**: Use git tags for releases
   ```bash
   git tag -a v2.2.0 -m "Comprehensive search workflow version"
   git push --tags
   ```

2. **Backup Strategy**: Keep multiple database backups
   ```bash
   python migrate_data.py export --source ./data --output "backup_$(date +%Y%m%d).tar.gz"
   ```

3. **Environment Management**: Use different API keys per environment
   ```bash
   # Development
   export OPENAI_API_KEY="sk-dev-key"
   
   # Production  
   export OPENAI_API_KEY="sk-prod-key"
   ```

4. **Configuration Sync**: Export/import model configurations
   ```bash
   python migrate_data.py export --source ./data --include-config
   ```

This approach gives you:
- ✅ **Code on GitHub**: Easy to clone and update
- ✅ **Data Portability**: Secure transfer of valuable databases
- ✅ **Clean Separation**: Code vs data properly separated
- ✅ **Version Control**: Track changes to enhanced system
- ✅ **Backup Safety**: Multiple copies of important data

The enhanced Howie system will work seamlessly on any computer once properly deployed!