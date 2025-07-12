# Data Directory Structure

This directory contains all runtime data for the Hephaestus RSI system:

## Subdirectories

- **models/**: Trained models, model checkpoints, and versioning data
- **cache/**: Performance caches, feature caches, and temporary computed data  
- **logs/**: System logs, audit logs, and performance metrics
- **temp/**: Temporary files and working directories
- **backups/**: System state backups and model snapshots

## Data Management

- All subdirectories are automatically created at runtime if missing
- Cache files are automatically cleaned up based on TTL policies
- Backup rotation is handled by the resource manager
- Logs are rotated daily and compressed after 7 days