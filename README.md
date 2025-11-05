# SNAC Codec Audio Processing Pipeline

High-performance multi-GPU pipeline for encoding audio datasets using SNAC (Multi-Scale Neural Audio Codec).

## 🎯 Features

- **Multi-GPU Processing**: Parallel encoding across all available GPUs
- **Multi-Process Architecture**: Separate reader and worker processes for maximum throughput
- **Hierarchical Token Encoding**: SNAC's multi-scale temporal resolution
- **HuggingFace Integration**: Direct dataset loading and uploading
- **Compressed Output**: GZIP compressed JSONL shards
- **Progress Tracking**: Real-time progress bars for all workers

## 📋 SNAC Models

| Model | Sample Rate | Layers | Bitrate | Use Case |
|-------|-------------|--------|---------|----------|
| `hubertsiuzdak/snac_24khz` | 24 kHz | 3 | 0.98 kbps | 🗣️ Speech |
| `hubertsiuzdak/snac_32khz` | 32 kHz | 4 | 1.9 kbps | 🎸 Music/SFX |
| `hubertsiuzdak/snac_44khz` | 44 kHz | 4 | 2.6 kbps | 🎸 Music/SFX |

### Layer Structure

**24kHz (3 layers)**: Token sequences of lengths [12, 24, 48]
**32kHz (4 layers)**: Token sequences of lengths [12, 24, 48, 96]
**44kHz (4 layers)**: Token sequences of lengths [16, 32, 64, 128]

## 🚀 Quick Start

### 1. Installation

```bash
chmod +x setup.sh
./setup.sh
```

### 2. Activate Environment

```bash
source venv/bin/activate
```

### 3. Configure Pipeline

Edit `config.yaml`:

```yaml
base_settings:
  audio_codec: hubertsiuzdak/snac_24khz  # Choose your model
  num_readers: 8
  qsize: 100000
  OUT_DIR: shards
  gzip_level: 1
  buffer_size: 16777216
  lines_per_file: 50000
  load_dataset_num_proc: 20

save_settings:
  local: train_dataset
  hf_upload: your_username/your_dataset

hf_datasets:
  - name: mozilla-foundation/common_voice_11_0
    sub_name: en
    split: train
    text_column_name: sentence
    audio_column_name: audio
    speaker_column_name: null
    add_constant:
      - key: lang
        value: en
```

### 4. Run Pipeline

```bash
python main.py
```

## 📁 Output Format

Each encoded sample contains:

```json
{
  "text": "example text",
  "snac_layer_1": [token_ids],
  "snac_layer_2": [token_ids],
  "snac_layer_3": [token_ids],
  "snac_layer_4": [token_ids],  // Only for 32kHz/44kHz
  "num_layers": 3,
  "token_lengths": [12, 24, 48],
  "speaker": "optional_speaker_id",
  "lang": "en"
}
```

## ⚙️ Configuration Options

### Base Settings

- **audio_codec**: SNAC model ID
- **num_readers**: Number of dataset reader processes
- **qsize**: Queue size for audio samples
- **OUT_DIR**: Output directory for shards
- **gzip_level**: Compression level (1-9)
- **buffer_size**: File write buffer size
- **lines_per_file**: Samples per output file
- **load_dataset_num_proc**: Processes for dataset loading

### Dataset Settings

- **name**: HuggingFace dataset name
- **sub_name**: Dataset subset/configuration
- **split**: Dataset split (train/test/validation)
- **text_column_name**: Column containing text
- **audio_column_name**: Column containing audio
- **speaker_column_name**: Column containing speaker ID (optional)
- **add_constant**: Additional constant fields to add

## 🏗️ Architecture

```
┌─────────────────┐
│  Dataset Loader │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Readers │ (Multiple processes)
    └────┬────┘
         │
    ┌────▼────┐
    │  Queue  │
    └────┬────┘
         │
    ┌────▼────┐
    │ GPU     │ (One per GPU)
    │ Workers │
    └────┬────┘
         │
    ┌────▼────┐
    │  Shards │ (JSONL.GZ files)
    └─────────┘
```

## 📊 Performance Tips

1. **num_readers**: Set to 2-4x number of GPUs
2. **qsize**: Larger queue for unstable I/O (50k-100k)
3. **lines_per_file**: Balance between file count and size (25k-100k)
4. **load_dataset_num_proc**: Match CPU cores for fast loading

## 🔍 Monitoring

The pipeline shows real-time progress:

```
📖 Reader-0: 1,234 items | 45.2 it/s | 00:27
📖 Reader-1: 1,189 items | 43.8 it/s | 00:27
🟢 GPU-0: 856 | 31.5 it/s | File 00000
🔵 GPU-1: 833 | 30.1 it/s | File 00000
```



## 📝 License

This pipeline is provided as-is. SNAC codec license applies to the models.

## 🐛 Troubleshooting

**Out of Memory**: Reduce `qsize` or `num_readers`
**Slow Processing**: Increase `num_readers` or `load_dataset_num_proc`
**File Too Large**: Decrease `lines_per_file`

## 🔗 Links

- [SNAC Repository](https://github.com/hubertsiuzdak/snac)
- [SNAC Paper](https://arxiv.org/abs/2410.14411)
- [HuggingFace Models](https://huggingface.co/hubertsiuzdak) 
