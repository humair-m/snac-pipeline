import torch
import multiprocessing as mp
import os
import gzip
import io
import time
from tqdm.auto import tqdm
from utils.snac_codec import SNACCoder
from utils.logging_config import setup_logging

try:
    import orjson
    USE_ORJSON = True
except Exception:
    import json
    USE_ORJSON = False


class AudioWorker:
    """Manages GPU worker processes for audio encoding"""

    SENTINEL = None

    def __init__(self, rank: int, in_q: mp.Queue, out_dir: str, dataset_prefix: str,
                 gzip_level: int, buffer_size: int, lines_per_file: int, num_readers: int,
                 model_id: str, num_layers: int):
        self.rank = rank
        self.in_q = in_q
        self.out_dir = out_dir
        self.dataset_prefix = dataset_prefix
        self.gzip_level = gzip_level
        self.buffer_size = buffer_size
        self.lines_per_file = lines_per_file
        self.num_readers = num_readers
        self.model_id = model_id
        self.num_layers = num_layers

    def _open_rotated_file(self, idx: int):
        """Opens a new file for writing"""
        path = os.path.join(
            self.out_dir,
            f"{self.dataset_prefix}-worker{self.rank:02d}-{idx:05d}.jsonl.gz"
        )
        raw = open(path, "wb", buffering=0)
        buf = io.BufferedWriter(raw, buffer_size=self.buffer_size)
        gz = gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=self.gzip_level, mtime=0)
        txt = io.TextIOWrapper(gz, encoding="utf-8", newline="\n")
        return path, raw, buf, gz, txt

    def _close_file(self, raw, buf, gz, txt):
        """Safely closes file"""
        for f, name in [(txt, "txt"), (gz, "gz"), (buf, "buf"), (raw, "raw")]:
            try:
                if f:
                    if hasattr(f, 'flush'):
                        f.flush()
                    if hasattr(f, 'detach') and name == "txt":
                        f.detach()
                    elif hasattr(f, 'close'):
                        f.close()
            except Exception:
                pass

    def _dump_line(self, obj, gz, txt):
        """Writes line to file"""
        if USE_ORJSON:
            b = orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS)
            gz.write(b)
            gz.write(b"\n")
        else:
            for i in range(1, self.num_layers + 1):
                k = f"snac_layer_{i}"
                if k in obj and hasattr(obj[k], "tolist"):
                    obj[k] = obj[k].tolist()
            if "token_lengths" in obj and hasattr(obj["token_lengths"], "tolist"):
                obj["token_lengths"] = obj["token_lengths"].tolist()
            txt.write(json.dumps(obj, ensure_ascii=False))
            txt.write("\n")

    @staticmethod
    def _flatten(x):
        """Converts array to flat form"""
        try:
            import numpy as np
            if x is None:
                return np.array([])
            return x.reshape(-1)
        except Exception:
            return x

    def run(self):
        """Worker process for audio processing"""
        setup_logging()
        tqdm.set_lock(mp.RLock())

        gpu_emoji = ["🟢", "🔵", "🟣", "🟡", "🔴", "⚫", "⚪", "🟠"][self.rank % 8]

        pbar = tqdm(
            desc=f"{gpu_emoji} GPU-{self.rank}",
            position=self.num_readers + self.rank + 1,
            leave=True,
            unit="items",
            bar_format='{desc}: {n_fmt} | {rate_fmt} | File {postfix}',
            dynamic_ncols=True,
            mininterval=0.5
        )

        torch.set_num_threads(1)

        try:
            pbar.set_description(f"{gpu_emoji} GPU-{self.rank} Loading...")
            model = SNACCoder(self.rank, model_id=self.model_id)
            pbar.set_description(f"{gpu_emoji} GPU-{self.rank} Ready")

            file_idx = 0
            lines_in_file = 0
            path, raw, buf, gz, txt = self._open_rotated_file(file_idx)
            pbar.set_postfix_str(f"{file_idx:05d}")

            n = 0

            while True:
                item = self.in_q.get()
                if item is self.SENTINEL:
                    break

                try:
                    codes = model(item["wave"])

                    rec = {"text": item["text"]}
                    
                    for i in range(1, self.num_layers + 1):
                        layer_key = f"snac_layer_{i}"
                        rec[layer_key] = self._flatten(codes[layer_key])
                    
                    rec["num_layers"] = codes["num_layers"]
                    rec["token_lengths"] = codes["token_lengths"]

                    if "speaker" in item:
                        rec["speaker"] = item["speaker"]

                    for key in item:
                        if key not in ["text", "wave", "speaker"]:
                            rec[key] = item[key]

                    self._dump_line(rec, gz, txt)
                    n += 1
                    lines_in_file += 1

                    pbar.update(1)

                    if lines_in_file >= self.lines_per_file:
                        self._close_file(raw, buf, gz, txt)
                        file_idx += 1
                        path, raw, buf, gz, txt = self._open_rotated_file(file_idx)
                        lines_in_file = 0
                        pbar.set_postfix_str(f"{file_idx:05d}")

                except Exception as e:
                    pbar.set_description(f"{gpu_emoji} GPU-{self.rank} ERROR: {str(e)[:30]}")
                    time.sleep(1)
                    pbar.set_description(f"{gpu_emoji} GPU-{self.rank}")
                    continue

        except Exception as e:
            pbar.set_description(f"{gpu_emoji} GPU-{self.rank} CRASHED: {e}")
        finally:
            self._close_file(raw, buf, gz, txt)
            total_files = file_idx + (1 if lines_in_file > 0 else 0)
            pbar.set_description(f"{gpu_emoji} GPU-{self.rank} DONE ({n:,} items, {total_files} files)")
            pbar.close()


def worker_process(rank: int, in_q: mp.Queue, out_dir: str, dataset_prefix: str,
                   gzip_level: int, buffer_size: int, lines_per_file: int, num_readers: int,
                   model_id: str, num_layers: int):
    """Entry point for worker process"""
    worker = AudioWorker(rank, in_q, out_dir, dataset_prefix, gzip_level, buffer_size,
                        lines_per_file, num_readers, model_id, num_layers)
    worker.run()
