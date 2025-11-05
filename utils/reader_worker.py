import multiprocessing as mp
from tqdm.auto import tqdm
from utils.dataset_processor import DatasetProcessor


class ReaderWorker:
    """Handles reading from dataset and pushing to queue"""

    def __init__(self, reader_id: int, total_readers: int, dataset_processor: DatasetProcessor, q: mp.Queue):
        self.reader_id = reader_id
        self.total_readers = total_readers
        self.dataset_processor = dataset_processor
        self.q = q

    def run(self):
        """Single reader worker that processes a portion of dataset"""
        tqdm.set_lock(mp.RLock())

        ds = self.dataset_processor.get_dataset()

        pbar = tqdm(
            desc=f"📖 Reader-{self.reader_id}",
            position=self.reader_id,
            leave=True,
            unit="items",
            bar_format='{desc}: {n_fmt} items | {rate_fmt} | {elapsed}',
            dynamic_ncols=True,
            mininterval=0.5
        )

        n = 0
        try:
            for i, ex in enumerate(ds):
                prepared_item = self.dataset_processor.prepare_item(ex)
                self.q.put(prepared_item)
                n += 1
                pbar.update(1)

                if n % 1000 == 0:
                    pbar.set_description(f"📖 Reader-{self.reader_id} ({n:,} processed)")

        except Exception as e:
            pbar.set_description(f"📖 Reader-{self.reader_id} ERROR: {e}")
        finally:
            pbar.set_description(f"📖 Reader-{self.reader_id} DONE ({n:,} items)")
            pbar.close()


def reader_worker_process(reader_id: int, total_readers: int, dataset_processor: DatasetProcessor, q: mp.Queue):
    """Entry point for reader worker process"""
    worker = ReaderWorker(reader_id, total_readers, dataset_processor, q)
    worker.run()
