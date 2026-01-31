import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Manager
from PIL import Image
from torchvision.transforms import ToTensor
from comer.datamodule import vocab
from comer.lit_comer import LitCoMER
import tqdm
import glob
from datetime import datetime
import gc
import tensorflow as tf

def tokens2sparse(tensor):
    indices = tf.where(tensor != 0)
    values = tf.gather_nd(tensor, indices)
    dense_shape = tf.cast(tf.shape(tensor), tf.int64)
    return tf.SparseTensor(indices, values, dense_shape)

def sparse2dense(tensor, shape):
    tensor = tf.sparse.reset_shape(tensor, shape)
    tensor = tf.sparse.to_dense(tensor, default_value=-1)
    tensor = tf.cast(tensor, tf.float32)
    return tensor

class SequenceAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='seq_acc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.count = self.add_weight(name='count', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
                
    def update_state(self, y_true, y_pred, **kwargs):
        batch_size, max_length = tf.shape(y_true)[0], tf.shape(y_true)[1]
        sparse_true = tokens2sparse(y_true)
        sparse_pred = tokens2sparse(y_pred)
        y_true = sparse2dense(sparse_true, [batch_size, max_length])
        y_pred = sparse2dense(sparse_pred, [batch_size, max_length])
        num_errors = tf.reduce_any(y_true != y_pred, axis=1)
        num_errors = tf.reduce_sum(tf.cast(num_errors, tf.float32))
        total = tf.cast(batch_size, tf.float32)
        self.count.assign_add(total - num_errors)
        self.total.assign_add(total)

    def result(self):
        return tf.math.divide_no_nan(self.count, self.total)

    def reset_state(self):
        self.count.assign(0)
        self.total.assign(0)

class CharacterAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='char_acc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.count = self.add_weight(name='count', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
                
    def update_state(self, y_true, y_pred, **kwargs):
        num_errors = tf.logical_and(y_true != y_pred, y_true != 0)
        num_errors = tf.reduce_sum(tf.cast(num_errors, tf.float32))
        total = tf.reduce_sum(tf.cast(y_true != 0, tf.float32))
        self.count.assign_add(total - num_errors)
        self.total.assign_add(total)

    def result(self):
        return tf.math.divide_no_nan(self.count, self.total)

    def reset_state(self):
        self.count.assign(0)
        self.total.assign(0)

def warp_cer_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int64)
    sparse_true = tokens2sparse(y_true)
    sparse_pred = tokens2sparse(y_pred)
    edit_distances = tf.edit_distance(sparse_pred, sparse_true, normalize=False)
    sum_distance = tf.reduce_sum(edit_distances)
    count_chars = tf.reduce_sum(tf.cast(y_true != 0, tf.float32))
    return tf.math.divide_no_nan(sum_distance, count_chars, name='cer')

def convert_to_tf_format(pred_indices, label_indices, max_len=None):
    if max_len is None:
        max_len = max(len(pred_indices), len(label_indices))
    pred_padded = pred_indices + [0] * (max_len - len(pred_indices))
    label_padded = label_indices + [0] * (max_len - len(label_indices))
    return (
        tf.constant([label_padded], dtype=tf.int64),
        tf.constant([pred_padded], dtype=tf.int64)
    )

def worker(args):
    img_paths, label_dict, ckpt, rank, progress_queue = args
    
    try:
        device = f'cuda:{rank % torch.cuda.device_count()}'
        print(f"[Worker {rank}] Loading model on {device}...")
        
        model = LitCoMER.load_from_checkpoint(ckpt, map_location=device)
        model.eval()
        model = model.to(device)
        
        print(f"[Worker {rank}] Processing {len(img_paths)} images...")
        
        total_seq_correct = 0.0
        total_char_correct = 0.0
        total_true_chars = 0.0
        sum_cer_per_image = 0.0 # Tổng CER của từng ảnh (để tính Mean CER)
        
        with tf.device('/cpu:0'):
            with torch.no_grad():
                for idx, img_path in enumerate(img_paths):
                    # PyTorch Inference
                    img = Image.open(img_path)
                    img_tensor = ToTensor()(img).to(device)
                    mask = torch.zeros_like(img_tensor, dtype=torch.bool)
                    
                    hyp = model.approximate_joint_search(img_tensor.unsqueeze(0), mask)[0]
                    pred_indices = hyp.seq
                    label = label_dict[img_path.split("/")[-1].split(".")[0]].replace(" ", "")
                    label_indices = vocab.words2indices(label)
                    
                    # Chuyển sang TF format
                    y_true, y_pred = convert_to_tf_format(pred_indices, label_indices) # [1, max_len], tf.int64

                    # --- 1. Sequence Accuracy Counts ---
                    batch_size, max_length = tf.shape(y_true)[0], tf.shape(y_true)[1]
                    sparse_true = tokens2sparse(y_true)
                    sparse_pred = tokens2sparse(y_pred)
                    y_true_dense = sparse2dense(sparse_true, [batch_size, max_length]) # padding: -1
                    y_pred_dense = sparse2dense(sparse_pred, [batch_size, max_length]) # padding: -1
                    
                    # Kiểm tra sự khác biệt trên toàn bộ chuỗi (bao gồm padding, nếu có)
                    num_errors_seq = tf.reduce_any(y_true_dense != y_pred_dense, axis=1) # [1,]
                    num_correct_seq = tf.reduce_sum(tf.cast(~num_errors_seq, tf.float32)).numpy()
                    total_seq_correct += num_correct_seq
                    
                    # --- 2. Character Accuracy Counts ---
                    
                    # Lỗi chỉ ở các vị trí y_true != 0 (không phải padding)
                    num_errors_char = tf.logical_and(y_true != y_pred, y_true != 0)
                    total_errors_char = tf.reduce_sum(tf.cast(num_errors_char, tf.float32)).numpy()
                    
                    # Tổng số ký tự thực tế
                    total_chars_in_sample = tf.reduce_sum(tf.cast(y_true != 0, tf.float32)).numpy()
                    
                    total_char_correct += (total_chars_in_sample - total_errors_char)
                    total_true_chars += total_chars_in_sample
                    
                    # --- 3. CER (Mean CER) ---
                    edit_distances = tf.edit_distance(sparse_pred, sparse_true, normalize=False)
                    sum_distance_sample = tf.reduce_sum(edit_distances).numpy()
                    
                    # CER của từng ảnh
                    cer_sample = sum_distance_sample / (total_chars_in_sample if total_chars_in_sample > 0 else 1.0)
                    sum_cer_per_image += cer_sample
                    
                    # Dọn dẹp
                    del img_tensor, mask, hyp
                    img.close()
                    
                    if (idx + 1) % 50 == 0:
                        torch.cuda.empty_cache()
                    
                    progress_queue.put(1)
            
            print(f"[Worker {rank}] Completed {len(img_paths)} images.")
            
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
            return {
                'seq_correct': total_seq_correct,
                'total_seq': float(len(img_paths)),
                'char_correct': total_char_correct,
                'total_chars': total_true_chars,
                'sum_cer': sum_cer_per_image
            }
        
    except Exception as e:
        print(f"[Worker {rank}] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()
        raise


def progress_tracker(progress_queue, total_images):
    pbar = tqdm.tqdm(total=total_images, desc="Overall Progress", unit="img", ncols=100)
    processed = 0
    
    while processed < total_images:
        try:
            progress_queue.get(timeout=0.1)
            processed += 1
            pbar.update(1)
            pbar.set_postfix({
                'processed': f'{processed}/{total_images}',
                'progress': f'{100*processed/total_images:.1f}%'
            })
        except:
            continue
    
    pbar.close()

def parallel_evaluate(image_paths, label_dict, ckpt, n_workers=4):
    print("\n" + "="*60)
    print("Starting Parallel Evaluation (TF Metrics)")
    print("="*60)
    print(f"Total images: {len(image_paths)}")
    print(f"Number of workers: {n_workers}")
    print(f"Checkpoint: {ckpt}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    chunk_size = len(image_paths) // n_workers
    chunks = [image_paths[i:i + chunk_size] for i in range(0, len(image_paths), chunk_size)]
    
    print("Data distribution across workers:")
    for i, chunk in enumerate(chunks):
        print(f"  Worker {i}: {len(chunk)} images")
    print("\n")
    
    mp.set_start_method('spawn', force=True)
    manager = Manager()
    progress_queue = manager.Queue()
    
    args = [(chunk, label_dict, ckpt, i, progress_queue) for i, chunk in enumerate(chunks)]
    
    tracker_process = mp.Process(target=progress_tracker, args=(progress_queue, len(image_paths)))
    tracker_process.start()
    
    print("Starting workers...\n")
    with Pool(n_workers) as pool:
        all_results = pool.map(worker, args)
    
    tracker_process.join()
    
    print("\n\nAggregating results...")
    flat_results = [r for r in all_results if r] # Đảm bảo không có kết quả None
    
    
    # 1. Sequence Accuracy (Exact Match Ratio)
    total_seq_correct = sum(r['seq_correct'] for r in flat_results)
    total_seq_samples = sum(r['total_seq'] for r in flat_results)
    sequence_accuracy = total_seq_correct / total_seq_samples if total_seq_samples > 0 else 0.0
    
    # 2. Character Accuracy (Tỉ lệ ký tự đúng trên tổng số ký tự thực)
    total_char_correct = sum(r['char_correct'] for r in flat_results)
    total_true_chars = sum(r['total_chars'] for r in flat_results)
    character_accuracy = total_char_correct / total_true_chars if total_true_chars > 0 else 0.0
    
    # 3. CER (Character Error Rate) - Tính Mean CER
    sum_cer_overall = sum(r['sum_cer'] for r in flat_results)
    n = len(image_paths) # Tổng số mẫu ban đầu
    cer = sum_cer_overall / n if n > 0 else 0.0
    
    metrics = {
        'sequence_accuracy': sequence_accuracy,
        'character_accuracy': character_accuracy,
        'cer': cer
    }
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults:")
    print(f"  Sequence Accuracy: {metrics['sequence_accuracy']*100:.2f}%")
    print(f"  Character Accuracy: {metrics['character_accuracy']*100:.2f}%")
    print(f"  Character Error Rate (CER): {metrics['cer']*100:.2f}%")
    print(f"  Total samples evaluated: {int(total_seq_samples)}")
    print("="*60 + "\n")
    
    return metrics

# ----------------------------------------------------------------------
# ===== Sequential Evaluate (Đã sửa logic tích lũy metric) =====
# ----------------------------------------------------------------------
def sequential_evaluate_with_progress(image_paths, label_dict, ckpt, device='cuda:0'):
    print("\n" + "="*60)
    print("Starting Sequential Evaluation (TF Metrics)")
    print("="*60)
    print(f"Total images: {len(image_paths)}")
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    print("Loading model...")
    model = LitCoMER.load_from_checkpoint(ckpt, map_location=device)
    model.eval()
    model = model.to(device)
    print("Model loaded successfully.\n")
    
    total_seq_correct = 0.0
    total_char_correct = 0.0
    total_true_chars = 0.0
    sum_cer_per_image = 0.0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm.tqdm(image_paths, desc="Processing", unit="img", ncols=100)
        
        for idx, img_path in enumerate(pbar):
            img = Image.open(img_path)
            img_tensor = ToTensor()(img).to(device)
            mask = torch.zeros_like(img_tensor, dtype=torch.bool)
            
            hyp = model.approximate_joint_search(img_tensor.unsqueeze(0), mask)[0]
            pred_indices = hyp.seq
            label = label_dict[img_path.split("/")[-1].split(".")[0]].replace(" ", "")
            label_indices = vocab.words2indices(label)
            
            y_true, y_pred = convert_to_tf_format(pred_indices, label_indices)
            
            with tf.device('/cpu:0'):
                # --- 1. Sequence Accuracy Counts ---
                batch_size, max_length = tf.shape(y_true)[0], tf.shape(y_true)[1]
                sparse_true = tokens2sparse(y_true)
                sparse_pred = tokens2sparse(y_pred)
                y_true_dense = sparse2dense(sparse_true, [batch_size, max_length])
                y_pred_dense = sparse2dense(sparse_pred, [batch_size, max_length])
                num_errors_seq = tf.reduce_any(y_true_dense != y_pred_dense, axis=1)
                num_correct_seq = tf.reduce_sum(tf.cast(~num_errors_seq, tf.float32)).numpy()
                
                # --- 2. Character Accuracy Counts ---
                num_errors_char = tf.logical_and(y_true != y_pred, y_true != 0)
                total_errors_char = tf.reduce_sum(tf.cast(num_errors_char, tf.float32)).numpy()
                total_chars_in_sample = tf.reduce_sum(tf.cast(y_true != 0, tf.float32)).numpy()
                
                # --- 3. CER ---
                edit_distances = tf.edit_distance(sparse_pred, sparse_true, normalize=False)
                sum_distance_sample = tf.reduce_sum(edit_distances).numpy()
                cer_sample = sum_distance_sample / (total_chars_in_sample if total_chars_in_sample > 0 else 1.0)
            
            total_seq_correct += num_correct_seq
            total_char_correct += (total_chars_in_sample - total_errors_char)
            total_true_chars += total_chars_in_sample
            sum_cer_per_image += cer_sample
            total_samples += 1
            
            del img_tensor, mask, hyp
            img.close()
            
            current_seq_acc = total_seq_correct / total_samples if total_samples > 0 else 0.0
            current_cer = sum_cer_per_image / total_samples if total_samples > 0 else 0.0
            pbar.set_postfix({
                'SeqAcc': f'{current_seq_acc*100:.1f}%',
                'CER': f'{current_cer*100:.1f}%'
            })
            
            if (idx + 1) % 100 == 0:
                torch.cuda.empty_cache()
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    metrics = {
        'sequence_accuracy': total_seq_correct / total_samples if total_samples > 0 else 0.0,
        'character_accuracy': total_char_correct / total_true_chars if total_true_chars > 0 else 0.0,
        'cer': sum_cer_per_image / total_samples if total_samples > 0 else 0.0
    }
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults:")
    print(f"  Sequence Accuracy: {metrics['sequence_accuracy']*100:.2f}%")
    print(f"  Character Accuracy: {metrics['character_accuracy']*100:.2f}%")
    print(f"  Character Error Rate (CER): {metrics['cer']*100:.2f}%")
    print(f"  Total samples evaluated: {total_samples}")
    print("="*60 + "\n")
    
    return metrics

if __name__ == '__main__':
    print("Loading data...")
    image_paths = glob.glob("comer_data_test/data/val/img/*.bmp")
    with open("comer_data_test/data/val/caption.txt", "r", encoding="utf-8") as f:
        val_caption = f.read()
    
    label_dict = {}
    for line in val_caption.split("\n"):
        if line.strip():
            label_dict[line[:9]] = line[10:]
    
    print(f"Loaded {len(image_paths)} images and {len(label_dict)} labels\n")
    
    ckpt = "lightning_logs/version_48/checkpoints/epoch=259-step=61880-val_ExpRate=0.4218.ckpt"
    
    metrics = sequential_evaluate_with_progress(image_paths, label_dict, ckpt, device='cuda:0')
    metrics = parallel_evaluate(image_paths, label_dict, ckpt, n_workers=3)