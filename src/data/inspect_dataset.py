# Şimdi metadata inspeciton yaparak dataseti RAM e yüklemeden dataset hakkında bilgi edinmek için bir fonksiyon yazalım. çünkü dataset büyük olabilir ve RAM'e yüklenmesi zaman alabilir, RAM taşabilir. 

from pathlib import Path
import numpy as np

from src.utils.paths import get_dataset_root



def list_top_level_contents():
    dataset_root = get_dataset_root()

    print(f"Dataset root: {dataset_root}")
    print(f"Exists: {dataset_root.exists()}")
    print(f"Is directory: {dataset_root.is_dir()}")


    contents = list(dataset_root.iterdir())
    # iterdir() fonksiyonu, belirtilen dizindeki Dosya yollarını döndürür. Veri içeriği RAM'e yüklenmez.Sadece path bilgilerini iterator olarak döndürür..
    # iterator: bir koleksiyonun elemanlarını sırayla döndüren bir nesnedir.

    print(f"Top-level item count: {len(contents)}")

    if len(contents) == 0:
        print("No files or folders found in dataset root.")
        return

    for item in contents:
        item_type = "DIR" if item.is_dir() else "FILE"

        if item.is_file():
            size =format_size(item.stat().st_size)
        else:
            size = "-"    

        print(f"[{item_type}] {item.name} | Size: {size}")

    # for döngüsü, dataset_root dizinindeki her bir öğeyi kontrol eder ve öğenin türünü (dosya veya dizin) ve boyutunu (sadece dosyalar için) yazdırır. 


def format_size(size_bytes:int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.2f} KB"      
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"  
    
# Bilgisayar dosya boyutunu byte olarak tutar.Ama insan için KB/MB/GB daha okunabilir.    


def list_directory_contents(relative_dir: str):
    dataset_root = get_dataset_root()
    target_dir = dataset_root / relative_dir

    print(f"Directory: {target_dir}")
    print(f"Exists: {target_dir.exists()}")
    print(f"Is directory: {target_dir.is_dir()}")

    if not target_dir.exists():
        raise FileNotFoundError(f"Directory not found {target_dir}")
    
    contents = list(target_dir.iterdir())

    print(f"Item count: {len(contents)}")

    for item in contents:
        item_type = "DIR" if item.is_dir() else "FILE"

        if item.is_file():
            size = format_size(item.stat().st_size)
        else:
            size = "-"

        print(f"[{item_type}] {item.name} | Size: {size}")

# Relative_dir parametresi, dataset_root dizinine göre bir alt dizini belirtir. 
# Fonksiyon, bu alt dizinin içeriğini listeler ve her öğenin türünü ve boyutunu yazdırır. 
# Eğer belirtilen alt dizin bulunmazsa, bir FileNotFoundError hatası fırlatır.
# Yani tek fonksiyonla dataset_root dizininin içeriğini ve istediğimiz alt dizinin içeriğini listeleyebiliriz.            


def inspect_npy_file(relative_path: str):
    dataset_root = get_dataset_root()
    file_path = dataset_root / relative_path

    print(f"File: {file_path}")
    print(f"File: {file_path.exists()}")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    #array = np.load(file_path, mmap_mode="r")
    
# np.load fonksiyonu, belirtilen .npy dosyasını yükler. 
# mmap_mode="r" parametresi, dosyanın belleğe haritalanarak okunmasını sağlar. 
# Bu, büyük dosyaların RAM'e tamamen yüklenmeden erişilmesine olanak tanır. 
# Böylece, büyük veri setleriyle çalışırken bellek kullanımını optimize eder.

    array = np.load(file_path, allow_pickle=True) # allow_pickle=True, numpy dizilerinin içindeki nesnelerin (örneğin, listeler veya sözlükler) pickle formatında saklanmasına izin verir.

    print(f"Shape: {array.shape}")
    print(f"Dtype: {array.dtype}")
    print(f"Type: {type(array)}")

    if array.dtype == object:
        loaded_object = array.item()

        print(f"Loaded object type: {type(loaded_object)}")

        if isinstance(loaded_object, dict):
            print(f"Dictionary keys: {list(loaded_object.keys())}")

            for key, value in loaded_object.items():
                print(f"\nKey: {key}")
                print(f"Value type: {type(value)}")

                if hasattr(value, "shape"):
                    print(f"Value shape: {value.shape}")
                    print(f"Value dtype: {value.dtype}")

    
    
def summarize_subject_file(relative_path: str):
    dataset_root = get_dataset_root()
    file_path = dataset_root / relative_path

    print(f"File: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    array = np.load(file_path, allow_pickle=True)
    data = array.item()

    clip_lengths = []

    for clip_name, clip_array in data.items():
        window_count = clip_array.shape[0]
        clip_lengths.append(window_count)
        print(f"{clip_name}: {clip_array.shape}")

    print(f"Clip count: {len(data)}")
    print(f"Min windows: {min(clip_lengths)}")
    print(f"Max windows: {max(clip_lengths)}")
    print(f"Mean windows: {sum(clip_lengths) / len(clip_lengths):.2f}")    


    first_clip_name=list(data.keys())[0]
    first_clip = data[first_clip_name]

    print(f"First clip name: {first_clip_name}")
    print(f"Fİrst clip shape: {first_clip.shape}")
    print(f"Feature shape per window: {first_clip.shape[1:]}")



def check_eeg_eye_alignment(session: str, subject_file: str):
    dataset_root = get_dataset_root()

    eeg_path = dataset_root / "eeg_features" / session / subject_file
    eye_path = dataset_root / "eye_features" / session / subject_file

    print(f"EEG file: {eeg_path}")
    print(f"Eye file: {eye_path}")

    if not eeg_path.exists():
        raise FileNotFoundError(f"EEG file not found: {eeg_path}")

    if not eye_path.exists():
        raise FileNotFoundError(f"Eye file not found: {eye_path}")
    
    eeg_data = np.load(eeg_path, allow_pickle=True).item()
    eye_data = np.load(eye_path, allow_pickle=True).item()

    eeg_clips = set(eeg_data.keys())
    eye_clips = set(eye_data.keys())

    if eeg_clips != eye_clips:
        print("Clip keys do not match!")
        print(f"Only in EEG: {eeg_clips - eye_clips}")
        print(f"Only in Eye: {eye_clips - eeg_clips}")
        return
    
    print(f"Clip keys match: {len(eeg_clips)} clips")

    mismatches = []

    for clip_name in sorted(eeg_data.keys()):
        eeg_windows = eeg_data[clip_name].shape[0]
        eye_windows = eye_data[clip_name].shape[0]

        if eeg_windows != eye_windows:
            mismatches.append((clip_name, eeg_windows, eye_windows))

    if len(mismatches) == 0:
        print("All clips are temporally aligned.")
    else:
        print("Alignment mismatches found:")
        for clip_name, eeg_windows, eye_windows in mismatches:
            print(f"{clip_name}: EEG_{eeg_windows}, Eye={eye_windows}")        



def check_all_eeg_eye_alignments():
    dataset_root = get_dataset_root()

    sessions = ["session_1", "session_2", "session_3"]

    total_pairs = 0
    failed_pairs = []

    for session in sessions:
        eeg_dir = dataset_root / "eeg_features" / session
        eye_dir = dataset_root / "eye_features" / session

        eeg_files = sorted(eeg_dir.glob("*.npy"))
        
        print(f"\nSession: {session}")
        print(f"EEG files: {len(eeg_files)}")

        for eeg_file in eeg_files:
            subject_file = eeg_file.name
            eye_file = eye_dir / subject_file

            total_pairs += 1

            if not eye_file.exists():
                failed_pairs.append((session, subject_file,"missing eye file"))
                continue

            eeg_data = np.load(eeg_file, allow_pickle=True).item()
            eye_data = np.load(eye_file, allow_pickle=True).item()


            for clip_name in eeg_data.keys():
                eeg_windows = eeg_data[clip_name].shape[0]
                eye_windows = eye_data[clip_name].shape[0]

                if eeg_windows != eye_windows:
                    failed_pairs.append(
                        (session, subject_file, f"{clip_name}: EEG={eeg_windows}, Eye={eye_windows}")
                    )
        print(f"Checked session: {session}")

    print("\n--- ALL EEG / EYE ALİGNMENT SUMMARY ---")
    print(f"Total file pairs checked: {total_pairs}")
    print(f"Failed pairs: {len(failed_pairs)}")

    if failed_pairs:
        for item in failed_pairs:
            print(item)
    else:
        print("All EEG and Eye feature files are temporally aligned.")


   



if __name__ == "__main__":
    list_top_level_contents()

    print("\n---EEG FEATURES---\n")
    list_directory_contents("eeg_features")

    print("\n---EYE FEATURES---\n")
    list_directory_contents("eye_features")

    print("\n---EEG FEATURES / SESSİON 1---\n")
    list_directory_contents("eeg_features/session_1")

    print("\n---EYE FEATURES / SESSİON 1---\n")
    list_directory_contents("eye_features/session_1")

    print("\n--- SAMPLE EEG FEATURE FILE ---")
    inspect_npy_file("eeg_features/session_1/sub10_20200903.npy")

    print("\n--- SAMPLE EYE FEATURE FILE ---")
    inspect_npy_file("eye_features/session_1/sub10_20200903.npy")

    print("\n--- EEG SUBJECT SUMMARY ---")
    summarize_subject_file("eeg_features/session_1/sub10_20200903.npy")

    print("\n--- EYE SUBJECT SUMMARY ---")
    summarize_subject_file("eye_features/session_1/sub10_20200903.npy")

    print("\n--- EEG / EYE ALIGNMENT CHECK ---")
    check_eeg_eye_alignment("session_1", "sub10_20200903.npy")

    print("\n--- ALL EEG / EYE ALIGNMENT CHECK---")
    check_all_eeg_eye_alignments()


    
# Eğer bu dosya doğrudan çalıştırılıyorsa, list_top_level_contents fonksiyonunu çağırır ve dataset_root dizininin içeriğini listeler.
# Ardından, "eeg_features" ve "eye_features" alt dizinlerinin içeriğini listelemek için list_directory_contents fonksiyonunu çağırır.
            