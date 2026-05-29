from pathlib import Path 
import yaml

def load_paths(config_path: str = "configs/paths.yaml") -> dict:
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open (config_file , "r", encoding="utf-8") as file:
        paths = yaml.safe_load(file)

    return paths

# Yukarıdaki fonksiyon, belirtilen bir YAML dosyasından yolları yükler ve bir sözlük olarak döndürür. 
# Dosya bulunamazsa, bir FileNotFoundError hatası fırlatır. 
# YAML dosyasının içeriği, yolların anahtar-değer çiftleri şeklinde düzenlenmiş olmalıdır.


def get_dataset_root(config_path:str = "configs/paths.yaml") -> Path:
    paths = load_paths(config_path)

    dataset_root = Path(paths["dataset_root"])
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    return dataset_root


# Code reuse, yazılım geliştirme sürecinde mevcut kod parçalarının tekrar kullanılmasıdır. 
# Bu, yeni bir proje veya uygulama geliştirirken, daha önce yazılmış ve test edilmiş kodun yeniden kullanılmasını sağlar. 
# Code reuse, geliştirme sürecini hızlandırır, hata oranını azaltır ve bakım maliyetlerini düşürür.


# Yukarıdaki fonksiyon, load_paths fonksiyonunu kullanarak YAML dosyasından yolları yükler ve dataset_root anahtarına karşılık gelen yolu döndürür. 
# Eğer dataset_root yolu bulunamazsa, bir FileNotFoundError hatası fırlatır. 
# Bu fonksiyon, dataset_root yolunu almak için tekrar kullanılabilir ve böylece kodun tekrar yazılmasını önler.

# Bu tür fonksiyonlar, projenin farklı bölümlerinde dataset_root yolunu almak için kullanılabilir ve böylece kodun tekrar yazılmasını önler.
# Ayrıca, eğer dataset_root yolunu değiştirmek isterseniz, sadece YAML dosyasını güncellemeniz yeterli olacaktır, kodun diğer bölümlerini değiştirmeye gerek kalmaz.
# Bu, kodun bakımını kolaylaştırır ve hata yapma olasılığını azaltır.
# Profesyonel yazılım geliştirme süreçlerinde, kodun tekrar kullanılabilir olması önemlidir çünkü bu, geliştirme sürecini hızlandırır ve hata oranını azaltır.
# Abstract, bir sınıfın veya fonksiyonun genel bir tanımını ifade eder.
# Örneğin, bir "Animal" sınıfı abstract olabilir çünkü bu sınıfın belirli bir türü (örneğin, "Dog" veya "Cat") yoktur. 
# Ancak, bu sınıfın genel özellikleri (örneğin, "eat" veya "sleep" gibi) tanımlanabilir.  


# Yukarıdaki fonksiyonlar, belirli bir yapılandırma dosyasından yolları yüklemek ve dataset_root yolunu almak için kullanılır.
# Path sınıfı, dosya yollarını daha kolay ve güvenli bir şekilde yönetmek için kullanılır. 
# Os bağımsız bir şekilde çalışır ve farklı işletim sistemlerinde uyumlu yollar oluşturmanıza olanak tanır.


