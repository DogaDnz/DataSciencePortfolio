import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import tkinter as tk

root = tk.Tk()
root.title("Snow-Covered Area Detection")

# Pencere boyutu
root.geometry("900x600")

# Arkaplan görüntüsünü ayarla
background_image_path = "white.jpg"  # Karlı fotoğrafın yolu
bg_image = Image.open(background_image_path)
bg_image = bg_image.resize((900, 600))  # Pencere boyutuna göre yeniden boyutlandır
bg_photo = ImageTk.PhotoImage(bg_image)

# Canvas oluştur ve arkaplanı ekle
canvas = tk.Canvas(root, width=900, height=600)
canvas.pack(fill="both", expand=True) # pencereyi sığrıdır,pencere büyüdükçe canvas da büyür
canvas.create_image(0, 0, image=bg_photo, anchor="nw")

form_frame = tk.Frame(root, bg="#ffffff", highlightbackground="#d1d1d1", highlightthickness=2)
form_frame.place(relx=0.5, rely=0.5, anchor="center", width=500, height=300)

# Ekran ortalama
root.update_idletasks()
x = (root.winfo_screenwidth() - root.winfo_width()) // 2
y = (root.winfo_screenheight() - root.winfo_height()) // 2
root.geometry(f"+{x}+{y}")

# Font ayarları
label_font = ("Helvetica", 20, "bold")  # Yazı boyutu ve kalınlık
button_font = ("Helvetica", 20)        # Buton yazı boyutu

# Dil değişkeni
language = tk.StringVar(value="English")

# Dil metinleri
texts = {
    "Türkçe": {
        "load_image": "Görüntü Yükle",
        "run_analysis": "Analizi Çalıştır",
        "select_image": "Görüntü yüklemek ve analizi başlatmak için düğmelere tıklayın.",
        "no_image": "Hiçbir görüntü seçilmedi.",
        "image_loaded": "Görüntü yüklendi: ",
        "start_analysis": "Lütfen önce bir görüntü seçin.",
        "snow_coverage": "Karlı alan yüzdesi: ",
        "visual_titles": ["Orijinal Görüntü", "Segmentasyon", "Kar Maskesi", "Konturlu Görüntü"],
        "percentage_title": "Karlı alan yüzdesi: ",
    },
    "English": {
        "load_image": "Load Image",
        "run_analysis": "Run Analysis",
        "select_image": "Click the buttons to load an image and start the analysis.",
        "no_image": "No image selected.",
        "image_loaded": "Image loaded: ",
        "start_analysis": "Please select an image first.",
        "snow_coverage": "Snow coverage percentage: ",
        "visual_titles": ["Original Image", "Segmentation", "Snow Mask", "Contoured Image"],
        "percentage_title": "Snow coverage percentage: ",
    }
}


# Çıkış mesajı için label (Canvas üzerine ekle)
# görüntü yüklendi , lütfen bir görüntü seçin
output_label = tk.Label(root, text=texts[language.get()]["select_image"], wraplength=400, font=label_font, bg="white")
output_label_window = canvas.create_window(450, 50, window=output_label)

# Global değişken
image_path = None

def custom_print(message):
    """Mesajları tkinter arayüzünde görüntüler."""
    output_label.config(text=message)

def load_single_image():
    """Bir görüntü dosyası seçer ve yolunu kaydeder."""
    global image_path
    image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png")])
    if image_path:
        custom_print(f"{texts[language.get()]['image_loaded']}{image_path}")
    else:
        custom_print(texts[language.get()]["no_image"])

def run_analysis():
    """Seçilen görüntü üzerinde analiz yapar."""
    if not image_path:
        custom_print(texts[language.get()]["start_analysis"])
        return

    try:
        snow_coverage, mask_image = calculate_snow_coverage(image_path, selected_language=language.get())
        custom_print(f"{texts[language.get()]['snow_coverage']}{snow_coverage:.2f}%")
    except Exception as e:
        custom_print(f"Hata: {str(e)}")


# K-means analiz fonksiyonu
# k parametresi küme sayısını ifade eder
def calculate_snow_coverage(image_path, k=3, selected_language="Türkçe"):
    """
    Belirtilen görüntüde k-means clustering ile karla kaplı alanın yüzdesini hesaplar
    ve karlı alanların konturlarını çizer.
    """
    # Görüntüyü yükle
    image = cv2.imread(image_path) #imread ile görüntü yüklenir
    if image is None:
        raise ValueError(f"Görüntü yüklenemedi: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#çoğu kütüphane RGB formatında daha iyi çalışır

    # Görüntüyü piksel düzeyinde yeniden şekillendir
    pixel_values = image.reshape((-1, 3))
    #K-means için float32 türüne dönüştür.
    pixel_values = np.float32(pixel_values)

    # K-means ile segmentasyon
    # k değeri küme sayısını ifade eder.(segmentasyonda kaç adet küme oluşacağını belirler.)
    kmeans = KMeans(n_clusters=k, random_state=42, max_iter=1000)
    labels = kmeans.fit_predict(pixel_values)

    # Küme merkezlerini ve etiketleri yeniden şekillendir
    centers = np.uint8(kmeans.cluster_centers_) # küme merkezlerini al
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    # Karlı alan kümesini belirle (en parlak renk kümesi)
    brightest_cluster = np.argmax(np.sum(centers, axis=1))
    snow_mask = (labels == brightest_cluster)

    # Maske görüntüsünü oluştur
    snow_mask_image = snow_mask.reshape(image.shape[:2]).astype(np.uint8)

    # Kar yüzdesini doğrudan kar maskesi üzerinden hesapla
    snow_pixels = np.sum(snow_mask_image)  # Kar piksellerinin sayısı
    total_pixels = image.shape[0] * image.shape[1]  # Toplam piksel sayısı
    snow_coverage_percentage = (snow_pixels / total_pixels) * 100  # Yüzde hesaplama

    print(f"Karlı alan (kar maskesi) yüzdesi: {snow_coverage_percentage:.2f}%")

    visual_titles = texts[selected_language]["visual_titles"]
    percentage_title = texts[selected_language]["percentage_title"]

    # Görselleştirmeyi yap
    plt.figure(figsize=(20, 5))

    # Orijinal görüntü
    plt.subplot(1, 4, 1)
    plt.title(visual_titles[0])
    plt.imshow(image)
    plt.axis('off')

    # Segmentasyon sonucu
    plt.subplot(1, 4, 2)
    plt.title(visual_titles[1])
    plt.imshow(segmented_image)
    plt.axis('off')

    # Snow mask image
    plt.subplot(1, 4, 3)
    plt.title(visual_titles[2])
    plt.imshow(snow_mask_image * 255, cmap='gray')
    plt.axis('off')

    # Konturlu görüntü (isteğe bağlı, maske üzerinden)
    contoured_image = image.copy()
    contours, _ = cv2.findContours(snow_mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoured_image = cv2.drawContours(contoured_image, contours, -1, (255, 0, 0), 2)
    plt.subplot(1, 4, 4)
    plt.title(visual_titles[3])
    plt.imshow(contoured_image)
    plt.axis('off')

    plt.suptitle(f"{percentage_title} {snow_coverage_percentage:.2f}%", fontsize=16)
    plt.show()

    return snow_coverage_percentage, contoured_image


# Dil değişikliği
def change_language(event):
    """Dili değiştirir ve metinleri günceller."""
    selected_language = language.get()
    output_label.config(text=texts[selected_language]["select_image"])
    load_image_button.config(text=texts[selected_language]["load_image"])
    run_analysis_button.config(text=texts[selected_language]["run_analysis"])

# Dil seçici ComboBox
load_image_button = tk.Button(form_frame, text=texts[language.get()]["load_image"], command=load_single_image, font=button_font, bg="#007BFF", fg="white", relief="raised", borderwidth=2)
load_image_button.pack(pady=10, fill="x", padx=20)

run_analysis_button = tk.Button(form_frame, text=texts[language.get()]["run_analysis"], command=run_analysis, font=button_font, bg="#28A745", fg="white", relief="raised", borderwidth=2)
run_analysis_button.pack(pady=10, fill="x", padx=20)

# Language Selector
language_selector = ttk.Combobox(form_frame, textvariable=language, values=["Türkçe", "English"], font=button_font, state="readonly")
language_selector.bind("<<ComboboxSelected>>", change_language)
language_selector.pack(pady=10, fill="x", padx=20)

root.mainloop()