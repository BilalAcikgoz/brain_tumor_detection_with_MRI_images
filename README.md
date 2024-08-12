BRAIN TUMOR SEGMENTATION AND CLASSIFICATION

Merhaba herkese! Bu projenin amacı beyin MR görüntüleri üzerinde tümörün bulunduğu bölgeleri segmente edip aynı zamanda bu tümörün 
hangi sınıfa ait olduğunu bulmak. Toplamda 4 sınıfımız mevcut: Glioma, Meningioma, Pituitary, No Tumor

ABOUT THE DATA AND METHOD

Verisetimizi sınıflandıma ve segmentasyon olmak üzere ikiye ayırabiliriz. Sınıflandırma veriseti içerisinde train-val-test klasörleri
bulunmakta ve herbirininin içerisinde yukarıda bahsi geçen 4 sınıf bulunmaktadır. Bu verisetini kullanarak sınıflandırma işlemimizi
gerçekleştiriyoruz. Sınıflandırma için resnet18 mimarisini kullanmaktayız ve başarı oranı hem eğitim hem doğrulama veri kümesi üzerinde
%98'dir. 

Segmentasyon verisetini sınıflandırma veriseti içerisinden almış olup bu verileri roboflow aracılığıyla segmentasyon için etiketledik.
Otomatik oluşturulan 'yaml' dosyası sayesinde verisetimizi kolay bir şekilde YOLOv8-seg modelimize vermiş olduk.

ABOUT THE APPLICATION

Uygulamamızın arayüzünü streamlit kütüphanesi ile yazmış olup modellerimizi de bu kütüphane sayesinde uygulamamıza aktradık ve modelimiz
ile ilgili tahminlerde bulunabildik. İki ayrı modelimiz olduğu için butona basıldığında iki modelimiz birden çalışacak ve görüntü
üzerinde hem segmentasyon hem de sınıflandırma işlemini uygulayacaktır. Aşağıda uygulamanın kısa bir gösterimi yer almaktadır.

![Screenshot from 2024-08-12 12-33-36](https://github.com/user-attachments/assets/b8132dad-5fe4-4f45-b529-abbd6e6272e1)
