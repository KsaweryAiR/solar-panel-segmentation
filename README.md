## ZPO Project - Segmentacja paneli słonecznych

Projekt koncentruje się na opracowaniu modelu głębokiego uczenia do **automatycznej segmentacji paneli słonecznych** z map o wysokiej rozdzielczośći. Model jest trenowany na niestandardowym zbiorze danych obrazów ortofotomapy Poznania z 2022 roku i odpowiadających im masek, przygotowanych za pomocą QGIS i Roboflow. Wytrenowany model jest następnie konwertowany do formatu ONNX w celu integracji z wtyczką Deepness w QGIS.

### Zbiór danych

* **Dane:** Obrazy z publicznie dostępnych baz danych oraz zebrane z ortofotomapy Poznania w QGIS obszarów mieszkalnych z widocznymi panelami słonecznymi.
* **Adnotacje:** ok. 350 obrazów z ręcznie oznaczonymi panelami słonecznymi (maski przygotowane w Roboflow).
* **Przygotowanie:**
    * Surowe obrazy zostały wstępnie przetworzone w QGIS w celu wyodrębnienia odpowiednich regionów i dostosowania rozdzielczości.
    * Panele słoneczne zostały ręcznie oznaczone za pomocą Roboflow.
    * Obrazy zostały przeskalowane do 400x400 pikseli.
    * Zbiór danych został podzielony na zbiory treningowe i testowe.
* **Przechowywanie:** [Odnośnik do zbioru danych](https://drive.google.com/drive/folders/1omvS3l6GGmVazqJ8FP0KWdyu66Pk_7I-?usp=sharing)
* **Format:** Obrazy w formacie PNG, maski w formacie PNG.

### Trenowanie

* **Sieć:** U-Net z enkoderem ResNet34.
* **Trenowanie:** Trenowane za pomocą PyTorch z optymalizatorem Adam i funkcją straty Binary Cross-Entropy.
* **Parametry:**
    * Rozmiar partii: 350
    * Epoki: 100
* **Augmentacja:**
    * Losowe przycinanie
    * Losowe odbicie lustrzane w poziomie i pionie
    * Losowy obrót
    * Zmiana jasności
      
* **Środowisko:**
    * Python 3.12
    * Wymagania: [requirements.txt](requirements.txt)
    

### Wyniki

* **Dobre wykrycie:**
  
**Wideo:** [https://youtu.be/g4D_EFdLHG0?si=63SvEFi4WHk7xS2h](https://www.youtube.com/watch?v=EItXpzcvu2k)
  
<img src="photos/1.png" style="width: 80%; height: 80%;">
<img src="photos/2.png" style="width: 80%; height: 80%;">
<img src="photos/3.png" style="width: 80%; height: 80%;">
<img src="photos/4.png" style="width: 80%; height: 80%;">
<img src="photos/5.png" style="width: 80%; height: 80%;">
<img src="photos/6.png" style="width: 80%; height: 80%;">
<img src="photos/7.png" style="width: 80%; height: 80%;">

  
* **Złe wykrycie:**
  
Model czasami napotyka trudności w rozpoznawaniu szklarni, niektórych fragmentów ciemnych dachów oraz samochodów.
Najbardziej obawialiśmy się problemów z rozróżnianiem okien dachowych, ponieważ na zdjęciach satelitarnych mogą przypominać panele fotowoltaiczne. Jednak nasz model skutecznie radzi sobie z ich odróżnianiem.

<img src="photos/s1.png" style="width: 80%; height: 80%;">
<img src="photos/s2.png" style="width: 80%; height: 80%;">
  

  
* **Metryki:**
    * F1-score: 0.81
    * Precision: 0.87
    * Recall: 0.75

### Wytrenowany model w ONNX

* **Model:** Folder ONNX
* **Format:** ONNX z metadanymi Deepness (rozdzielczość przestrzenna, progi).

### Demo

* **Ortofotomapa:** Poznan 2022 aerial ortophoto high resolution
* **Lokalizacja:** zachodnia część Poznania, na zachód rzeki Warty. Wrocław, Biskupin i okolice

## 📌 Instrukcja uruchomienia  

### **1️⃣ Przygotowanie danych**  
Otwórz plik **`datamodulepanels.py`** i w oznaczonym miejscu wpisz ścieżkę do folderu zawierającego obrazy do trenowania.  
Następnie uruchom plik:  

```bash
python datamodulepanels.py
```

---

### **2️⃣ Trenowanie modelu**  
Po przygotowaniu danych uruchom proces trenowania modelu, wykonując:  

```bash
python train.py
```

Model zostanie zapisany jako **checkpoint** w folderze wyjściowym.  

---

### **3️⃣ Eksport modelu do ONNX**  
Po zakończeniu treningu otwórz **`evaluate.py`** i w wyznaczonym miejscu wpisz ścieżkę do pliku checkpointu z wytrenowanym modelem.  
Następnie uruchom skrypt, aby wyeksportować model do formatu **ONNX**:  

```bash
python evaluate.py
```

Model zostanie zapisany jako plik `.onnx`.  

---

### **4️⃣ Przygotowanie modelu do użycia w QGIS**  
W pliku **`output_model.py`** podaj ścieżkę do wygenerowanego modelu ONNX, a następnie uruchom skrypt:  

```bash
python output_model.py
```

---

### **5️⃣ Wykorzystanie modelu w QGIS**  
Po wygenerowaniu pliku w formacie ONNX, możesz załadować go do QGIS i przeprowadzić segmentację paneli fotowoltaicznych przy użyciu wtyczki Deepness.
  
---

## 📦 Wymagania  
Przed uruchomieniem projektu zainstaluj wymagane biblioteki:  

```bash
pip install -r requirements.txt
```

---

### Osoby

* Antonina Frąckowiak
* Ksawery Giera
* Daniel Błaszkiewicz
