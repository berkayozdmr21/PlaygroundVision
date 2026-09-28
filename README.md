# Çocuk Oyun Alanı — Test Uygulaması (Görüntü İşleme)

PC kamerasıyla çalışan test iskeleti. Kamerada gözüken herkesi tespit
edip takip eder, mouse ile tıkladığın kişiyi **HEDEF** yapar ve şu
tehlike durumlarını izler:

- **Kadraj dışına çıkma:** Hedef görüntüden çıkıp N saniye (varsayılan 10)
  geri dönmezse alarm.
- **Düşme:** Hedef düşüp M saniye (varsayılan 60) hareketsiz yerde
  kalırsa alarm.
- **Kavga (deneysel, düşük güven):** Kadrajdaki **herhangi iki kişi**
  temas hâlinde + yoğun/hızlı hareket ederse alarm. İsteğe bağlı olarak
  bir **AI şiddet modeli** (MoViNet/VD-MIL) bu sezgiyi doğrular — model
  de "şiddet" demedikçe alarm çalmaz (yanlış alarmı azaltır).

Bu, gerçek sistemdeki "aile mobilden çocuğunu etiketler, tehlike anında
bildirim gelir" akışının masaüstünde denenebilir ilk prototipidir.

## Dosyalar

| Dosya | Ne işe yarar |
|---|---|
| `tracking_core.py` | Ortak çekirdek. `SafetyMonitor` = tespit + takip + kadraj-dışı + düşme + kavga + tek alarm. |
| `violence_model.py` | Opsiyonel AI şiddet sınıflandırıcısı (MoViNet/VD-MIL veya ViT), ayrı thread'de çalışır. |
| `clip_recorder.py` | Test sırasında kısa video klipleri kaydeder (kendi modelini eğitmek için veri). |
| `desktop_app.py` | **Ana uygulama.** Modern Qt (PySide6) arayüzü. |
| `playground_test.py` | Sade OpenCV penceresi sürümü (yedek). |
| `models/vdmil/` | VD-MIL kavga modeli ağırlıkları (CC BY-NC 4.0 — sadece test, bkz. `NOTICE.txt`). |
| `dataset/` | Kaydedilen eğitim klipleri (`dataset/kavga/`, `dataset/normal/`). |

## Kurulum

```bash
pip install -r requirements.txt
```

> İlk çalıştırmada `yolov8n-pose.pt` model ağırlığı (~6 MB) otomatik
> indirilir; `ultralytics` + PyTorch kurulumu yüzlerce MB olabilir.
> İnternet bağlantısı gerekir. `PySide6` (Qt) de bu kurulumla gelir.

### Opsiyonel — AI kavga doğrulaması (MoViNet/VD-MIL)

"AI kavga doğrulaması" düğmesini kullanacaksan bir kütüphane daha gerekir:

```bash
pip install "git+https://github.com/roggerfq/MoViNet-pytorch.git"
```

Model ağırlıkları zaten `models/vdmil/` içinde. **Lisans CC BY-NC 4.0 —
sadece test/prototip, ticari ürüne geçmeden değiştirilmeli**
(`models/vdmil/NOTICE.txt`). Bu paket kurulu değilse uygulama yine
çalışır, kavga tespiti sadece sezgisel modda kalır.

## Çalıştırma (asıl uygulama)

```bash
python desktop_app.py
```

Açılışta kamera otomatik başlar (model yüklenirken birkaç saniye
"Kamera başlatılıyor..." yazabilir).

1. Video panelinde kendi kutuna **tıkla** → HEDEF olursun (kırmızı kutu +
   iskelet çizimi), rozet "Takip ediliyor" der.
2. **Kadraj dışı testi:** Kameradan çık → rozet "Kayıp — alarm N sn sonra",
   geri sayım. Süre dolunca "dıt dıt dıt" + kırmızı kenarlık. Geri dön →
   kıyafet rengi + beden oranıyla yeniden tanınır, alarm durur.
3. **Düşme testi:** Yere uzan ve hareketsiz kal → rozet "Hedef yerde —
   alarm M sn sonra", kutuda "YERDE 12sn" sayacı. Süre dolunca alarm.
   Ayağa kalk → "düşme takibi kapandı", alarm durur.
   (Test için "Yerde kalma alarmı" kaydırıcısını 10-15 sn'ye çekmen
   önerilir, 60 sn beklemek yerine.)
4. **Kavga testi:** İki kişi yan yana geçip birbirine yakın dururken
   kollarını hızlıca sallayın/itişin → iki kutu **mor** olur, aralarında
   mor çizgi çizilir, ~2 sn sonra "OLASI KAVGA" alarmı. Ayrılınca /
   sakinleşince durur. Çok yanlış alarm veriyorsa hassasiyeti düşür ya
   da "Kavga tespiti" düğmesiyle kapat.

5. **AI doğrulaması (opsiyonel):** "AI kavga doğrulaması" düğmesine bas
   (ilk sefer model yüklenir, birkaç sn). Artık kavga alarmı sadece
   sezgisel + AI ikisi birden "şiddet" derse çalar. Sağ altta "AI şiddet: 0.xx".
6. **Eğitim klibi topla:** Siz oyun oynarken / itişirken **"📹 Kavga klibi"**
   veya **"📹 Normal klip"** düğmesine bas → son ~8 sn + sonraki ~3 sn
   `dataset/kavga/` ya da `dataset/normal/` içine kaydedilir. Kavga alarmı
   olunca otomatik de kaydeder. Bu klipler sonra kendi modelini eğitmek için.

**Arayüzdeki kontroller**
- **Hedefi Sıfırla / Kamerayı Durdur-Başlat**
- **Kadraj dışı alarmı** kaydırıcısı — 3-30 sn (varsayılan 10).
- **Yerde kalma alarmı** kaydırıcısı — 10-120 sn (varsayılan 60).
- **Kavga tespiti** düğmesi + **Kavga hassasiyeti** kaydırıcısı (1 katı … 10 hassas).
- **AI kavga doğrulaması** düğmesi — MoViNet modelini devreye alır.
- **📹 Kavga klibi / Normal klip** — eğitim verisi kaydeder.
- **Kavga alarmında otomatik kaydet** — alarm anında klibi otomatik saklar.
- **Duruş** satırı + **Olay Günlüğü**.

## Hafif alternatif

```bash
python playground_test.py
```

Tuşlar: `q` çıkış · `r` hedefi sıfırla · `f` kavga tespiti · `a` AI doğrulama ·
`c` kavga klibi · `n` normal klip.

## Nasıl çalışıyor (kısaca)

1. **Tespit:** YOLOv8-pose her karede kişi kutuları + 17 iskelet noktası verir.
2. **Kimlik:** `SimpleTracker` kare-kareye ID atar — konum (IoU) + kıyafet
   rengi (histogram) + beden oranı (boy/en) birlikte skorlanır.
3. **Hedef:** Tıklanan kişi `TargetTracker`'a hedef olur. Kadraj dışına
   çıkarsa sayaç; geri gelince görünüm + beden oranıyla yeniden tanınır.
4. **Düşme:** `FallMonitor` hedefin gövde eksenine (omuz→kalça açısı) ve
   kutu oranına bakar. Sürekli "yatık" + **hareketsiz** ise yerde kalma
   süresi birikir; eşiği aşınca alarm. Belirgin hareket varsa ("yerde
   oynuyor") sayaç ilerlemez; ayağa kalkınca sıfırlanır.
5. **Kavga:** `FightMonitor` tüm kişi çiftlerine bakar. Temas (kutu
   çakışması / yakın merkez) + iki kişinin de bilek-dirsek noktalarının
   son ~0.5 sn'deki yol uzunluğu (salınımlı hareketi de yakalar) eşik
   üstündeyse ve bu ~2 sn sürerse o çift "alarm". Sakinleşince/ayrılınca düşer.
6. **AI kavga doğrulaması (opsiyonel):** Sezgisel bir çift yakalayınca
   sahnenin son ~1 sn'si (8 kare) `violence_model` üzerinden bir şiddet
   modeline (MoViNet/VD-MIL, RWF-2000+SCVD ile eğitilmiş) sorulur —
   AYRI thread'de, video donmadan. Model de "şiddet" (skor ≥ eşik)
   demedikçe alarm çalmaz.
7. **Alarm:** `SafetyMonitor` kadraj-dışı VEYA düşme VEYA kavga
   durumunda tek bir sesli alarmı yönetir.
8. **Klip kaydı:** `clip_recorder` her karenin HAM (çizimsiz) hâlini
   ~9 sn'lik bir tampona alır; kayıt istenince tamponu + sonraki birkaç
   saniyeyi `dataset/<etiket>/<zaman>.mp4` olarak yazar.

## Ayarlanabilir parametreler

`tracking_core.py` en üstünde (`DEFAULT_...` sabitleri):

| Parametre | Anlamı |
|---|---|
| `DEFAULT_CONF_THRESHOLD` | Kişi tespiti güven eşiği |
| `DEFAULT_MAX_LOST_FRAMES` | Bir takip kaç kare görünmezse silinir |
| `DEFAULT_REID_MATCH_THRESHOLD` | Kaybolan hedefi kıyafet rengi + boy/en oranıyla yeniden tanıma eşiği (0-1) |
| `DEFAULT_MISSING_ALARM_SECONDS` | Kadraj dışı alarm süresi (arayüzde kaydırıcı da var) |
| `DEFAULT_GROUND_ALARM_SECONDS` | Yerde kalma alarm süresi (arayüzde kaydırıcı da var) |
| `DEFAULT_AGITATION_THRESHOLD` | Kavga: uzuv hız eşiği — düşük = daha hassas (arayüzde hassasiyet kaydırıcısı) |

`FallMonitor` içinde: `FALL_DEBOUNCE_SECONDS`, `STAND_DEBOUNCE_SECONDS`,
`MOTION_PAUSE_THRESHOLD` (hareket eşiği — üstü "oynuyor" sayılır).
`FightMonitor` içinde: `CONTACT_IOU_THRESHOLD`, `CONTACT_DIST_FACTOR`,
`MIN_BOTH_ACTIVITY`, `FIGHT_DEBOUNCE_SECONDS`, `CALM_DEBOUNCE_SECONDS`.

## Bilinen sınırlamalar (bilerek basit tutuldu, test amaçlı)

- **Kimlik:** konum + kıyafet rengi + boy/en oranına dayanıyor. Hem aynı
  renk giyen HEM fiziksel olarak çok benzeyen biri karışabilir. Gerçek
  sistemde marker/bileklik veya derin ReID modeli daha sağlam.
- **Düşme:** basit sezgisel (gövde açısı + kutu oranı + hareketsizlik).
  Oyun aleti arkasında kalma (okluzyon), kameradan çok uzak/küçük çocuk,
  ya da yüzüstü değil de oturur/emekler pozisyonda kalma yanlış
  sonuç verebilir. "1 dk + hareketsizlik" filtresi yanlış alarmın çoğunu
  eler ama sahada eşik ayarı gerekir.
- **Kavga:** deneysel. Sezgisel kısım güreşme/şakalaşmayı gerçek kavga
  sanabilir. AI doğrulaması (VD-MIL) bunu azaltır ama o model de
  yetişkin güvenlik-kamerası verisiyle eğitilmiş — çocuk oyun alanında
  ve çocuk bedeninde doğruluğu düşebilir. Sağlam sonuç için `dataset/`
  klasörüne kendi kliplerinizi toplayıp ince ayar (fine-tune) gerekir.
- **VD-MIL modeli CC BY-NC 4.0** — ticari üründe kullanılamaz, sadece test.
- Tek kamera, tek PC. Çoklu kamera ve mobil bildirim (push) kapsam dışı.

## Sırada ne var?

- [x] Düşme tespiti (pose + yerde kalma sayacı)
- [x] Kavga / itiş-kakış tespiti (sezgisel + opsiyonel AI doğrulama)
- [x] Eğitim klibi kaydetme (`dataset/`)
- [ ] Kendi kliplerinle kavga modelini ince ayar (ticari-uygun model)
- [ ] Mobil bildirim (FCM push) — mobil uygulamayla entegrasyon
- [ ] Marker/bileklik ile daha sağlam kimlik takibi
- [ ] Çoklu kamera desteği
