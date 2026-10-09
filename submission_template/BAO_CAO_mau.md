# Báo cáo lab: chọn tracker cho 5 video

**Nhóm:** Nguyễn Đức Đông (2A202602367) **Thành viên:** Nguyễn Đức Đông

Detector cố định: `yolo26n.pt`, ảnh 640 px, Re-ID `osnet_x0_25_msmt17`. Không đổi các mục này trong bài nộp chính.

Môi trường chạy: Google Colab, GPU Tesla T4, Python 3.13, `boxmot==10.0.42`, notebook [`Lab22_Tracking_Colab.ipynb`](../Lab22_Tracking_Colab.ipynb). Bảng quét đầy đủ nằm trong [`ket_qua/`](../ket_qua/), file nộp trong `runs/nop_bai/`.

Cách thử: mỗi video chạy cả 5 tracker ở `conf=0.3, iou=0.5`, rồi lấy tracker tốt nhất và đổi **một** tham số mỗi lượt (`conf` 0.15 / 0.5, `iou` 0.4 / 0.7). `video_1` chọn theo HOTA (chạy đủ 600 frame). `video_2`–`video_5` không có nhãn nên so trên 300 frame đầu bằng thống kê track (số ID trên 100 hộp càng thấp = càng ít đứt/tạo ID mới; `short_frac` = tỉ lệ track ngắn dưới 10 frame) cùng với xem ảnh có vẽ ID (`ket_qua/video_N_frames.jpg`).

## 1. Cấu hình đã chọn

| Video | Tracker | conf | iou | Quan sát khi xem video | Đã thử nhưng loại |
|---|---|---|---|---|---|
| video_1 (quảng trường, tĩnh, ban ngày) | botsort | 0.3 | 0.7 | Người ở gần giữ được một màu ID suốt đoạn đi ngang; lỗi chính là bỏ sót người nhỏ ở xa (DetRe chỉ 19%) chứ không phải đổi ID (AssA 49). Có vài lần ID nhảy khi nhóm người đi sát nhau ở tiền cảnh. | `bytetrack` 0.3/0.5: ít đổi ID nhất (12) nhưng bỏ sót nhiều hơn, HOTA 26.9. `strongsort`: nhiều ID nhất (85), một nửa track ngắn < 10 frame. `conf=0.5`: HOTA tụt còn 27.2 vì mất người xa. |
| video_2 (phố đêm, tĩnh, rất đông) | botsort | 0.15 | 0.5 | Camera đứng yên nên track của người đã bắt được rất dài (TB 223 frame, chỉ 6.5% track ngắn). Nhưng YOLO nano bỏ sót rất nhiều người nhỏ và tối ở xa, chỉ khoảng 13 hộp mỗi frame trong một cảnh đông hơn nhiều. Hạ `conf` xuống 0.15 bắt thêm người mà không làm tăng số ID. | `bytetrack`: số ID trên 100 hộp thấp nhất nhưng bắt ít người hơn ~15% (9.8 so với 11.7 hộp/frame). `conf=0.5`: mất gần 1/3 số hộp. `ocsort`/`deepocsort`: nhiều ID hơn mà không bắt thêm người. |
| video_3 (camera di động, ảnh nhỏ) | ocsort | 0.3 | 0.5 | Camera đi theo dòng người, người ở gần rất to và bị cắt khung, người xa nhỏ. ID nhảy nhiều (137 ID / 837 frame, 34% track ngắn) vì khung hình chậm làm vị trí giữa hai frame chênh lớn. OC-SORT bù chuyển động tốt hơn một chút. | Ba tracker có Re-ID (`botsort`/`strongsort`/`deepocsort`) không giảm đổi ID mà chậm gấp đôi. `conf=0.15`: thêm 60% ID, nhiều hộp chập chờn. `bytetrack`: bắt ít người nhất. |
| video_4 (trong nhà, camera di chuyển) | deepocsort | 0.3 | 0.5 | Camera tiến tới trong hành lang sáng, có phản chiếu trên sàn và kính. Re-ID giúp giữ ID khi người lớn dần khi đến gần camera; DeepOCSORT giữ ID trung bình 78 frame và ít ID nhất trong nhóm tracker có độ phủ tương đương. | `conf=0.15`: số ID gấp đôi (53), có hộp trên bóng phản chiếu. `iou=0.7`: thêm ID mà không thêm người. `bytetrack`: ít ID nhưng bắt ít người hơn ~15%. |
| video_5 (trên xe bus, giao lộ đông) | botsort | 0.15 | 0.5 | Xe rung và rẽ nên cả khung cảnh trượt mạnh; người đi bộ phần lớn nhỏ và ở xa, nhiều đoạn không có ai được bắt. BoT-SORT có bù chuyển động camera (CMC) nên ít đứt track hơn các tracker khác ở cùng độ phủ. | `strongsort`: nhiều ID nhất, 54% track ngắn. `conf=0.5`: chỉ còn 2.6 hộp/frame. `bytetrack`: ít ID nhưng bắt ít hơn ~28% số hộp. |

## 2. Số liệu video_1

Cấu hình nộp: `botsort`, `conf=0.3`, `iou=0.7`. Bảng do `scripts/evaluate_practice.py` in ra:

```
HOTA: nop_bai_video1-pedestrian    HOTA      DetA      AssA      DetRe     DetPr     AssRe     AssPr     LocA
video_1                            29.969    18.408    49.061    19.176    74.385    52.38     80.959    83.019
CLEAR: nop_bai_video1-pedestrian   MOTA      MOTP      MODA      CLR_Re    CLR_Pr    CLR_TP    CLR_FN    CLR_FP    IDSW   Frag
video_1                            19.025    80.817    19.202    22.491    87.244    4179      14402     611       33     105
Identity: nop_bai_video1-pedestrian IDF1     IDR       IDP       IDTP      IDFN      IDFP
video_1                            29.703    18.68     72.463    3471      15110     1319
Count: nop_bai_video1-pedestrian   Dets      GT_Dets   IDs       GT_IDs
video_1                            4790      18581     55        62
```

So sánh các lượt thử trên video_1 (đủ 600 frame):

| Tracker | conf | iou | HOTA | MOTA | IDF1 | IDSW | FP | FN |
|---|---|---|---|---|---|---|---|---|
| bytetrack | 0.3 | 0.5 | 26.91 | 17.29 | 25.71 | 12 | 107 | 15249 |
| ocsort | 0.3 | 0.5 | 27.46 | 19.81 | 28.73 | 42 | 253 | 14605 |
| botsort | 0.3 | 0.5 | 29.46 | 19.81 | 29.35 | 25 | 337 | 14538 |
| strongsort | 0.3 | 0.5 | 28.66 | 19.70 | 29.85 | 41 | 231 | 14649 |
| deepocsort | 0.3 | 0.5 | 27.38 | 19.76 | 27.80 | 51 | 248 | 14610 |
| botsort | 0.15 | 0.5 | 29.34 | **20.73** | 29.56 | 27 | 505 | 14197 |
| botsort | 0.5 | 0.5 | 27.17 | 15.25 | 24.56 | **10** | 229 | 15508 |
| botsort | 0.3 | 0.4 | 29.32 | 19.47 | **29.82** | 19 | 195 | 14750 |
| **botsort** | **0.3** | **0.7** | **29.97** | 19.03 | 29.70 | 33 | 611 | 14402 |

`video_2` đến `video_5` không có nhãn trong gói lab. Không điền số cho các video đó.

## 3. Phân tích

**video_1 (có nhãn).** BoT-SORT có HOTA cao nhất (29.5 → 30.0 khi `iou=0.7`) vì nó vừa bắt được nhiều người hơn ByteTrack (FN 14.4k so với 15.2k) vừa giữ AssA ngang ByteTrack (~49). ByteTrack có ít lần đổi ID nhất (12) và MOTA thấp nhất, đúng như bài ôn: MOTA bị kéo xuống vì bỏ sót, không phải vì đổi ID. Điểm thấp nói chung đến từ detector: DetRe chỉ 19%, tức YOLO nano ở 640 px bỏ sót hơn 3/4 số người nhỏ ở xa; còn khi đã bắt được người thì tracker giữ ID khá tốt (AssPr 81%). Vì camera tĩnh và ban ngày, ngoại hình (Re-ID) và bù chuyển động camera của BoT-SORT giúp nối lại người sau khi bị che ngắn, nên tốt hơn hai tracker chỉ dùng chuyển động.

**video_2 (phố đêm, chỉ xem bằng mắt).** Camera đứng yên trên cao nên chuyển động trong ảnh đều; khi đã bắt được người, mọi tracker đều giữ ID dài. Khác biệt chính nằm ở việc bắt được bao nhiêu người: ảnh tối và người nhỏ nên `conf=0.15` bắt thêm khoảng 12% hộp mà số ID trên 100 hộp còn giảm (0.91 → 0.74), tức hộp mới là người thật đi liên tục chứ không phải nhiễu nhấp nháy. BoT-SORT được chọn thay cho ByteTrack vì ByteTrack bỏ qua nhiều người tối hơn.

**video_3 (camera di chuyển, ít frame/giây).** Đây là cảnh khó nhất: vị trí người nhảy xa giữa hai frame và người ở gần bị cắt khung, nên Kalman dự đoán sai và ID đổi liên tục. Re-ID không giúp vì người gần camera đổi hình dạng rất nhanh; OC-SORT với bước sửa quán tính (observation-centric) cho ít ID trên 100 hộp nhất và chạy nhanh gấp đôi, nên hợp cảnh này hơn.

**video_4 (trong nhà, camera tiến tới).** Ánh sáng tốt, người rõ, nhưng kích thước người thay đổi nhanh khi camera tiến lại gần. DeepOCSORT kết hợp chuyển động kiểu OC-SORT với ngoại hình nên giữ ID tốt mà không tạo thêm ID như StrongSORT. Hạ `conf` làm xuất hiện hộp trên bóng phản chiếu ở sàn/kính, nên giữ `conf=0.3`.

## 4. Nếu có thêm thời gian

Thử detector lớn hơn hoặc `imgsz` 1280 như một phần mở rộng (ngoài bài nộp chính) vì lỗi chính ở cả 5 video là bỏ sót người nhỏ, không phải đổi ID. Xem từng frame lỗi trong `video_3`/`video_5` và quét `conf` mịn hơn (0.2–0.25) cho hai cảnh đêm và rung lắc.
