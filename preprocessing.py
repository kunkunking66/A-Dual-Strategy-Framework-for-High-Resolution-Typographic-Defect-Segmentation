from pathlib import Path

import cv2
import numpy as np
import os


def imread_any(p):
    """讀取可能包含非 ASCII 字元路徑的圖片"""
    a = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(a, cv2.IMREAD_COLOR)


def odd(n): return n if n % 2 == 1 else n + 1


# ==================================================================
# 以下函式完全維持您提供的原始版本，未做任何修改
# ==================================================================、
def brighten_gray_to_white_sparkle_safe(
        img_bgr,
        chroma_thr=14,  # 判定“近灰”的色度阈
        L_low=40, L_high=210,  # 只处理中等亮度的灰
        stroke_px=11,  # ≈ 笔画宽度
        tophat_thr=12,  # Top-hat 响应阈（略高一点抗亮斑）
        prefilter="median",  # "median" 或 "bilateral" 或 None
        speckle_px=3,  # 认为亮斑半径≤3px
        elong_min=1.6,  # 细长度阈（minAreaRect 长/短 边比）
        circ_max=0.75,  # 圆度上限：4πA/P^2，越圆越接近1
        debug_dir=None
):
    # ---------- 1) BGR->Lab & 选择近灰 ----------
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    a0 = a.astype(np.int16) - 128
    b0 = b.astype(np.int16) - 128
    chroma = np.sqrt(a0 * a0 + b0 * b0).astype(np.float32)
    gray_like = (chroma <= chroma_thr)
    mid_L = (L >= L_low) & (L <= L_high)

    # ---------- 2) 预抑制微亮点 ----------
    Lp = L.copy()
    if prefilter == "median":
        Lp = cv2.medianBlur(Lp, 3)  # 对1~2px亮点非常有效
    elif prefilter == "bilateral":
        Lp = cv2.bilateralFilter(Lp, d=5, sigmaColor=15, sigmaSpace=15)

    # ---------- 3) Top-hat找“比周围更亮”的细结构 ----------
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (stroke_px, stroke_px))
    opened = cv2.morphologyEx(Lp, cv2.MORPH_OPEN, se)
    tophat = cv2.subtract(Lp, opened)
    cand = (tophat >= tophat_thr) & gray_like & mid_L  # 初筛

    # === 放到 Top-hat 阶段后 ===
    # 原有:
    # tophat = Lp - opened
    # cand   = (tophat >= tophat_thr) & gray_like & mid_L

    # 改成“双阈值 + 连通性”
    T_high = tophat_thr  # 比如 12
    T_low = max(4, T_high // 2)  # 比如 6

    seeds = (tophat >= T_high) & gray_like & mid_L
    weak = (tophat >= T_low) & gray_like & mid_L

    seeds_u8 = (seeds.astype(np.uint8) * 255)
    weak_u8 = (weak.astype(np.uint8) * 255)

    # 形态学重建：从 seeds 出发，在 weak 掩膜里“泛洪”生长
    def morph_reconstruct(seeds, mask, se=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))):
        prev = np.zeros_like(seeds)
        cur = seeds.copy()
        while True:
            dil = cv2.dilate(cur, se, iterations=1)
            nxt = cv2.bitwise_and(dil, mask)
            if np.array_equal(nxt, cur): break
            cur = nxt
        return cur

    cand_u8 = morph_reconstruct(seeds_u8, weak_u8)  # ← 这就是迟滞筛过的候选
    # 后面形状筛选（elong/circ/area）就用 cand_u8 替代原先的 cand_u8

    # === 额外：灰域“黑顶帽”补偿（抓更暗的灰字） ===
    # 只在近灰区域做，避免误伤彩色底
    black = cv2.morphologyEx(Lp, cv2.MORPH_BLACKHAT, se)  # 暗笔画响应
    Bh = 8  # 黑顶帽阈值，可从 6~10 试
    dark_seeds = ((black >= Bh) & gray_like).astype(np.uint8) * 255

    # 把“暗笔画”的候选也纳入（同样可做一次重建防噪）
    dark_u8 = morph_reconstruct(dark_seeds, (gray_like.astype(np.uint8) * 255))
    # 合并亮/暗两路候选
    cand_u8 = cv2.bitwise_or(cand_u8, dark_u8)

    # === 放到 Top-hat 阶段后 ===
    # 原有:
    # tophat = Lp - opened
    # cand   = (tophat >= tophat_thr) & gray_like & mid_L

    # 改成“双阈值 + 连通性”
    T_high = tophat_thr  # 比如 12
    T_low = max(4, T_high // 2)  # 比如 6

    seeds = (tophat >= T_high) & gray_like & mid_L
    weak = (tophat >= T_low) & gray_like & mid_L

    seeds_u8 = (seeds.astype(np.uint8) * 255)
    weak_u8 = (weak.astype(np.uint8) * 255)

    # 形态学重建：从 seeds 出发，在 weak 掩膜里“泛洪”生长
    def morph_reconstruct(seeds, mask, se=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))):
        prev = np.zeros_like(seeds)
        cur = seeds.copy()
        while True:
            dil = cv2.dilate(cur, se, iterations=1)
            nxt = cv2.bitwise_and(dil, mask)
            if np.array_equal(nxt, cur): break
            cur = nxt
        return cur

    cand_u8 = morph_reconstruct(seeds_u8, weak_u8)  # ← 这就是迟滞筛过的候选
    # 后面形状筛选（elong/circ/area）就用 cand_u8 替代原先的 cand_u8

    # === 额外：灰域“黑顶帽”补偿（抓更暗的灰字） ===
    # 只在近灰区域做，避免误伤彩色底
    black = cv2.morphologyEx(Lp, cv2.MORPH_BLACKHAT, se)  # 暗笔画响应
    Bh = 8  # 黑顶帽阈值，可从 6~10 试
    dark_seeds = ((black >= Bh) & gray_like).astype(np.uint8) * 255

    # 把“暗笔画”的候选也纳入（同样可做一次重建防噪）
    dark_u8 = morph_reconstruct(dark_seeds, (gray_like.astype(np.uint8) * 255))
    # 合并亮/暗两路候选
    cand_u8 = cv2.bitwise_or(cand_u8, dark_u8)

    # ---------- 4) 形状筛选：像笔画的才保留 ----------
    cand_u8 = (cand.astype(np.uint8) * 255)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cand_u8, connectivity=8)

    keep = np.zeros_like(cand_u8)
    # 面积阈：比亮斑稍大一些，避免误剔细字
    area_min = max(25, int(0.6 * stroke_px * 0.6 * stroke_px))
    for i in range(1, num):
        A = int(stats[i, cv2.CC_STAT_AREA])
        if A < area_min:
            continue
        mask_i = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        P = max(1.0, cv2.arcLength(c, True))
        circ = 4.0 * np.pi * A / (P * P)  # 圆度：亮斑≈1，笔画更小
        rect = cv2.minAreaRect(c)
        (w, h) = rect[1]
        if w == 0 or h == 0:
            continue
        elong = max(w, h) / max(1e-6, min(w, h))

        # 保留规则：
        #  - 够细长（像笔画），或
        #  - 面积已经足够大（明显不是点），且不太圆
        if (elong >= elong_min) or (A >= (stroke_px * stroke_px) and circ <= circ_max):
            keep[labels == i] = 255

    # 适度闭合一下笔画
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # ---------- 5) 只对白/灰的笔画区提白 ----------
    L2 = L.copy()
    L2[keep > 0] = 255
    lab2 = cv2.merge([L2, a, b])
    whitened = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # ---------- 6) Otsu 得到最终二值 ----------
    _, bin_mask = cv2.threshold(L2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "01_L.png"), L)
        vis_gray = (np.clip(chroma, 0, chroma_thr) / chroma_thr * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, "02_grayness.png"), 255 - vis_gray)
        cv2.imwrite(os.path.join(debug_dir, "03_tophat.png"), tophat)
        cv2.imwrite(os.path.join(debug_dir, "04_cand.png"), cand_u8)
        cv2.imwrite(os.path.join(debug_dir, "05_keep.png"), keep)
        cv2.imwrite(os.path.join(debug_dir, "06_binary.png"), bin_mask)

    return whitened, bin_mask


# ==================================================================
# ===== 主程式執行區塊 (批次處理版本) =====
# ==================================================================
if __name__ == "__main__":
    # 1. 設定來源與輸出資料夾
    input_folder = r"E:\SURF\pic"
    output_folder = "out_bh"

    # 2. 檢查來源資料夾是否存在
    if not os.path.isdir(input_folder):
        print(f"錯誤：來源資料夾不存在 -> {input_folder}")
    else:
        # 建立輸出資料夾 (如果不存在)
        os.makedirs(output_folder, exist_ok=True)
        print(f"所有處理結果將儲存於 '{output_folder}' 資料夾中。")

        # 3. 遍歷來源資料夾中的所有檔案
        for filename in os.listdir(input_folder):
            # 判斷是否為支援的圖片格式
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(input_folder, filename)
                print(f"\n正在處理: {input_path}")

                # 讀取圖片
                img = imread_any(input_path)
                if img is None:
                    print(f"  -> 讀取失敗，跳過此檔案。")
                    continue

                # 執行二值化處理
                _, binary_mask = brighten_gray_to_white_sparkle_safe(
                    img,
                    chroma_thr=14, stroke_px=11, tophat_thr=12,
                    prefilter="median", elong_min=1.6, circ_max=0.75,
                    debug_dir=None  # 不產生除錯檔案
                )

                # 對二值化結果進行顏色反轉
                inverted_mask = cv2.bitwise_not(binary_mask)

                # 準備輸出路徑和檔名
                base_name = Path(filename).stem
                output_filename = f"{base_name}_binary_inverted.png"
                output_path = os.path.join(output_folder, output_filename)

                # 儲存最終的反轉圖片
                try:
                    # 使用 imencode/tofile 確保路徑與檔名無誤
                    cv2.imencode(".png", inverted_mask)[1].tofile(output_path)
                    print(f"  -> 處理完成，已儲存至: {output_path}")
                except Exception as e:
                    print(f"  -> 儲存失敗: {e}")

        print("\n所有圖片處理完畢。")
