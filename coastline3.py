from pathlib import Path
import random

import cv2
import numpy as np


# =========================================================
# 設定
# =========================================================

INPUT_FILE = Path("input.png")
OUTPUT_FILE = Path("output_coastline.png")

# 同じ陸地として接続するノード間の最大距離
MAX_LAND_NODE_DISTANCE = 600

# ノードから周囲へ広げる陸地の半径
NODE_LAND_RADIUS = 140

# ノード間を結ぶ陸地の幅
LAND_BRIDGE_WIDTH = 180

# ノード円の検出条件
NODE_MIN_AREA = 800
NODE_MAX_AREA = 10000
NODE_MIN_CIRCULARITY = 0.70

# 海の色：BGR形式
SEA_COLOR = (235, 205, 155)

# 陸地の色：BGR形式
LAND_COLOR = (250, 250, 245)

# 海岸線の色：BGR形式
COAST_COLOR = (40, 40, 40)

# 海岸線の太さ
COAST_THICKNESS = 3

# ポイントや線から陸地をどれくらい広げるか
# 大きくすると陸地が広くなる
LAND_EXPANSION = 150

# 陸地の隙間をつなぐ強さ
# 大きくすると離れた地点同士が陸地としてつながりやすい
LAND_CONNECTION = 170

# 海岸線の細かさ
# 大きいほど細かく入り組む
COAST_DETAIL = 18

# 海岸線の揺れ幅
COAST_ROUGHNESS = 20

# 小さすぎる陸地を削除する面積
MIN_LAND_AREA = 2000

# 元画像の白と判定する下限値
WHITE_THRESHOLD = 245

# 毎回同じ形にするための乱数シード
RANDOM_SEED = 12345


# =========================================================
# 補助関数
# =========================================================

import cv2
import numpy as np
import math


def detect_node_centers(image: np.ndarray) -> list[tuple[int, int]]:
    """
    画像内の円形ノードを検出し、中心座標の一覧を返す。
    文字枠や接続線は、円形度によって除外する。
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray,
        180,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    centers = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < NODE_MIN_AREA or area > NODE_MAX_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        circularity = 4.0 * math.pi * area / (perimeter * perimeter)

        if circularity < NODE_MIN_CIRCULARITY:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        # 極端な横長・縦長の輪郭を除外
        aspect_ratio = width / height

        if not 0.75 <= aspect_ratio <= 1.25:
            continue

        center_x = x + width // 2
        center_y = y + height // 2

        # 同じ円の内周・外周が重複検出されるのを防ぐ
        duplicate = False

        for existing_x, existing_y in centers:
            distance = math.hypot(
                center_x - existing_x,
                center_y - existing_y
            )

            if distance < 20:
                duplicate = True
                break

        if not duplicate:
            centers.append((center_x, center_y))

    return centers

def make_odd(value: int) -> int:
    """OpenCV用に奇数へ変換する。"""
    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    return value


def remove_small_regions(mask: np.ndarray, min_area: int) -> np.ndarray:
    """小さな白領域を削除する。"""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    result = np.zeros_like(mask)

    for label_number in range(1, count):
        area = stats[label_number, cv2.CC_STAT_AREA]

        if area >= min_area:
            result[labels == label_number] = 255

    return result


def add_smooth_noise(
    mask: np.ndarray,
    detail: int,
    roughness: int
) -> np.ndarray:
    """
    マスクの境界に滑らかなノイズを加え、
    海岸線を自然な形にする。
    """
    height, width = mask.shape

    small_width = max(2, width // max(2, detail))
    small_height = max(2, height // max(2, detail))

    noise_small = np.random.normal(
        0,
        1,
        (small_height, small_width)
    ).astype(np.float32)

    noise = cv2.resize(
        noise_small,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

    noise = cv2.GaussianBlur(
        noise,
        (0, 0),
        sigmaX=max(1, detail / 4)
    )

    noise_min = float(noise.min())
    noise_max = float(noise.max())

    if noise_max > noise_min:
        noise = (noise - noise_min) / (noise_max - noise_min)
        noise = noise * 2.0 - 1.0
    else:
        noise = np.zeros_like(noise)

    # マスク内外の距離を計算する
    inside_distance = cv2.distanceTransform(
        mask,
        cv2.DIST_L2,
        5
    )

    outside_distance = cv2.distanceTransform(
        cv2.bitwise_not(mask),
        cv2.DIST_L2,
        5
    )

    signed_distance = inside_distance - outside_distance

    noisy_distance = signed_distance + noise * roughness

    result = np.where(noisy_distance >= 0, 255, 0).astype(np.uint8)

    return result


def smooth_land_mask(mask: np.ndarray) -> np.ndarray:
    """陸地の形を滑らかに整える。"""
    blur_size = make_odd(max(9, LAND_EXPANSION // 5))

    blurred = cv2.GaussianBlur(
        mask,
        (blur_size, blur_size),
        0
    )

    _, result = cv2.threshold(
        blurred,
        100,
        255,
        cv2.THRESH_BINARY
    )

    return result


def create_land_mask(image: np.ndarray) -> np.ndarray:
    """
    円形ノードを基準に陸地を生成する。

    1200ピクセル未満の距離にあるノード間だけ陸地で接続し、
    1200ピクセル以上離れたノード間は海として分断する。
    """

    height, width = image.shape[:2]

    MAX_LAND_NODE_DISTANCE = int(
        min(image.shape[:2]) * 0.5
    )

    node_centers = detect_node_centers(image)

    if not node_centers:
        raise RuntimeError(
            "円形ノードを検出できませんでした。"
            "NODE_MIN_AREAなどの設定を調整してください。"
        )

    print(f"検出したノード数: {len(node_centers)}")

    # 陸地の元になるマスク
    land_base = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    # 各ノードの周囲を陸地として描画
    for center_x, center_y in node_centers:
        cv2.circle(
            land_base,
            (center_x, center_y),
            NODE_LAND_RADIUS,
            255,
            thickness=-1,
            lineType=cv2.LINE_AA
        )

    # 1200ピクセル未満のノード間だけを接続する
    for first_index in range(len(node_centers)):
        first_x, first_y = node_centers[first_index]

        for second_index in range(
            first_index + 1,
            len(node_centers)
        ):
            second_x, second_y = node_centers[second_index]

            distance = math.hypot(
                second_x - first_x,
                second_y - first_y
            )

            if distance < MAX_LAND_NODE_DISTANCE:
                cv2.line(
                    land_base,
                    (first_x, first_y),
                    (second_x, second_y),
                    255,
                    thickness=LAND_BRIDGE_WIDTH,
                    lineType=cv2.LINE_AA
                )

    # 陸地を滑らかにする
    land_mask = smooth_land_mask(land_base)

    # 海岸線に自然な凹凸を加える
    land_mask = add_smooth_noise(
        land_mask,
        COAST_DETAIL,
        COAST_ROUGHNESS
    )

    # 小さすぎる陸地を削除
    land_mask = remove_small_regions(
        land_mask,
        MIN_LAND_AREA
    )

    return land_mask


def overlay_original_map(
    background: np.ndarray,
    original: np.ndarray
) -> np.ndarray:
    """
    元画像の白以外の部分だけを背景へ重ねる。
    """
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    foreground_mask = np.where(
        gray < WHITE_THRESHOLD,
        255,
        0
    ).astype(np.uint8)

    foreground_mask = cv2.GaussianBlur(
        foreground_mask,
        (3, 3),
        0
    )

    alpha = foreground_mask.astype(np.float32) / 255.0
    alpha = alpha[:, :, np.newaxis]

    result = (
        original.astype(np.float32) * alpha
        + background.astype(np.float32) * (1.0 - alpha)
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def draw_coastline(
    canvas: np.ndarray,
    land_mask: np.ndarray
) -> np.ndarray:
    """
    陸地マスクの外周と、内海・湖などの内側輪郭を描画する。
    """

    contours, _ = cv2.findContours(
        land_mask,
        cv2.RETR_LIST,          # 外側・内側を含むすべての輪郭を取得
        cv2.CHAIN_APPROX_NONE
    )

    cv2.drawContours(
        canvas,
        contours,
        -1,                     # すべての輪郭を描画
        COAST_COLOR,
        COAST_THICKNESS,
        lineType=cv2.LINE_AA
    )

    return canvas


# =========================================================
# メイン処理
# =========================================================

def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not INPUT_FILE.exists():
        print(f"入力画像が見つかりません: {INPUT_FILE}")
        print("プログラムと同じフォルダに input.png を置いてください。")
        return

    original = cv2.imread(
        str(INPUT_FILE),
        cv2.IMREAD_COLOR
    )

    if original is None:
        print("画像を読み込めませんでした。")
        return

    height, width = original.shape[:2]

    land_mask = create_land_mask(original)

    # 海でキャンバス全体を塗る
    canvas = np.full(
        (height, width, 3),
        SEA_COLOR,
        dtype=np.uint8
    )

    # 陸地部分を塗る
    canvas[land_mask > 0] = LAND_COLOR

    # 海岸線を描く
    canvas = draw_coastline(canvas, land_mask)

    # 元のポイントマップを重ねる
    result = overlay_original_map(canvas, original)

    success = cv2.imwrite(
        str(OUTPUT_FILE),
        result
    )

    if success:
        print("海岸線付き画像を出力しました。")
        print(f"出力先: {OUTPUT_FILE.resolve()}")
    else:
        print("画像の保存に失敗しました。")


if __name__ == "__main__":
    main()