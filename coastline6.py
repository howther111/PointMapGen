from pathlib import Path
import math
import random

import cv2
import numpy as np


# =========================================================
# 設定
# =========================================================

INPUT_FILE = Path("input.png")
OUTPUT_FILE = Path("output_coastline.png")


# ---------------------------------------------------------
# ノード間の接続確率
# ---------------------------------------------------------

# 陸続きになる確率が50％になる距離
# 画像の短辺の40％
CONNECTION_CENTER_RATIO = 0.40

# シグモイド曲線の傾き
# 大きいほど30～50％付近で急激に変化する
CONNECTION_STEEPNESS = 30.0


# ---------------------------------------------------------
# 陸地生成
# ---------------------------------------------------------

# ノードから周囲へ広げる陸地の半径
NODE_LAND_RADIUS = 140

# ノード間を結ぶ陸地の幅
LAND_BRIDGE_WIDTH = 180

# ノード円の検出条件
NODE_MIN_AREA = 800
NODE_MAX_AREA = 10000
NODE_MIN_CIRCULARITY = 0.70


# ---------------------------------------------------------
# 色設定
# OpenCVではBGR順
# ---------------------------------------------------------

# 海の色
SEA_COLOR = (235, 205, 155)

# 陸地の色
LAND_COLOR = (250, 250, 245)

# 海岸線の色
COAST_COLOR = (40, 40, 40)

# 海岸線の太さ
COAST_THICKNESS = 3


# ---------------------------------------------------------
# 海岸線生成
# ---------------------------------------------------------

# 陸地の形を滑らかにする強さ
LAND_EXPANSION = 150

# 海岸線の細かさ
# 大きいほど大きな周期の凹凸になる
COAST_DETAIL = 18

# 海岸線の揺れ幅
COAST_ROUGHNESS = 20

# 小さすぎる陸地を削除する面積
MIN_LAND_AREA = 2000

# 元画像の白と判定する下限値
WHITE_THRESHOLD = 245

# 毎回同じ結果にするための乱数シード
RANDOM_SEED = 12345


# =========================================================
# 補助関数
# =========================================================

def make_odd(value: int) -> int:
    """
    OpenCVのカーネルサイズに使用できるよう、
    1以上の奇数へ変換する。
    """

    value = max(1, int(value))

    if value % 2 == 0:
        value += 1

    return value


def land_connection_probability(
    distance: float,
    image_size: float
) -> float:
    """
    ノード間の距離に応じて、
    陸続きになる確率を0.0～1.0で返す。

    距離比が小さいほど確率が高く、
    距離比が大きいほど確率が低くなる。

    標準設定の場合：
        画像短辺の30％：陸続き約95％
        画像短辺の40％：陸続き50％
        画像短辺の50％：陸続き約5％
    """

    if image_size <= 0:
        return 0.0

    distance_ratio = distance / image_size

    exponent = CONNECTION_STEEPNESS * (
        distance_ratio - CONNECTION_CENTER_RATIO
    )

    # 非常に大きな値によるオーバーフロー対策
    exponent = max(-700.0, min(700.0, exponent))

    probability = 1.0 / (1.0 + math.exp(exponent))

    return probability


def detect_node_centers(
    image: np.ndarray
) -> list[tuple[int, int]]:
    """
    画像内の円形ノードを検出し、
    中心座標の一覧を返す。

    文字枠や接続線は、面積・円形度・縦横比によって除外する。
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

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

    centers: list[tuple[int, int]] = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < NODE_MIN_AREA:
            continue

        if area > NODE_MAX_AREA:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        circularity = (
            4.0
            * math.pi
            * area
            / (perimeter * perimeter)
        )

        if circularity < NODE_MIN_CIRCULARITY:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        if height <= 0:
            continue

        aspect_ratio = width / height

        # 極端な横長・縦長の輪郭を除外
        if not 0.75 <= aspect_ratio <= 1.25:
            continue

        center_x = x + width // 2
        center_y = y + height // 2

        # 同じ円の内周・外周が重複検出されるのを防ぐ
        duplicate = False

        for existing_x, existing_y in centers:
            center_distance = math.hypot(
                center_x - existing_x,
                center_y - existing_y
            )

            if center_distance < 20:
                duplicate = True
                break

        if not duplicate:
            centers.append(
                (center_x, center_y)
            )

    return centers


def remove_small_regions(
    mask: np.ndarray,
    min_area: int
) -> np.ndarray:
    """
    指定面積より小さな陸地領域を削除する。
    """

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    result = np.zeros_like(mask)

    for label_number in range(1, count):
        area = stats[
            label_number,
            cv2.CC_STAT_AREA
        ]

        if area >= min_area:
            result[labels == label_number] = 255

    return result


def add_smooth_noise(
    mask: np.ndarray,
    detail: int,
    roughness: int
) -> np.ndarray:
    """
    陸地マスクの境界に滑らかなノイズを加え、
    海岸線を自然な形にする。
    """

    height, width = mask.shape

    small_width = max(
        2,
        width // max(2, detail)
    )

    small_height = max(
        2,
        height // max(2, detail)
    )

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
        noise = (
            noise - noise_min
        ) / (
            noise_max - noise_min
        )

        noise = noise * 2.0 - 1.0

    else:
        noise = np.zeros_like(noise)

    # 陸地内部から境界までの距離
    inside_distance = cv2.distanceTransform(
        mask,
        cv2.DIST_L2,
        5
    )

    # 海側から境界までの距離
    outside_distance = cv2.distanceTransform(
        cv2.bitwise_not(mask),
        cv2.DIST_L2,
        5
    )

    signed_distance = (
        inside_distance - outside_distance
    )

    noisy_distance = (
        signed_distance
        + noise * roughness
    )

    result = np.where(
        noisy_distance >= 0,
        255,
        0
    ).astype(np.uint8)

    return result


def smooth_land_mask(
    mask: np.ndarray
) -> np.ndarray:
    """
    陸地の形をぼかして滑らかに整える。
    """

    blur_size = make_odd(
        max(
            9,
            LAND_EXPANSION // 5
        )
    )

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


def create_land_mask(
    image: np.ndarray
) -> np.ndarray:
    """
    円形ノードを基準に陸地を生成する。

    ノード間の距離が近いほど高確率で陸続きになり、
    遠いほど高確率で海によって分断される。
    """

    height, width = image.shape[:2]

    # 距離比を計算するときは画像の短辺を基準とする
    image_size = float(
        min(width, height)
    )

    node_centers = detect_node_centers(image)

    if not node_centers:
        raise RuntimeError(
            "円形ノードを検出できませんでした。"
            "NODE_MIN_AREA、NODE_MAX_AREA、"
            "NODE_MIN_CIRCULARITYを調整してください。"
        )

    print(
        f"検出したノード数: {len(node_centers)}"
    )

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

    # ノード間を確率的に陸地で接続する
    for first_index in range(
        len(node_centers)
    ):
        first_x, first_y = (
            node_centers[first_index]
        )

        for second_index in range(
            first_index + 1,
            len(node_centers)
        ):
            second_x, second_y = (
                node_centers[second_index]
            )

            distance = math.hypot(
                second_x - first_x,
                second_y - first_y
            )

            probability = (
                land_connection_probability(
                    distance,
                    image_size
                )
            )

            random_value = random.random()

            if random_value < probability:
                cv2.line(
                    land_base,
                    (first_x, first_y),
                    (second_x, second_y),
                    255,
                    thickness=LAND_BRIDGE_WIDTH,
                    lineType=cv2.LINE_AA
                )

                connection_result = "陸続き"

            else:
                connection_result = "海で分断"

            distance_ratio = (
                distance / image_size
            )

            print(
                f"ノード{first_index + 1}"
                f"－ノード{second_index + 1}: "
                f"距離={distance:.1f}px、"
                f"距離比={distance_ratio:.1%}、"
                f"陸続き確率={probability:.1%}、"
                f"結果={connection_result}"
            )

    # 陸地を滑らかにする
    land_mask = smooth_land_mask(
        land_base
    )

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
    元画像の白以外の部分だけを、
    海岸線付き背景へ重ねる。
    """

    gray = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2GRAY
    )

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

    alpha = (
        foreground_mask.astype(np.float32)
        / 255.0
    )

    alpha = alpha[:, :, np.newaxis]

    result = (
        original.astype(np.float32) * alpha
        + background.astype(np.float32)
        * (1.0 - alpha)
    )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


def draw_coastline(
    canvas: np.ndarray,
    land_mask: np.ndarray
) -> np.ndarray:
    """
    陸地の外周と、
    内海・湖などの内側輪郭を描画する。
    """

    contours, _ = cv2.findContours(
        land_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    cv2.drawContours(
        canvas,
        contours,
        -1,
        COAST_COLOR,
        COAST_THICKNESS,
        lineType=cv2.LINE_AA
    )

    return canvas


# =========================================================
# メイン処理
# =========================================================

def main() -> None:
    """
    input.pngを読み込み、
    海岸線付きのoutput_coastline.pngを生成する。
    """

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not INPUT_FILE.exists():
        print(
            f"入力画像が見つかりません: "
            f"{INPUT_FILE}"
        )

        print(
            "プログラムと同じフォルダに"
            "input.pngを置いてください。"
        )

        return

    original = cv2.imread(
        str(INPUT_FILE),
        cv2.IMREAD_COLOR
    )

    if original is None:
        print(
            "画像を読み込めませんでした。"
        )

        return

    height, width = original.shape[:2]

    print(
        f"画像サイズ: {width} × {height}px"
    )

    try:
        land_mask = create_land_mask(
            original
        )

    except RuntimeError as error:
        print(error)
        return

    # キャンバス全体を海で塗る
    canvas = np.full(
        (height, width, 3),
        SEA_COLOR,
        dtype=np.uint8
    )

    # 陸地部分を塗る
    canvas[land_mask > 0] = LAND_COLOR

    # 外側・内側の海岸線を描く
    canvas = draw_coastline(
        canvas,
        land_mask
    )

    # 元のポイントマップを重ねる
    result = overlay_original_map(
        canvas,
        original
    )

    success = cv2.imwrite(
        str(OUTPUT_FILE),
        result
    )

    if success:
        print(
            "海岸線付き画像を出力しました。"
        )

        print(
            f"出力先: "
            f"{OUTPUT_FILE.resolve()}"
        )

    else:
        print(
            "画像の保存に失敗しました。"
        )


if __name__ == "__main__":
    main()