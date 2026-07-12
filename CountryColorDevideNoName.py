from pathlib import Path
import random

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


# =========================================================
# 設定
# =========================================================

INPUT_FILE = Path("input_coastline.png")
OUTPUT_FILE = Path("output_country.png")

# 同じ結果を再現するための乱数シード
RANDOM_SEED = 42

# 1か国が持つノード数
MIN_NODES_PER_COUNTRY = 1
MAX_NODES_PER_COUNTRY = 10

# 国の色の透明度
# 0.0で元画像、1.0で完全な塗りつぶし
COLOR_ALPHA = 0.63

# 国境線の太さ
BORDER_WIDTH = 2

# 国境線の色：BGR
BORDER_COLOR = (245, 245, 245)

# 元画像の道路・文字・ノードなどを復元する際の明るさ基準
# この値より暗い画素は元画像をそのまま残す
DARK_PIXEL_THRESHOLD = 205

# 海の色（BGR）
SEA_COLOR = (235, 205, 155)

# 海色判定の許容誤差
SEA_COLOR_TOLERANCE = 25

# ノード円の検出設定
HOUGH_DP = 1.2
HOUGH_MIN_DISTANCE_RATIO = 0.025
NODE_MIN_RADIUS_RATIO = 0.010
NODE_MAX_RADIUS_RATIO = 0.022

# ノード検出結果を確認する画像を出力するか
OUTPUT_NODE_DEBUG_IMAGE = False
NODE_DEBUG_FILE = Path("ノード検出結果.png")


# =========================================================
# 国の色
# OpenCVではBGR順
# =========================================================

COUNTRY_COLORS = [
    (180, 220, 255),   # 薄い黄
    (190, 225, 180),   # 薄い緑
    (205, 190, 240),   # 薄い紫
    (220, 220, 170),   # 薄い水色
    (180, 190, 245),   # 薄い赤
    (180, 215, 245),   # 薄いオレンジ
    (230, 200, 170),   # 薄い青
    (200, 230, 230),   # 薄いクリーム
    (220, 185, 215),   # 薄いピンク紫
    (190, 225, 245),   # 薄い桃
    (210, 210, 210),   # 薄い灰色
    (185, 235, 215),   # ミント
    (225, 205, 175),   # 薄い青紫
    (195, 220, 235),   # 薄いベージュ
    (215, 235, 185),   # 黄緑
    (235, 195, 195),   # 薄い紫青
    (175, 215, 235),   # 黄橙
    (210, 190, 230),   # 薄い赤紫
    (190, 235, 225),   # 緑水色
    (230, 215, 185),   # 薄い空色
]


# =========================================================
# 画像読み込み
# =========================================================

def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(
            f"画像を読み込めませんでした。\n"
            f"ファイル名と配置場所を確認してください。\n"
            f"対象ファイル：{path}"
        )

    return image


# =========================================================
# 海・陸地マスクの作成
# =========================================================

def create_land_mask(image: np.ndarray) -> np.ndarray:
    """
    海色(BGR=(235,205,155))を固定値として陸地マスクを作成する。
    """

    image_float = image.astype(np.float32)

    sea_color = np.array(SEA_COLOR, dtype=np.float32)

    # 色距離
    distance = np.linalg.norm(
        image_float - sea_color[None, None, :],
        axis=2
    )

    sea_mask = distance <= SEA_COLOR_TOLERANCE

    kernel = np.ones((5, 5), np.uint8)

    sea_mask = cv2.morphologyEx(
        sea_mask.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        kernel
    )

    sea_mask = cv2.morphologyEx(
        sea_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    sea_mask = sea_mask > 0

    land_mask = ~sea_mask

    # 小さなノイズ除去
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        land_mask.astype(np.uint8),
        connectivity=8
    )

    cleaned = np.zeros_like(land_mask)

    minimum_area = image.shape[0] * image.shape[1] * 0.001

    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            cleaned[labels == label] = True

    return cleaned


# =========================================================
# ノード円の検出
# =========================================================

def detect_nodes(image: np.ndarray, land_mask: np.ndarray) -> list[tuple[int, int]]:
    height, width = image.shape[:2]
    base_size = min(height, width)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)

    minimum_distance = max(
        20,
        int(base_size * HOUGH_MIN_DISTANCE_RATIO)
    )

    minimum_radius = max(
        7,
        int(base_size * NODE_MIN_RADIUS_RATIO)
    )

    maximum_radius = max(
        minimum_radius + 2,
        int(base_size * NODE_MAX_RADIUS_RATIO)
    )

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=minimum_distance,
        param1=100,
        param2=28,
        minRadius=minimum_radius,
        maxRadius=maximum_radius
    )

    if circles is None:
        raise RuntimeError(
            "ノード円を検出できませんでした。\n"
            "NODE_MIN_RADIUS_RATIO、NODE_MAX_RADIUS_RATIO、"
            "またはHoughCirclesのparam2を調整してください。"
        )

    circles = np.round(circles[0]).astype(int)

    node_centers = []

    for x, y, radius in circles:
        if not (0 <= x < width and 0 <= y < height):
            continue

        # 陸地上にある円だけをノードとして採用
        if not land_mask[y, x]:
            continue

        node_centers.append((x, y))

    node_centers = remove_duplicate_nodes(
        node_centers,
        minimum_distance=max(10, minimum_distance // 2)
    )

    node_centers.sort(key=lambda point: (point[1], point[0]))

    if not node_centers:
        raise RuntimeError("有効なノードが見つかりませんでした。")

    return node_centers


def remove_duplicate_nodes(
    node_centers: list[tuple[int, int]],
    minimum_distance: float
) -> list[tuple[int, int]]:
    result = []

    for candidate in node_centers:
        candidate_array = np.array(candidate, dtype=np.float32)

        is_duplicate = False

        for registered in result:
            registered_array = np.array(registered, dtype=np.float32)

            distance = np.linalg.norm(candidate_array - registered_array)

            if distance < minimum_distance:
                is_duplicate = True
                break

        if not is_duplicate:
            result.append(candidate)

    return result


# =========================================================
# ノードを国ごとにグループ化
# =========================================================

def create_country_groups(
    node_centers: list[tuple[int, int]]
) -> list[list[int]]:
    """
    近いノードを順番に取り込みながら、
    1か国あたり1～10ノードになるようにグループ化する。
    """

    rng = random.Random(RANDOM_SEED)

    number_of_nodes = len(node_centers)
    unassigned = set(range(number_of_nodes))
    country_groups = []

    points = np.array(node_centers, dtype=np.float32)

    while unassigned:
        remaining_count = len(unassigned)

        target_size = decide_country_size(
            remaining_count=remaining_count,
            rng=rng
        )

        seed_index = choose_country_seed(
            unassigned=unassigned,
            points=points,
            rng=rng
        )

        group = [seed_index]
        unassigned.remove(seed_index)

        while len(group) < target_size and unassigned:
            next_index = find_nearest_node_to_group(
                group=group,
                unassigned=unassigned,
                points=points
            )

            group.append(next_index)
            unassigned.remove(next_index)

        country_groups.append(group)

    return country_groups


def decide_country_size(
    remaining_count: int,
    rng: random.Random
) -> int:
    maximum_size = min(MAX_NODES_PER_COUNTRY, remaining_count)
    minimum_size = min(MIN_NODES_PER_COUNTRY, maximum_size)

    if remaining_count <= MAX_NODES_PER_COUNTRY:
        return remaining_count

    size = rng.randint(minimum_size, maximum_size)

    nodes_after_creation = remaining_count - size

    # 端数として1個だけ残ることを少し避ける
    if (
        nodes_after_creation > 0
        and nodes_after_creation < MIN_NODES_PER_COUNTRY
    ):
        size = remaining_count - MIN_NODES_PER_COUNTRY

    return max(minimum_size, min(size, maximum_size))


def choose_country_seed(
    unassigned: set[int],
    points: np.ndarray,
    rng: random.Random
) -> int:
    """
    未割り当てノードのうち、外側寄りのノードを優先して開始点にする。
    外周から国を作ることで、比較的まとまった領域になりやすい。
    """

    indexes = list(unassigned)

    if len(indexes) == 1:
        return indexes[0]

    unassigned_points = points[indexes]
    center = np.mean(unassigned_points, axis=0)

    distances = np.linalg.norm(
        unassigned_points - center,
        axis=1
    )

    maximum_distance = np.max(distances)

    candidates = [
        indexes[i]
        for i, distance in enumerate(distances)
        if distance >= maximum_distance * 0.85
    ]

    return rng.choice(candidates)


def find_nearest_node_to_group(
    group: list[int],
    unassigned: set[int],
    points: np.ndarray
) -> int:
    group_points = points[group]

    best_index = None
    best_distance = float("inf")

    for candidate_index in unassigned:
        candidate_point = points[candidate_index]

        distances = np.linalg.norm(
            group_points - candidate_point,
            axis=1
        )

        minimum_distance = float(np.min(distances))

        if minimum_distance < best_distance:
            best_distance = minimum_distance
            best_index = candidate_index

    if best_index is None:
        raise RuntimeError("次に追加するノードを決定できませんでした。")

    return best_index


# =========================================================
# 各ノードに国番号を割り当てる
# =========================================================

def create_node_country_ids(
    number_of_nodes: int,
    country_groups: list[list[int]]
) -> np.ndarray:
    node_country_ids = np.full(number_of_nodes, -1, dtype=np.int32)

    for country_id, group in enumerate(country_groups):
        for node_index in group:
            node_country_ids[node_index] = country_id

    if np.any(node_country_ids < 0):
        raise RuntimeError("国が割り当てられていないノードがあります。")

    return node_country_ids


# =========================================================
# 陸地を最寄りノードの国に割り当てる
# =========================================================

def create_country_map(
    image_shape: tuple[int, int, int],
    land_mask: np.ndarray,
    node_centers: list[tuple[int, int]],
    node_country_ids: np.ndarray
) -> np.ndarray:
    height, width = image_shape[:2]

    # distance_transform_edtでは0の場所が距離計算の基準になる
    seed_image = np.ones((height, width), dtype=np.uint8)

    valid_nodes = []

    for node_index, (x, y) in enumerate(node_centers):
        if 0 <= x < width and 0 <= y < height:
            seed_image[y, x] = 0
            valid_nodes.append((x, y, node_index))

    _, nearest_indices = distance_transform_edt(
        seed_image,
        return_indices=True
    )

    nearest_y = nearest_indices[0]
    nearest_x = nearest_indices[1]

    seed_to_node_index = np.full(
        (height, width),
        -1,
        dtype=np.int32
    )

    for x, y, node_index in valid_nodes:
        seed_to_node_index[y, x] = node_index

    nearest_node_index = seed_to_node_index[
        nearest_y,
        nearest_x
    ]

    country_map = np.full(
        (height, width),
        -1,
        dtype=np.int32
    )

    valid_pixel_mask = land_mask & (nearest_node_index >= 0)

    country_map[valid_pixel_mask] = node_country_ids[
        nearest_node_index[valid_pixel_mask]
    ]

    return country_map


# =========================================================
# 国別に着色
# =========================================================

def colorize_countries(
    original_image: np.ndarray,
    land_mask: np.ndarray,
    country_map: np.ndarray,
    number_of_countries: int
) -> np.ndarray:
    result = original_image.copy()
    color_layer = original_image.copy()

    rng = random.Random(RANDOM_SEED)

    colors = COUNTRY_COLORS.copy()

    # 国数が色数より多い場合は色を自動生成
    while len(colors) < number_of_countries:
        hue = rng.randint(0, 179)
        saturation = rng.randint(45, 105)
        value = rng.randint(205, 245)

        hsv_color = np.uint8([[[hue, saturation, value]]])
        bgr_color = cv2.cvtColor(
            hsv_color,
            cv2.COLOR_HSV2BGR
        )[0, 0]

        colors.append(tuple(int(value) for value in bgr_color))

    rng.shuffle(colors)

    for country_id in range(number_of_countries):
        country_mask = country_map == country_id
        color_layer[country_mask] = colors[country_id]

    blended = cv2.addWeighted(
        original_image,
        1.0 - COLOR_ALPHA,
        color_layer,
        COLOR_ALPHA,
        0
    )

    result[land_mask] = blended[land_mask]

    # 道路、円、文字、海岸線など暗い部分を元画像から復元
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    dark_mask = (gray < DARK_PIXEL_THRESHOLD) & land_mask

    result[dark_mask] = original_image[dark_mask]

    return result


# =========================================================
# 国境線の作成
# =========================================================

def draw_country_borders(
    image: np.ndarray,
    country_map: np.ndarray,
    land_mask: np.ndarray
) -> np.ndarray:
    result = image.copy()

    border_mask = np.zeros(
        country_map.shape,
        dtype=np.uint8
    )

    # 左右で国番号が異なる箇所
    horizontal_difference = (
        country_map[:, 1:] != country_map[:, :-1]
    )

    horizontal_valid = (
        land_mask[:, 1:]
        & land_mask[:, :-1]
        & (country_map[:, 1:] >= 0)
        & (country_map[:, :-1] >= 0)
    )

    horizontal_border = (
        horizontal_difference
        & horizontal_valid
    )

    border_mask[:, 1:][horizontal_border] = 255
    border_mask[:, :-1][horizontal_border] = 255

    # 上下で国番号が異なる箇所
    vertical_difference = (
        country_map[1:, :] != country_map[:-1, :]
    )

    vertical_valid = (
        land_mask[1:, :]
        & land_mask[:-1, :]
        & (country_map[1:, :] >= 0)
        & (country_map[:-1, :] >= 0)
    )

    vertical_border = (
        vertical_difference
        & vertical_valid
    )

    border_mask[1:, :][vertical_border] = 255
    border_mask[:-1, :][vertical_border] = 255

    # 国境を滑らかにする
    border_mask = cv2.GaussianBlur(
        border_mask,
        (5, 5),
        0
    )

    _, border_mask = cv2.threshold(
        border_mask,
        50,
        255,
        cv2.THRESH_BINARY
    )

    if BORDER_WIDTH > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (BORDER_WIDTH, BORDER_WIDTH)
        )

        border_mask = cv2.dilate(
            border_mask,
            kernel,
            iterations=1
        )

    # 海上には描画しない
    border_mask[~land_mask] = 0

    # 白い縁取り
    result[border_mask > 0] = BORDER_COLOR

    # 国境中央に細い暗色線を描く
    thin_border = cv2.erode(
        border_mask,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1
    )

    result[thin_border > 0] = (115, 115, 115)

    return result


# =========================================================
# 元の道路・文字・ノードを最前面に復元
# =========================================================

def restore_map_details(
    original_image: np.ndarray,
    colored_image: np.ndarray,
    land_mask: np.ndarray
) -> np.ndarray:
    result = colored_image.copy()

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    dark_mask = (
        (gray < DARK_PIXEL_THRESHOLD)
        & land_mask
    )

    # 暗い線の周囲にあるアンチエイリアスも含める
    dark_mask_uint8 = dark_mask.astype(np.uint8) * 255

    dark_mask_uint8 = cv2.dilate(
        dark_mask_uint8,
        np.ones((2, 2), dtype=np.uint8),
        iterations=1
    )

    restore_mask = dark_mask_uint8 > 0

    result[restore_mask] = original_image[restore_mask]

    return result


# =========================================================
# ノード検出確認画像
# =========================================================

def save_node_debug_image(
    image: np.ndarray,
    node_centers: list[tuple[int, int]],
    country_groups: list[list[int]]
) -> None:
    debug_image = image.copy()

    group_lookup = {}

    for country_id, group in enumerate(country_groups):
        for node_index in group:
            group_lookup[node_index] = country_id

    for node_index, (x, y) in enumerate(node_centers):
        country_id = group_lookup.get(node_index, -1)

        cv2.circle(
            debug_image,
            (x, y),
            24,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            debug_image,
            f"N{node_index + 1}/C{country_id + 1}",
            (x + 15, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    cv2.imwrite(str(NODE_DEBUG_FILE), debug_image)


# =========================================================
# メイン処理
# =========================================================

def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("画像を読み込んでいます...")
    original_image = load_image(INPUT_FILE)

    print("陸地を検出しています...")
    land_mask = create_land_mask(original_image)

    print("ノード円を検出しています...")
    node_centers = detect_nodes(
        original_image,
        land_mask
    )

    print(f"検出したノード数：{len(node_centers)}")

    print("ノードを国ごとに分けています...")
    country_groups = create_country_groups(node_centers)

    print(f"作成する国の数：{len(country_groups)}")

    for country_id, group in enumerate(country_groups, start=1):
        print(
            f"国{country_id:02d}："
            f"{len(group)}都市"
        )

    node_country_ids = create_node_country_ids(
        number_of_nodes=len(node_centers),
        country_groups=country_groups
    )

    print("陸地を国別に分割しています...")
    country_map = create_country_map(
        image_shape=original_image.shape,
        land_mask=land_mask,
        node_centers=node_centers,
        node_country_ids=node_country_ids
    )

    print("国別に色を付けています...")
    result = colorize_countries(
        original_image=original_image,
        land_mask=land_mask,
        country_map=country_map,
        number_of_countries=len(country_groups)
    )

    print("国境線を描いています...")
    result = draw_country_borders(
        image=result,
        country_map=country_map,
        land_mask=land_mask
    )

    print("道路、文字、ノードを復元しています...")
    result = restore_map_details(
        original_image=original_image,
        colored_image=result,
        land_mask=land_mask
    )

    if OUTPUT_NODE_DEBUG_IMAGE:
        save_node_debug_image(
            original_image,
            node_centers,
            country_groups
        )

    success = cv2.imwrite(
        str(OUTPUT_FILE),
        result
    )

    if not success:
        raise RuntimeError(
            f"画像を保存できませんでした：{OUTPUT_FILE}"
        )

    print()
    print("処理が完了しました。")
    print(f"出力先：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()