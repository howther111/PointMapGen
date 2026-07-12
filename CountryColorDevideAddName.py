from pathlib import Path
import random

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image, ImageDraw, ImageFont


# =========================================================
# 基本設定
# =========================================================

INPUT_FILE = Path("input_coastline.png")
OUTPUT_FILE = Path("output_country.png")

# 同じ結果を再現するための乱数シード
RANDOM_SEED = 42

# 1か国が持つノード数
MIN_NODES_PER_COUNTRY = 1
MAX_NODES_PER_COUNTRY = 10


# =========================================================
# 海・陸地判定設定
# =========================================================

# 海色：OpenCVのBGR形式
SEA_COLOR = (235, 205, 155)

# 海色判定の許容誤差
# JPG画像などで色がわずかに変化している場合に対応
SEA_COLOR_TOLERANCE = 25

# 小さな陸地候補を除去する面積比率
MINIMUM_LAND_AREA_RATIO = 0.001


# =========================================================
# 国の着色設定
# =========================================================

# 国色の透明度
# 0.0：元画像そのまま
# 1.0：国色で完全に塗りつぶす
COLOR_ALPHA = 0.63

# 国境線の太さ
BORDER_WIDTH = 3

# 国境線の縁取り色：BGR
BORDER_OUTLINE_COLOR = (245, 245, 245)

# 国境線中央の色：BGR
BORDER_CENTER_COLOR = (110, 110, 110)

# 元画像の道路・文字・ノードなどを復元するときの明度基準
# 小さいほど黒い部分だけを復元する
DARK_PIXEL_THRESHOLD = 205


# =========================================================
# ノード検出設定
# =========================================================

HOUGH_DP = 1.2

# 画像短辺に対するノード間最小距離
HOUGH_MIN_DISTANCE_RATIO = 0.025

# 画像短辺に対するノード半径
NODE_MIN_RADIUS_RATIO = 0.010
NODE_MAX_RADIUS_RATIO = 0.022

# 円検出の厳しさ
# 小さくすると多く検出され、大きくすると厳しくなる
HOUGH_CIRCLE_THRESHOLD = 28

# ノード検出確認画像を出力するか
OUTPUT_NODE_DEBUG_IMAGE = False
NODE_DEBUG_FILE = Path("ノード検出結果.png")


# =========================================================
# 国名設定
# =========================================================

# Windows用日本語太字フォント
COUNTRY_NAME_FONT = Path(r"C:\Windows\Fonts\meiryob.ttc")

# 上記フォントが見つからない場合の候補
COUNTRY_NAME_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\meiryob.ttc"),
    Path(r"C:\Windows\Fonts\YuGothB.ttc"),
    Path(r"C:\Windows\Fonts\YuGothM.ttc"),
    Path(r"C:\Windows\Fonts\msgothic.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc"),
    Path("/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf"),
]

# 国名の文字色：PillowのRGB形式
COUNTRY_NAME_COLOR = (55, 45, 40)

# 国名の縁取り色：RGB形式
COUNTRY_NAME_STROKE_COLOR = (248, 248, 242)

# 国名の縁取り幅
COUNTRY_NAME_STROKE_WIDTH = 3

# 国名の最小・最大フォントサイズ
COUNTRY_NAME_MIN_FONT_SIZE = 18
COUNTRY_NAME_MAX_FONT_SIZE = 72

# 国領域に対する基本フォントサイズ倍率
COUNTRY_NAME_FONT_SCALE = 0.12

# 国名が収まる横幅の割合
COUNTRY_NAME_MAX_WIDTH_RATIO = 0.80

# 国名が収まる高さの割合
COUNTRY_NAME_MAX_HEIGHT_RATIO = 0.30

# 小さすぎる国には国名を描画しない
MINIMUM_COUNTRY_LABEL_AREA = 1500


# =========================================================
# 国名一覧
# 上から順番に国へ割り当てる
# =========================================================

COUNTRY_NAMES = [
    "アルディア王国",
    "ベルネシア",
    "ロザリア共和国",
    "セントール帝国",
    "ミレニア公国",
    "ガルディア",
    "ノルフェン王国",
    "エルメリア",
    "ザルバード",
    "フェルニア",
    "オルディス",
    "レムナント",
    "ヴァルハイム",
    "ユースティア",
    "グランベル",
    "リュミエール",
    "カルディナ",
    "ネフティス",
    "アストリア",
    "ディルムント",
]


# =========================================================
# 国の色
# OpenCVのBGR形式
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
# 国名自動生成用
# =========================================================

COUNTRY_NAME_PREFIXES = [
    "アル", "ベル", "セル", "ガル", "ヴァル",
    "ロス", "エル", "ノル", "フェル", "ミル",
    "レグ", "ゼル", "カル", "ドル", "リュ",
    "アス", "ネフ", "グラン", "ラグ", "イル",
]

COUNTRY_NAME_MIDDLES = [
    "ディ", "ネ", "リ", "ラ", "ティ",
    "ヴァ", "メ", "シ", "ト", "ル",
    "ニ", "フェ", "ザ", "ベル", "リア",
]

COUNTRY_NAME_SUFFIXES = [
    "ア", "リア", "ニア", "ネス", "ディア",
    "ランド", "王国", "帝国", "公国", "共和国",
    "連邦", "領", "自治国", "皇国", "同盟",
]


# =========================================================
# 画像入出力
# =========================================================

def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(
            "画像を読み込めませんでした。\n"
            f"対象ファイル：{path}\n"
            "Pythonファイルと画像の配置場所を確認してください。"
        )

    return image


def save_image(path: Path, image: np.ndarray) -> None:
    success = cv2.imwrite(str(path), image)

    if not success:
        raise RuntimeError(
            f"画像を保存できませんでした：{path}"
        )


# =========================================================
# 陸地マスク作成
# =========================================================

def create_land_mask(image: np.ndarray) -> np.ndarray:
    """
    BGR=(235,205,155)を海色として陸地を検出する。
    """

    image_float = image.astype(np.float32)
    sea_color = np.array(SEA_COLOR, dtype=np.float32)

    color_distance = np.linalg.norm(
        image_float - sea_color[None, None, :],
        axis=2
    )

    sea_mask = color_distance <= SEA_COLOR_TOLERANCE

    # 小さな色ノイズを整える
    kernel = np.ones((5, 5), dtype=np.uint8)

    sea_mask_uint8 = sea_mask.astype(np.uint8) * 255

    sea_mask_uint8 = cv2.morphologyEx(
        sea_mask_uint8,
        cv2.MORPH_OPEN,
        kernel
    )

    sea_mask_uint8 = cv2.morphologyEx(
        sea_mask_uint8,
        cv2.MORPH_CLOSE,
        kernel
    )

    sea_mask = sea_mask_uint8 > 0
    land_mask = ~sea_mask

    # 小さなノイズ領域を除去
    number_of_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            land_mask.astype(np.uint8),
            connectivity=8
        )
    )

    cleaned_land_mask = np.zeros_like(land_mask)

    image_area = image.shape[0] * image.shape[1]
    minimum_land_area = int(
        image_area * MINIMUM_LAND_AREA_RATIO
    )

    for label_id in range(1, number_of_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area >= minimum_land_area:
            cleaned_land_mask[labels == label_id] = True

    return cleaned_land_mask


# =========================================================
# ノード円検出
# =========================================================

def detect_nodes(
    image: np.ndarray,
    land_mask: np.ndarray
) -> list[tuple[int, int]]:
    height, width = image.shape[:2]
    base_size = min(height, width)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        1.2
    )

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
        param2=HOUGH_CIRCLE_THRESHOLD,
        minRadius=minimum_radius,
        maxRadius=maximum_radius
    )

    if circles is None:
        raise RuntimeError(
            "ノード円を検出できませんでした。\n"
            "HOUGH_CIRCLE_THRESHOLDを小さくするか、"
            "ノード半径設定を調整してください。"
        )

    circles = np.round(circles[0]).astype(int)

    node_centers = []

    for x, y, radius in circles:
        if not (0 <= x < width and 0 <= y < height):
            continue

        # 円の中心が陸地上にある場合だけ採用
        if not land_mask[y, x]:
            continue

        node_centers.append((x, y))

    node_centers = remove_duplicate_nodes(
        node_centers=node_centers,
        minimum_distance=max(
            10,
            minimum_distance // 2
        )
    )

    # 上から下、左から右の順に並べる
    node_centers.sort(
        key=lambda point: (point[1], point[0])
    )

    if not node_centers:
        raise RuntimeError(
            "陸地上の有効なノードが見つかりませんでした。"
        )

    return node_centers


def remove_duplicate_nodes(
    node_centers: list[tuple[int, int]],
    minimum_distance: float
) -> list[tuple[int, int]]:
    result = []

    for candidate in node_centers:
        candidate_array = np.array(
            candidate,
            dtype=np.float32
        )

        is_duplicate = False

        for registered in result:
            registered_array = np.array(
                registered,
                dtype=np.float32
            )

            distance = np.linalg.norm(
                candidate_array - registered_array
            )

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
    近いノードを集め、1か国1～10都市で国を作る。
    """

    rng = random.Random(RANDOM_SEED)

    number_of_nodes = len(node_centers)
    unassigned = set(range(number_of_nodes))
    country_groups = []

    points = np.array(
        node_centers,
        dtype=np.float32
    )

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

        while (
            len(group) < target_size
            and unassigned
        ):
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
    maximum_size = min(
        MAX_NODES_PER_COUNTRY,
        remaining_count
    )

    minimum_size = min(
        MIN_NODES_PER_COUNTRY,
        maximum_size
    )

    if remaining_count <= MAX_NODES_PER_COUNTRY:
        return remaining_count

    size = rng.randint(
        minimum_size,
        maximum_size
    )

    nodes_after_creation = remaining_count - size

    if (
        nodes_after_creation > 0
        and nodes_after_creation < MIN_NODES_PER_COUNTRY
    ):
        size = (
            remaining_count
            - MIN_NODES_PER_COUNTRY
        )

    return max(
        minimum_size,
        min(size, maximum_size)
    )


def choose_country_seed(
    unassigned: set[int],
    points: np.ndarray,
    rng: random.Random
) -> int:
    """
    未割り当てノードの外側にあるノードを
    国作成の開始地点として優先する。
    """

    indexes = list(unassigned)

    if len(indexes) == 1:
        return indexes[0]

    unassigned_points = points[indexes]
    center = np.mean(
        unassigned_points,
        axis=0
    )

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

        minimum_distance = float(
            np.min(distances)
        )

        if minimum_distance < best_distance:
            best_distance = minimum_distance
            best_index = candidate_index

    if best_index is None:
        raise RuntimeError(
            "次に追加するノードを決定できませんでした。"
        )

    return best_index


# =========================================================
# ノードと国番号の関連付け
# =========================================================

def create_node_country_ids(
    number_of_nodes: int,
    country_groups: list[list[int]]
) -> np.ndarray:
    node_country_ids = np.full(
        number_of_nodes,
        -1,
        dtype=np.int32
    )

    for country_id, group in enumerate(country_groups):
        for node_index in group:
            node_country_ids[node_index] = country_id

    if np.any(node_country_ids < 0):
        raise RuntimeError(
            "国が割り当てられていないノードがあります。"
        )

    return node_country_ids


# =========================================================
# 陸地を最寄りノードの国へ割り当て
# =========================================================

def create_country_map(
    image_shape: tuple[int, int, int],
    land_mask: np.ndarray,
    node_centers: list[tuple[int, int]],
    node_country_ids: np.ndarray
) -> np.ndarray:
    height, width = image_shape[:2]

    # 0の場所を距離計算の基準点とする
    seed_image = np.ones(
        (height, width),
        dtype=np.uint8
    )

    valid_nodes = []

    for node_index, (x, y) in enumerate(node_centers):
        if 0 <= x < width and 0 <= y < height:
            seed_image[y, x] = 0
            valid_nodes.append(
                (x, y, node_index)
            )

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

    valid_pixel_mask = (
        land_mask
        & (nearest_node_index >= 0)
    )

    country_map[valid_pixel_mask] = (
        node_country_ids[
            nearest_node_index[valid_pixel_mask]
        ]
    )

    return country_map


# =========================================================
# 国色作成
# =========================================================

def create_country_colors(
    number_of_countries: int
) -> list[tuple[int, int, int]]:
    rng = random.Random(RANDOM_SEED)

    colors = COUNTRY_COLORS.copy()

    # 色一覧より国数が多い場合は自動生成
    while len(colors) < number_of_countries:
        hue = rng.randint(0, 179)
        saturation = rng.randint(45, 100)
        value = rng.randint(205, 245)

        hsv_color = np.uint8([
            [[hue, saturation, value]]
        ])

        bgr_color = cv2.cvtColor(
            hsv_color,
            cv2.COLOR_HSV2BGR
        )[0, 0]

        colors.append(
            tuple(int(channel) for channel in bgr_color)
        )

    rng.shuffle(colors)

    return colors[:number_of_countries]


# =========================================================
# 国別着色
# =========================================================

def colorize_countries(
    original_image: np.ndarray,
    land_mask: np.ndarray,
    country_map: np.ndarray,
    number_of_countries: int
) -> np.ndarray:
    result = original_image.copy()
    color_layer = original_image.copy()

    colors = create_country_colors(
        number_of_countries
    )

    for country_id in range(number_of_countries):
        country_mask = (
            country_map == country_id
        )

        color_layer[country_mask] = colors[country_id]

    blended = cv2.addWeighted(
        original_image,
        1.0 - COLOR_ALPHA,
        color_layer,
        COLOR_ALPHA,
        0
    )

    result[land_mask] = blended[land_mask]

    return result


# =========================================================
# 国境線作成
# =========================================================

def create_country_border_mask(
    country_map: np.ndarray,
    land_mask: np.ndarray
) -> np.ndarray:
    border_mask = np.zeros(
        country_map.shape,
        dtype=np.uint8
    )

    # 左右方向の国番号の違い
    horizontal_difference = (
        country_map[:, 1:]
        != country_map[:, :-1]
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

    # 上下方向の国番号の違い
    vertical_difference = (
        country_map[1:, :]
        != country_map[:-1, :]
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

    # 境界を滑らかにする
    border_mask = cv2.GaussianBlur(
        border_mask,
        (5, 5),
        0
    )

    _, border_mask = cv2.threshold(
        border_mask,
        40,
        255,
        cv2.THRESH_BINARY
    )

    # 海上には描かない
    border_mask[~land_mask] = 0

    return border_mask


def draw_country_borders(
    image: np.ndarray,
    country_map: np.ndarray,
    land_mask: np.ndarray
) -> np.ndarray:
    result = image.copy()

    border_mask = create_country_border_mask(
        country_map=country_map,
        land_mask=land_mask
    )

    outline_size = max(
        3,
        BORDER_WIDTH * 2 + 1
    )

    if outline_size % 2 == 0:
        outline_size += 1

    outline_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (outline_size, outline_size)
    )

    outline_mask = cv2.dilate(
        border_mask,
        outline_kernel,
        iterations=1
    )

    outline_mask[~land_mask] = 0

    result[outline_mask > 0] = (
        BORDER_OUTLINE_COLOR
    )

    center_size = max(
        1,
        BORDER_WIDTH
    )

    center_kernel_size = (
        center_size,
        center_size
    )

    center_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        center_kernel_size
    )

    center_mask = cv2.dilate(
        border_mask,
        center_kernel,
        iterations=1
    )

    center_mask[~land_mask] = 0

    result[center_mask > 0] = (
        BORDER_CENTER_COLOR
    )

    return result


# =========================================================
# 道路・ノード・都市名などを復元
# =========================================================

def restore_map_details(
    original_image: np.ndarray,
    colored_image: np.ndarray,
    land_mask: np.ndarray
) -> np.ndarray:
    result = colored_image.copy()

    gray = cv2.cvtColor(
        original_image,
        cv2.COLOR_BGR2GRAY
    )

    dark_mask = (
        (gray < DARK_PIXEL_THRESHOLD)
        & land_mask
    )

    dark_mask_uint8 = (
        dark_mask.astype(np.uint8) * 255
    )

    # 線のアンチエイリアス部分も復元する
    dark_mask_uint8 = cv2.dilate(
        dark_mask_uint8,
        np.ones((2, 2), dtype=np.uint8),
        iterations=1
    )

    restore_mask = dark_mask_uint8 > 0

    result[restore_mask] = (
        original_image[restore_mask]
    )

    return result


# =========================================================
# 国名生成
# =========================================================

def generate_country_name(
    rng: random.Random,
    used_names: set[str]
) -> str:
    for _ in range(1000):
        prefix = rng.choice(
            COUNTRY_NAME_PREFIXES
        )

        if rng.random() < 0.55:
            middle = rng.choice(
                COUNTRY_NAME_MIDDLES
            )
        else:
            middle = ""

        suffix = rng.choice(
            COUNTRY_NAME_SUFFIXES
        )

        name = prefix + middle + suffix

        if name not in used_names:
            used_names.add(name)
            return name

    fallback_name = (
        f"第{len(used_names) + 1}国"
    )

    used_names.add(fallback_name)

    return fallback_name


def create_country_names(
    number_of_countries: int
) -> list[str]:
    rng = random.Random(
        RANDOM_SEED + 1000
    )

    names = []
    used_names = set()

    for country_id in range(number_of_countries):
        if country_id < len(COUNTRY_NAMES):
            name = COUNTRY_NAMES[country_id]

            # 指定名が重複していた場合は番号を付ける
            if name in used_names:
                name = (
                    f"{name}{country_id + 1}"
                )

            used_names.add(name)

        else:
            name = generate_country_name(
                rng=rng,
                used_names=used_names
            )

        names.append(name)

    return names


# =========================================================
# 日本語フォント検索
# =========================================================

def find_country_name_font() -> Path:
    if COUNTRY_NAME_FONT.exists():
        return COUNTRY_NAME_FONT

    for font_path in COUNTRY_NAME_FONT_CANDIDATES:
        if font_path.exists():
            return font_path

    raise FileNotFoundError(
        "日本語フォントが見つかりませんでした。\n"
        "COUNTRY_NAME_FONTへ日本語対応フォントの"
        "ファイルパスを指定してください。\n\n"
        "Windowsの例：\n"
        r"C:\Windows\Fonts\meiryob.ttc"
    )


# =========================================================
# 国名配置位置
# =========================================================

def find_country_label_position(
    country_mask: np.ndarray
) -> tuple[int, int, float]:
    """
    国の最大連結領域内で、境界から最も遠い地点を探す。
    湖や海の中へ文字が入ることを防ぎやすい。
    """

    mask_uint8 = (
        country_mask.astype(np.uint8)
    )

    number_of_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask_uint8,
            connectivity=8
        )
    )

    if number_of_labels <= 1:
        height, width = country_mask.shape

        return (
            width // 2,
            height // 2,
            10.0
        )

    largest_label = 1
    largest_area = stats[
        1,
        cv2.CC_STAT_AREA
    ]

    for label_id in range(
        2,
        number_of_labels
    ):
        area = stats[
            label_id,
            cv2.CC_STAT_AREA
        ]

        if area > largest_area:
            largest_area = area
            largest_label = label_id

    largest_region = (
        labels == largest_label
    ).astype(np.uint8)

    distance_map = cv2.distanceTransform(
        largest_region,
        cv2.DIST_L2,
        5
    )

    _, maximum_distance, _, maximum_location = (
        cv2.minMaxLoc(distance_map)
    )

    x, y = maximum_location

    return (
        int(x),
        int(y),
        float(maximum_distance)
    )


# =========================================================
# 文字範囲取得
# =========================================================

def get_text_bbox(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    stroke_width: int
) -> tuple[int, int, int, int]:
    return draw.textbbox(
        (0, 0),
        text,
        font=font,
        stroke_width=stroke_width
    )


# =========================================================
# 国名フォントサイズ決定
# =========================================================

def choose_country_name_font(
    draw: ImageDraw.ImageDraw,
    font_path: Path,
    country_name: str,
    country_mask: np.ndarray,
    maximum_distance: float
) -> ImageFont.FreeTypeFont:
    ys, xs = np.where(country_mask)

    if len(xs) == 0:
        return ImageFont.truetype(
            str(font_path),
            COUNTRY_NAME_MIN_FONT_SIZE
        )

    country_width = int(
        xs.max() - xs.min() + 1
    )

    country_height = int(
        ys.max() - ys.min() + 1
    )

    base_size = min(
        country_width,
        country_height
    )

    estimated_size = int(
        base_size
        * COUNTRY_NAME_FONT_SCALE
    )

    estimated_size = max(
        COUNTRY_NAME_MIN_FONT_SIZE,
        min(
            COUNTRY_NAME_MAX_FONT_SIZE,
            estimated_size
        )
    )

    # 境界までの距離に応じて上限を調整
    distance_based_size = max(
        COUNTRY_NAME_MIN_FONT_SIZE,
        int(maximum_distance * 0.90)
    )

    estimated_size = min(
        estimated_size,
        distance_based_size
    )

    maximum_text_width = (
        country_width
        * COUNTRY_NAME_MAX_WIDTH_RATIO
    )

    maximum_text_height = (
        country_height
        * COUNTRY_NAME_MAX_HEIGHT_RATIO
    )

    for font_size in range(
        estimated_size,
        COUNTRY_NAME_MIN_FONT_SIZE - 1,
        -2
    ):
        font = ImageFont.truetype(
            str(font_path),
            font_size
        )

        left, top, right, bottom = (
            get_text_bbox(
                draw=draw,
                text=country_name,
                font=font,
                stroke_width=(
                    COUNTRY_NAME_STROKE_WIDTH
                )
            )
        )

        text_width = right - left
        text_height = bottom - top

        if (
            text_width <= maximum_text_width
            and text_height <= maximum_text_height
        ):
            return font

    return ImageFont.truetype(
        str(font_path),
        COUNTRY_NAME_MIN_FONT_SIZE
    )


# =========================================================
# 国名描画
# =========================================================

def draw_country_names(
    image: np.ndarray,
    country_map: np.ndarray,
    country_names: list[str]
) -> np.ndarray:
    font_path = find_country_name_font()

    # OpenCVのBGRからPillowのRGBへ変換
    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        rgb_image
    )

    draw = ImageDraw.Draw(
        pil_image
    )

    for country_id, country_name in enumerate(
        country_names
    ):
        country_mask = (
            country_map == country_id
        )

        country_area = int(
            np.count_nonzero(country_mask)
        )

        if country_area < MINIMUM_COUNTRY_LABEL_AREA:
            print(
                f"国{country_id + 1:02d}は小さいため"
                "国名表示を省略しました。"
            )
            continue

        center_x, center_y, maximum_distance = (
            find_country_label_position(
                country_mask
            )
        )

        font = choose_country_name_font(
            draw=draw,
            font_path=font_path,
            country_name=country_name,
            country_mask=country_mask,
            maximum_distance=maximum_distance
        )

        left, top, right, bottom = (
            get_text_bbox(
                draw=draw,
                text=country_name,
                font=font,
                stroke_width=(
                    COUNTRY_NAME_STROKE_WIDTH
                )
            )
        )

        text_width = right - left
        text_height = bottom - top

        text_x = (
            center_x
            - text_width // 2
            - left
        )

        text_y = (
            center_y
            - text_height // 2
            - top
        )

        draw.text(
            (text_x, text_y),
            country_name,
            font=font,
            fill=COUNTRY_NAME_COLOR,
            stroke_width=(
                COUNTRY_NAME_STROKE_WIDTH
            ),
            stroke_fill=(
                COUNTRY_NAME_STROKE_COLOR
            )
        )

    result = cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR
    )

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

    for country_id, group in enumerate(
        country_groups
    ):
        for node_index in group:
            group_lookup[node_index] = country_id

    for node_index, (x, y) in enumerate(
        node_centers
    ):
        country_id = group_lookup.get(
            node_index,
            -1
        )

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
            (
                f"N{node_index + 1}"
                f"/C{country_id + 1}"
            ),
            (x + 15, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    save_image(
        NODE_DEBUG_FILE,
        debug_image
    )


# =========================================================
# メイン処理
# =========================================================

def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("画像を読み込んでいます...")
    original_image = load_image(
        INPUT_FILE
    )

    print("陸地を検出しています...")
    land_mask = create_land_mask(
        original_image
    )

    print("ノード円を検出しています...")
    node_centers = detect_nodes(
        original_image,
        land_mask
    )

    print(
        f"検出したノード数："
        f"{len(node_centers)}"
    )

    print("ノードを国ごとに分けています...")
    country_groups = create_country_groups(
        node_centers
    )

    print(
        f"作成する国の数："
        f"{len(country_groups)}"
    )

    for country_id, group in enumerate(
        country_groups,
        start=1
    ):
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

    print(
        "道路、都市名、ノードなどを"
        "復元しています..."
    )
    result = restore_map_details(
        original_image=original_image,
        colored_image=result,
        land_mask=land_mask
    )

    print("国名を作成しています...")
    country_names = create_country_names(
        number_of_countries=len(
            country_groups
        )
    )

    for country_id, country_name in enumerate(
        country_names,
        start=1
    ):
        number_of_cities = len(
            country_groups[country_id - 1]
        )

        print(
            f"国{country_id:02d}："
            f"{country_name} "
            f"（{number_of_cities}都市）"
        )

    print("国名を描画しています...")
    result = draw_country_names(
        image=result,
        country_map=country_map,
        country_names=country_names
    )

    if OUTPUT_NODE_DEBUG_IMAGE:
        print(
            "ノード検出確認画像を"
            "保存しています..."
        )

        save_node_debug_image(
            image=original_image,
            node_centers=node_centers,
            country_groups=country_groups
        )

    print("完成画像を保存しています...")
    save_image(
        OUTPUT_FILE,
        result
    )

    print()
    print("処理が完了しました。")
    print(f"出力先：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()