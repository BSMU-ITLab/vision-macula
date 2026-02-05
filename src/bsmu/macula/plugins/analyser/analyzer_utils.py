import os 
import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline

# === пути ===
images_dir = "images"
masks_dir = "masks"
fovea_dir = "masks-fovea"
out_dir = "choroid_smooth_vis"
os.makedirs(out_dir, exist_ok=True)

CHOROID_ID = 14
CHOROID_COLOR = (200, 255, 255)      # бирюзовый
FOVEOLA_COLOR = (0, 255, 255)        # жёлтый  (1)
FOVEA_COLOR   = (0, 165, 255)        # оранжевый (2)
LINE_COLOR = (0, 255, 0)

# Константы для классов патологий
RETINA_LAYERS = {4, 5, 6, 9}  # Субретинальный гиперрефлективный материал (СРГМ), интраретинальный гиперрефлективный материал (ИРГМ), субретинальная жидкость (СРЖ), РПЭ
RETINA_LAYER_COLORS = {
    4: (255, 0, 0),   # Субретинальный гиперрефлективный материал (СРГМ) - Красный
    5: (0, 0, 255),   # Интраретинальный гиперрефлективный материал (ИРГМ) - Синий
    6: (255, 255, 0), # Субретинальная жидкость (СРЖ) - Желтый
    9: (0, 255, 0)    # Ретинальный пигментный эпителий (РПЭ) - Зеленый
}

# ============================================================
#     СГЛАЖИВАНИЕ МАСКИ ХОРИОИДЕИ
# ============================================================
def smooth_mask(mask):
    mask_bin = ((mask == CHOROID_ID).astype(np.uint8)) * 255
    kernel = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = cv2.GaussianBlur(m, (7, 7), 0)
    m = (m > 127).astype(np.uint8)
    return m


# ============================================================
#     ВЕРХНЯЯ ГРАНИЦА ХОРИОИДЕИ
# ============================================================
def extract_upper_boundary(mask):
    mask_smooth = smooth_mask(mask)
    contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None, None

    cnt = max(contours, key=cv2.contourArea)
    h, w = mask.shape
    xs = np.arange(w)
    upper = np.full(w, np.nan)

    for pt in cnt:
        x, y = pt[0]
        if np.isnan(upper[x]) or y < upper[x]:
            upper[x] = y

    ok = ~np.isnan(upper)
    upper_interp = np.interp(xs, xs[ok], upper[ok])
    return xs, upper_interp


# ============================================================
#     СПЛАЙН
# ============================================================
def smooth_boundary(xs, ys, smooth_factor=50):
    spline = UnivariateSpline(xs, ys, s=smooth_factor)
    return spline(xs), spline


# ============================================================
#     ЦЕНТР ФОВЕОЛЫ (mask==1)
# ============================================================
def get_foveola_center(fovea_mask):
    ys, xs = np.where(fovea_mask == 1)
    if len(xs) == 0:
        return None
    return int(np.mean(xs)), int(np.mean(ys))


def central_width_of_retina(smooth_upper, spline, foveola_center, mask,
                                  perp_len=200):

    cx, cy = foveola_center  
    h, w = mask.shape

    if cx < 0 or cx >= len(smooth_upper):
        return None

    # верхняя граница (сглаженная)
    y0 = float(smooth_upper[cx])

    slope = float(spline.derivative()(cx))

    # нормаль вниз
    nx = -slope
    ny = 1.0
    L = np.sqrt(nx*nx + ny*ny)
    nx /= L
    ny /= L
    cts_dist = 0
    inside = False          # флаг "мы внутри хориоидеи"
    best_point = None

    for t in range(0, perp_len):
        xx = int(cx + nx * t)
        yy = int(cy + ny * t)

        if xx < 0 or xx >= w or yy < 0 or yy >= h:
            break

        if mask[yy, xx] == CHOROID_ID or mask[yy, xx] in RETINA_LAYERS:
            best_point = (xx, yy)
            cts_dist = int(np.sqrt((best_point[0] - cx) ** 2 + (best_point[1] - y0) ** 2))
            break

    if best_point is None:
        return None

    # расстояние считаем по t
    dist = int(np.sqrt((best_point[0] - cx)**2 + (best_point[1] - y0)**2))

    return cx, cy, dist, best_point


def central_width_near(img, smooth_upper, spline, fovea_mask, mask, ind):
    """
    Находит крайние левую и правую точки на границе зоны фовеа/фовеолы с фоном.
    
    Args:
        ind: 1 для фовеолы, 2 для фовеа
        
    Ищет точки, в окрестности которых есть и зона фовеа/фовеолы, и фон (класс 0).
    """
    h, w = mask.shape
    
    # Ищем все точки, где в окрестности есть и зона фовеа, и фон
    left_point = None
    right_point = None
    
    for y in range(h):
        for x in range(w):
            # Проверяем соседей: должны быть и зона фовеа/фовеолы, и фон
            has_zone = False
            has_background = False
            
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if fovea_mask[ny, nx] == ind:
                            has_zone = True
                        if mask[ny, nx] == 0:
                            has_background = True
            
            # Если в окрестности есть и зона, и фон - это граничная точка
            if has_zone and has_background:
                if left_point is None or x < left_point[0]:
                    left_point = (x, y)
                if right_point is None or x > right_point[0]:
                    right_point = (x, y)
    
    if left_point is None or right_point is None:
        return None
    
    return left_point, right_point


def perpendicular_through_foveola(smooth_upper, spline, foveola_center, mask,
                                  perp_len=200):

    cx, cy = foveola_center  
    h, w = mask.shape

    if cx < 0 or cx >= len(smooth_upper):
        return None

    # верхняя граница (сглаженная)
    y0 = float(smooth_upper[cx])

    slope = float(spline.derivative()(cx))

    # нормаль вниз
    nx = -slope
    ny = 1.0
    L = np.sqrt(nx*nx + ny*ny)
    nx /= L
    ny /= L
    cts_dist = 0
    inside = False          # флаг "мы внутри хориоидеи"
    best_point = None

    for t in range(0, perp_len):
        xx = int(cx + nx * t)
        yy = int(y0 + ny * t)

        if xx < 0 or xx >= w or yy < 0 or yy >= h:
            break

        if mask[yy, xx] == CHOROID_ID:
            inside = True
            best_point = (xx, yy)
            cts_dist = int(np.sqrt((best_point[0] - cx) ** 2 + (best_point[1] - y0) ** 2))
        else:
            if inside:
                # мы были внутри и вышли — нижняя граница найдена
                break
            # если ещё не вошли → продолжаем искать вход
            continue

    if best_point is None:
        return None

    # расстояние считаем по t
    dist = int(np.sqrt((best_point[0] - cx)**2 + (best_point[1] - y0)**2))

    return cx, int(y0), dist, best_point


def measure_rpe_thickness(mask, fovea_mask, scale_x=None, scale_y=None, sample_interval=50):
    """
    Измеряет высоту (толщину) РПЭ методом скелетизации и построения перпендикуляров.
    
    Алгоритм:
    1. Скелетизирует РПЭ
    2. Строит перпендикуляры к линии скелета
    3. Измеряет часть перпендикуляра, проходящую через маску РПЭ
    
    Возвращает:
        - Список точек скелета для визуализации
        - Список перпендикуляров (пары точек) для визуализации
        - Среднюю высоту в пикселях или микрометрах
        - Стандартное отклонение высоты
    """
    from skimage.morphology import skeletonize
    
    # Создаем маску для РПЭ (класс 9)
    rpe_mask = (mask == 9).astype(np.uint8)
    
    if rpe_mask.sum() == 0:
        return None, None, None, None
    
    # Скелетизация РПЭ
    skeleton = skeletonize(rpe_mask > 0)
    skeleton_points = np.argwhere(skeleton > 0)
    
    if len(skeleton_points) == 0:
        return None, None, None, None
    
    h, w = mask.shape
    
    # Собираем измерения высоты и перпендикуляры для визуализации
    thicknesses = []
    perpendiculars = []
    
    # Берем точки скелета с интервалом для ускорения
    for i, point in enumerate(skeleton_points[::sample_interval]):
        y_skel, x_skel = point
        
        # Вычисляем касательную к скелету через ближайшие точки
        # Ищем соседние точки скелета в окрестности
        window_size = 10
        neighbors = []
        
        for dy in range(-window_size, window_size + 1):
            for dx in range(-window_size, window_size + 1):
                if dy == 0 and dx == 0:
                    continue
                yy = y_skel + dy
                xx = x_skel + dx
                if 0 <= yy < h and 0 <= xx < w and skeleton[yy, xx] > 0:
                    neighbors.append((xx, yy))
        
        if len(neighbors) < 2:
            # Недостаточно соседей, пропускаем эту точку
            continue
        
        # Используем линейную регрессию для аппроксимации локального направления скелета
        neighbors_x = np.array([n[0] for n in neighbors])
        neighbors_y = np.array([n[1] for n in neighbors])
        
        # Добавляем текущую точку
        neighbors_x = np.append(neighbors_x, x_skel)
        neighbors_y = np.append(neighbors_y, y_skel)
        
        # Вычисляем направление через PCA (главные компоненты)
        mean_x = np.mean(neighbors_x)
        mean_y = np.mean(neighbors_y)
        
        # Центрируем данные
        centered_x = neighbors_x - mean_x
        centered_y = neighbors_y - mean_y
        
        # Вычисляем ковариационную матрицу
        cov_xx = np.mean(centered_x * centered_x)
        cov_yy = np.mean(centered_y * centered_y)
        cov_xy = np.mean(centered_x * centered_y)
        
        # Собственные значения и векторы
        # Направление главной компоненты - это касательная к скелету
        trace = cov_xx + cov_yy
        det = cov_xx * cov_yy - cov_xy * cov_xy
        
        # Собственное значение (большее)
        lambda1 = trace / 2 + np.sqrt(max(0, (trace / 2) ** 2 - det))
        
        # Собственный вектор для lambda1
        if abs(cov_xy) > 1e-6:
            tangent_x = cov_xy
            tangent_y = lambda1 - cov_xx
        elif abs(cov_xx - lambda1) > 1e-6:
            tangent_x = 1.0
            tangent_y = 0.0
        else:
            tangent_x = 0.0
            tangent_y = 1.0
        
        # Нормализуем касательную
        tangent_norm = np.sqrt(tangent_x**2 + tangent_y**2)
        if tangent_norm > 0:
            tangent_x /= tangent_norm
            tangent_y /= tangent_norm
        else:
            # Если не удалось вычислить, используем вертикальное направление
            tangent_x, tangent_y = 0, 1
        
        # Перпендикуляр к касательной: если касательная (tx, ty), то перпендикуляр (-ty, tx)
        perp_x = -tangent_y
        perp_y = tangent_x
        
        # Строим перпендикуляр в обе стороны от точки скелета
        max_len = 50  # Максимальная длина перпендикуляра
        
        # Ищем границы РПЭ вдоль перпендикуляра
        boundary_1 = None  # Первая граница
        boundary_2 = None  # Вторая граница
        
        # Идем в одну сторону по перпендикуляру до границы РПЭ
        for t in range(0, max_len):
            xx = int(x_skel + perp_x * t)
            yy = int(y_skel + perp_y * t)
            if xx < 0 or xx >= w or yy < 0 or yy >= h:
                break
            if rpe_mask[yy, xx] > 0:
                boundary_1 = (xx, yy)
            else:
                break
        
        # Идем в другую сторону по перпендикуляру до границы РПЭ
        for t in range(0, max_len):
            xx = int(x_skel - perp_x * t)
            yy = int(y_skel - perp_y * t)
            if xx < 0 or xx >= w or yy < 0 or yy >= h:
                break
            if rpe_mask[yy, xx] > 0:
                boundary_2 = (xx, yy)
            else:
                break
        
        # Если нашли обе границы, вычисляем расстояние между ними
        if boundary_1 is not None and boundary_2 is not None and boundary_1 != boundary_2:
            # Расстояние между двумя точками с учетом масштаба
            if scale_x is not None and scale_y is not None:
                dx_px = abs(boundary_1[0] - boundary_2[0])
                dy_px = abs(boundary_1[1] - boundary_2[1])
                dx_um = dx_px * scale_x
                dy_um = dy_px * scale_y
                thickness = np.sqrt(dx_um**2 + dy_um**2)
            else:
                dx_px = boundary_1[0] - boundary_2[0]
                dy_px = boundary_1[1] - boundary_2[1]
                thickness = np.sqrt(dx_px**2 + dy_px**2)
            
            thicknesses.append(thickness)
            perpendiculars.append((boundary_1, boundary_2))
    
    if len(thicknesses) == 0:
        return skeleton_points, [], None, None
    
    # Вычисляем статистику
    mean_thickness = np.mean(thicknesses)
    std_thickness = np.std(thicknesses)
    
    return skeleton_points, perpendiculars, mean_thickness, std_thickness

def analyze_rpe(mask, fovea_mask, smooth_upper, spline):
    """
    Анализирует состояние РПЭ (ретинального пигментного эпителия).
    
    Возвращает одно из состояний:
    - "Эпителий сохранен" - нет разрывов, равномерная высота
    - "Эпителий неравномерный" - нет разрывов, но высота меняется
    - "Единичные разрывы" - до 3 разрывов
    - "Множественные разрывы" - больше 3 разрывов
    - "Эпителий не определяется" - не виден или отсутствует
    """
    from skimage.morphology import skeletonize
    
    # Создаем маску для РПЭ (класс 9)
    rpe_mask = (mask == 9).astype(np.uint8)
    
    # Проверяем, есть ли вообще РПЭ
    if rpe_mask.sum() == 0:
        return "Эпителий не определяется"
    
    # Проверяем минимальное количество пикселей РПЭ
    if rpe_mask.sum() < 100:  # Если РПЭ слишком мало
        return "Эпителий не определяется"
    
    # Скелетизация РПЭ
    skeleton = skeletonize(rpe_mask > 0)
    
    # Находим контуры РПЭ для определения разрывов
    contours, _ = cv2.findContours(rpe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    num_gaps = len(contours) - 1  # Количество разрывов = количество контуров - 1
    
    # Если разрывов много, классифицируем
    if num_gaps > 3:
        return "Множественные разрывы"
    elif num_gaps > 0:
        return "Единичные разрывы"
    
    # Если разрывов нет, проверяем равномерность высоты
    # Для этого используем скелет и вычисляем перпендикуляры
    skeleton_points = np.argwhere(skeleton > 0)
    
    if len(skeleton_points) == 0:
        return "Эпителий не определяется"
    
    # Вычисляем высоту РПЭ в разных точках
    heights = []
    h, w = mask.shape
    
    for point in skeleton_points[::10]:  # Берем каждую 10-ю точку для ускорения
        y_skel, x_skel = point
        
        # Вычисляем перпендикуляр к скелету
        # Упрощенно: считаем высоту как количество пикселей РПЭ по вертикали в этой точке
        height_count = 0
        for dy in range(-20, 20):
            yy = y_skel + dy
            if 0 <= yy < h and 0 <= x_skel < w:
                if rpe_mask[yy, x_skel] > 0:
                    height_count += 1
        
        if height_count > 0:
            heights.append(height_count)
    
    if len(heights) == 0:
        return "Эпителий не определяется"
    
    # Анализируем вариацию высоты
    heights = np.array(heights)
    height_std = np.std(heights)
    height_mean = np.mean(heights)
    
    # Коэффициент вариации
    cv = height_std / height_mean if height_mean > 0 else 0
    
    # Если коэффициент вариации низкий - эпителий равномерный
    if cv < 0.2:  # Менее 20% вариации
        return "Эпителий сохранен"
    else:
        return "Эпителий неравномерный"

def detect_rpe_defects(mask, fovea_mask, smooth_upper, spline, min_gap_width=10):
    """
    Определяет локализацию дефектов РПЭ.
    
    Дефектами РПЭ считаются его истончения и разрывы.
    Отслойка от мембраны Бруха без повреждения целостности РПЭ дефектом не является.
    
    Возвращает кортеж: (строка_с_локализацией, список_контуров_разрывов, координата_дефекта)
    
    Строка с локализацией:
    • 0 – Дефекты отсутствуют
    • 1 – Фовеа + макула (без фовеолы) - хотя бы один дефект присутствует в фовеа
    • 2 – Фовеола + фовеа + макула - хотя бы один дефект присутствует в фовеоле
    • 3 – Макула (без фовеа и фовеолы) - дефекты только в макуле; фовеа и фовеола без дефектов
    
    координата_дефекта - координата (x, y) центра первого дефекта или None
    
    Args:
        min_gap_width: минимальная ширина разрыва в пикселях (по умолчанию 10)
    """
    # Создаем маску для РПЭ (класс 9)
    rpe_mask = (mask == 9).astype(np.uint8)
    
    if rpe_mask.sum() == 0:
        return "0 – Дефекты отсутствуют", [], None
    
    # Находим контуры РПЭ
    contours, _ = cv2.findContours(rpe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) <= 1:
        return "0 – Дефекты отсутствуют", [], None
    
    h, w = mask.shape
    
    # Фильтруем контуры: оставляем только значимые (не очень маленькие)
    # и определяем разрывы между ними
    significant_contours = []
    gap_regions = []  # Список областей разрывов для визуализации
    defect_coordinate = None  # Координата первого дефекта
    
    # Сортируем контуры по X-координате их центра
    contours_with_centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Получаем bounding box для определения размера
            x, y, w_box, h_box = cv2.boundingRect(contour)
            if w_box >= min_gap_width:  # Контур достаточно широкий
                contours_with_centers.append((cx, cy, contour, x, y, w_box, h_box))
    
    if len(contours_with_centers) <= 1:
        return "0 – Дефекты отсутствуют", [], None
    
    # Сортируем по X
    contours_with_centers.sort(key=lambda item: item[0])
    
    # Определяем значимые разрывы между контурами
    for i in range(len(contours_with_centers) - 1):
        cx1, cy1, cnt1, x1, y1, w1, h1 = contours_with_centers[i]
        cx2, cy2, cnt2, x2, y2, w2, h2 = contours_with_centers[i + 1]
        
        # Ширина разрыва = расстояние между правым краем левого контура и левым краем правого
        gap_width = x2 - (x1 + w1)
        
        if gap_width >= min_gap_width:
            # Центр разрыва для визуализации
            gap_x_center = (x1 + w1 + x2) // 2
            gap_y_center = (y1 + y2 + h1 + h2) // 4
            
            # Сохраняем координату первого дефекта
            if defect_coordinate is None:
                defect_coordinate = (gap_x_center, gap_y_center)
            
            # Сохраняем координаты центра и радиус для визуализации окружности
            gap_radius = max(gap_width, max(h1, h2)) // 2
            gap_regions.append((gap_x_center, gap_y_center, gap_radius))
            
            significant_contours.append(cnt1)
    
    # Добавляем последний контур
    if contours_with_centers:
        significant_contours.append(contours_with_centers[-1][2])
    
    # Если нет значимых разрывов
    if len(gap_regions) == 0:
        return "0 – Дефекты отсутствуют", [], None
    
    # Анализируем локализацию разрывов
    defects_in_foveola = False
    defects_in_fovea = False
    defects_in_macula = False
    
    for gap_x_center, gap_y_center, gap_radius in gap_regions:
        if 0 <= gap_y_center < h and 0 <= gap_x_center < w:
            # Проверяем к какой зоне относится
            zone = fovea_mask[gap_y_center, gap_x_center]
            if zone == 1:  # Фовеола
                defects_in_foveola = True
            elif zone == 2:  # Фовеа
                defects_in_fovea = True
            else:
                defects_in_macula = True
    
    # Применяем алгоритм определения:
    # Если дефектов нет → 0
    if not (defects_in_foveola or defects_in_fovea or defects_in_macula):
        result = "0 – Дефекты отсутствуют"
    # Иначе если хоть один дефект в фовеоле → 2
    elif defects_in_foveola:
        result = "2 – Фовеола + фовеа + макула"
    # Иначе если хоть один дефект в фовеа → 1
    elif defects_in_fovea:
        result = "1 – Фовеа + макула (без фовеолы)"
    # Иначе → 3
    else:
        result = "3 – Макула (без фовеа и фовеолы)"
    
    return result, gap_regions, defect_coordinate

def detect_and_measure_detachments(mask, smooth_upper, spline, target_class, fovea_mask=None, scale_x=None, scale_y=None, perp_len=400):
    """
    Измеряет отслойку и определяет её локализацию.
    
    Возвращает: (width, height, area, left_x, right_x, max_perp_point, location, choroid_segment)
    где:
        width - длина сглаженной верхней линии хориоидеи под отслойкой
        height - наибольший перпендикуляр к верхней границе хориоидеи
        area - площадь отслойки
        left_x, right_x - границы для визуализации
        max_perp_point - точка максимального перпендикуляра (x, y_base, y_top)
        location - строка с локализацией ('фовеола', 'фовеа', 'макула', или комбинация)
        choroid_segment - список точек сглаженной линии хориоидеи под отслойкой для визуализации
    """
    h, w = mask.shape
    target_mask = (mask == target_class).astype(np.uint8)
    
    # Вычисляем площадь отслойки (количество пикселей)
    area_px = np.sum(target_mask)
    if area_px == 0:
        return None, None, None, None, None, None, None, None
    
    # Площадь в мкм² если есть масштаб
    if scale_x is not None and scale_y is not None:
        area_um2 = area_px * scale_x * scale_y
    else:
        area_um2 = None
    
    # Найдем левую и правую границы отслойки по горизонтали
    # Ищем минимальный и максимальный x, где есть пиксели отслойки
    ys, xs = np.where(target_mask == 1)
    if len(xs) == 0:
        return None, None, None, None, None, None, None, None
    
    left_x = int(np.min(xs))
    right_x = int(np.max(xs))
    
    # 1. ШИРИНА: Длина сглаженной верхней линии хориоидеи от left_x до right_x
    choroid_segment = []
    width_um = 0.0
    
    for x in range(left_x, right_x + 1):
        y = smooth_upper[x]
        choroid_segment.append((x, int(y)))
        if x > left_x:
            # Расстояние до предыдущей точки в пикселях
            dx_px = 1
            dy_px = smooth_upper[x] - smooth_upper[x - 1]
            
            # Переводим компоненты в микрометры и вычисляем длину отрезка кривой
            if scale_x is not None and scale_y is not None:
                dx_um = dx_px * scale_x
                dy_um = dy_px * scale_y
                dist_um = np.sqrt(dx_um**2 + dy_um**2)
                width_um += dist_um
    
    # Если масштабы не заданы, результат None
    if scale_x is None or scale_y is None:
        width_um = None
    
    # 2. ВЫСОТА: Наибольший перпендикуляр к smooth_upper, проходящий через отслойку
    max_height_px = 0
    max_perp_point = None
    
    debug_printed = False  # Печатаем отладку только для первой точки
    
    for x in range(left_x, right_x + 1):
        y_base = int(smooth_upper[x])
        
        # Вычисляем перпендикуляр используя производную сплайна
        slope = float(spline.derivative()(x))
        
        # Перпендикуляр к касательной (1, slope) направленный ВВЕРХ
        # Из условия перпендикулярности: 1*nx + slope*ny = 0
        # При ny = -1 (вверх): nx = slope
        nx = slope
        ny = -1.0  # Вверх (y уменьшается)
        L = np.sqrt(nx*nx + ny*ny)
        nx /= L
        ny /= L
        
        # ОТЛАДКА: выводим информацию для первой точки
        if not debug_printed:
            print(f"\n=== ОТЛАДКА ПЕРПЕНДИКУЛЯРА (отслойка) ===")
            print(f"Точка на хориоидее: x={x}, y_base={y_base}")
            print(f"Наклон касательной (slope): {slope:.4f}")
            print(f"Перпендикуляр ДО нормализации: nx={-slope:.4f}, ny={-1.0:.4f}")
            print(f"Перпендикуляр ПОСЛЕ нормализации: nx={nx:.4f}, ny={ny:.4f}")
            print(f"Длина вектора L: {L:.4f}")
        
        # Идем по перпендикуляру в ОБЕ СТОРОНЫ от точки (x, y_base) на хориоидее
        points_in_detachment = []
        
        # Используем float для точного движения по перпендикуляру
        fx, fy = float(x), float(y_base)
        
        # Направление 1: ВВЕРХ (ny < 0, y уменьшается)
        for t in range(0, perp_len):
            fx_new = fx + nx * t
            fy_new = fy + ny * t
            
            px = int(round(fx_new))
            py = int(round(fy_new))
            
            if not debug_printed and t < 5:
                print(f"  ВВЕРХ t={t}: fx_new={fx_new:.2f}, fy_new={fy_new:.2f}, px={px}, py={py}, delta_x={px-x}, delta_y={py-y_base}")
            
            if px < 0 or px >= w or py < 0 or py >= h:
                break
            
            if target_mask[py, px] == 1:
                points_in_detachment.append((fx_new, fy_new))
                if not debug_printed:
                    print(f"  >>> Нашли точку в отслойке (вверх): ({fx_new:.2f}, {fy_new:.2f}) -> px={px}, py={py}")
        
        # Направление 2: ВНИЗ (противоположное направление)
        for t in range(1, perp_len):  # начинаем с 1, чтобы не дублировать t=0
            fx_new = fx - nx * t  # обратное направление
            fy_new = fy - ny * t
            
            px = int(round(fx_new))
            py = int(round(fy_new))
            
            if not debug_printed and t < 5:
                print(f"  ВНИЗ t={t}: fx_new={fx_new:.2f}, fy_new={fy_new:.2f}, px={px}, py={py}, delta_x={px-x}, delta_y={py-y_base}")
            
            if px < 0 or px >= w or py < 0 or py >= h:
                break
            
            if target_mask[py, px] == 1:
                points_in_detachment.append((fx_new, fy_new))
                if not debug_printed:
                    print(f"  >>> Нашли точку в отслойке (вниз): ({fx_new:.2f}, {fy_new:.2f}) -> px={px}, py={py}")
        
        if not debug_printed:
            debug_printed = True
            print(f"Всего точек в отслойке: {len(points_in_detachment)}")
            print("==========================================\n")
        
        # Если нашли точки в отслойке, вычисляем длину перпендикуляра
        if len(points_in_detachment) >= 2:
            # Находим две самые удаленные друг от друга точки
            max_dist = 0
            point1 = None
            point2 = None
            
            for i in range(len(points_in_detachment)):
                for j in range(i + 1, len(points_in_detachment)):
                    dx = points_in_detachment[j][0] - points_in_detachment[i][0]
                    dy = points_in_detachment[j][1] - points_in_detachment[i][1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist > max_dist:
                        max_dist = dist
                        point1 = points_in_detachment[i]
                        point2 = points_in_detachment[j]
            
            if point1 is not None and point2 is not None and max_dist > max_height_px:
                max_height_px = max_dist
                # Сохраняем обе крайние точки
                max_perp_point = (point1[0], point1[1], point2[0], point2[1])
    
    # Преобразуем в микрометры (учитываем наклон перпендикуляра)
    if max_height_px > 0 and max_perp_point is not None:
        if scale_x is not None and scale_y is not None:
            # Для наклонного перпендикуляра нужно учесть оба масштаба
            point1_x, point1_y, point2_x, point2_y = max_perp_point
            dx_px = point2_x - point1_x
            dy_px = point2_y - point1_y
            # Евклидово расстояние в микрометрах
            dx_um = dx_px * scale_x
            dy_um = dy_px * scale_y
            max_height_um = np.sqrt(dx_um**2 + dy_um**2)
        else:
            max_height_um = None
        
        # ОТЛАДКА: выводим финальные координаты
        if max_perp_point is not None:
            x_base, y_base_final, x_top, y_top_final = max_perp_point
            print(f"\n=== ФИНАЛЬНЫЙ РАСЧЕТ ВЫСОТЫ (отслойка) ===")
            print(f"max_height_px = {max_height_px:.2f} пикселей")
            print(f"max_height_um = {max_height_um:.2f} мкм")
            print(f"Начальная точка (на хориоидее): ({x_base}, {y_base_final})")
            print(f"Конечная точка (в отслойке): ({x_top:.2f}, {y_top_final:.2f})")
            print(f"Расстояние: sqrt(({x_top:.2f}-{x_base})^2 + ({y_top_final:.2f}-{y_base_final})^2) = {max_height_px:.2f}")
            print("==========================================\n")
    else:
        max_height_um = None
    
    # Определяем локализацию: проверяем совпадение X-координат объекта с зонами фовеа
    # Если у объекта и фовеолы/фовеа есть хотя бы одна общая X-координата - объект в этой зоне
    location = None
    if fovea_mask is not None:
        in_foveola = False
        in_fovea = False
        in_macula = False
        
        # Получаем уникальные X координаты отслойки
        ys, xs = np.where(target_mask == 1)
        if len(xs) > 0:
            unique_xs_detachment = set(xs)
            
            # Получаем X координаты для каждой зоны фовеа
            h, w = fovea_mask.shape
            for x in unique_xs_detachment:
                if x >= w:
                    continue
                # Проверяем все Y в этом X в маске фовеа
                column = fovea_mask[:, x]
                if 1 in column:  # Фовеола
                    in_foveola = True
                if 2 in column:  # Фовеа
                    in_fovea = True
                if 0 in column or 3 in column:  # Макула
                    in_macula = True
                
                # Если уже нашли фовеолу, можем выйти
                if in_foveola:
                    break
        
        # Логика определения локализации по приоритету
        # 2 – Фовеола + фовеа + макула (хотя бы одна общая X с фовеолой)
        if in_foveola:
            location = "2 – Фовеола + фовеа + макула"
        # 1 – Фовеа + макула (без фовеолы, но хотя бы одна общая X с фовеа)
        elif in_fovea:
            location = "1 – Фовеа + макула (без фовеолы)"
        # 3 – Макула (без фовеа и фовеолы)
        elif in_macula:
            location = "3 – Макула (без фовеа и фовеолы)"
        else:
            location = "0 – отсутствует"
    
    print(f"Detachment: width_um={width_um}, height_um={max_height_um}, area_um2={area_um2}, location={location}")
    return width_um, max_height_um, area_um2, left_x, right_x, max_perp_point, location, choroid_segment


def measure_drusen(mask, smooth_upper, spline, contour, fovea_mask=None, scale_x=None, scale_y=None, perp_len=400):
    """
    Измеряет одну друзу (контур) относительно сглаженной линии хориоидеи.
    
    - Ширина = расстояние между самой левой и самой правой точками друзы (с учетом scale_x и scale_y)
    - Высота = максимальный перпендикуляр к линии хориоидеи (как для отслоек)
    
    Args:
        contour: контур конкретной друзы
    
    Возвращает: (width, height, area, left_x, right_x, max_perp_point, location, choroid_segment)
    """
    h, w = mask.shape
    
    # Вычисляем площадь друзы напрямую из контура
    area_px = cv2.contourArea(contour)
    if area_px == 0:
        return None, None, None, None, None, None, None, None
    
    # Площадь в мкм² если есть масштаб
    if scale_x is not None and scale_y is not None:
        area_um2 = area_px * scale_x * scale_y
    else:
        area_um2 = None
    
    # Создаем маску для этого конкретного контура (для измерения ширины и высоты)
    target_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(target_mask, [contour], -1, 1, -1)
    
    # Найдем левую и правую границы друзы по горизонтали
    ys, xs = np.where(target_mask == 1)
    if len(xs) == 0:
        return None, None, None, None, None, None, None, None
    
    left_x = int(np.min(xs))
    right_x = int(np.max(xs))
    
    # Находим самую левую и самую правую точки друзы
    left_idx = np.argmin(xs)
    right_idx = np.argmax(xs)
    
    left_point_x = xs[left_idx]
    left_point_y = ys[left_idx]
    right_point_x = xs[right_idx]
    right_point_y = ys[right_idx]
    
    # 1. ШИРИНА: Расстояние между самой левой и самой правой точками друзы
    # с учетом обоих масштабов
    if scale_x is not None and scale_y is not None:
        dx_px = right_point_x - left_point_x
        dy_px = right_point_y - left_point_y
        
        dx_um = dx_px * scale_x
        dy_um = dy_px * scale_y
        
        width_um = np.sqrt(dx_um**2 + dy_um**2)
    else:
        width_um = None
    
    # Сохраняем сегмент линии хориоидеи для визуализации
    choroid_segment = []
    for x in range(left_x, right_x + 1):
        y = smooth_upper[x]
        choroid_segment.append((x, int(y)))
    
    # 2. ВЫСОТА: Наибольший перпендикуляр к smooth_upper, проходящий через друзу
    max_height_px = 0
    max_perp_point = None
    
    for x in range(left_x, right_x + 1):
        y_base = int(smooth_upper[x])
        
        # Вычисляем перпендикуляр используя производную сплайна
        slope = float(spline.derivative()(x))
        
        # Перпендикуляр к касательной (1, slope) направленный ВВЕРХ
        # Из условия перпендикулярности: 1*nx + slope*ny = 0
        # При ny = -1 (вверх): nx = slope
        nx = slope
        ny = -1.0  # Вверх (y уменьшается)
        L = np.sqrt(nx*nx + ny*ny)
        nx /= L
        ny /= L
        
        # Идем по перпендикуляру в ОБЕ СТОРОНЫ от точки (x, y_base) на хориоидее
        points_in_drusen = []
        
        # Используем float для точного движения по перпендикуляру
        fx, fy = float(x), float(y_base)
        
        # Направление 1: ВВЕРХ (ny < 0, y уменьшается)
        for t in range(0, perp_len):
            fx_new = fx + nx * t
            fy_new = fy + ny * t
            
            px = int(round(fx_new))
            py = int(round(fy_new))
            
            if px < 0 or px >= w or py < 0 or py >= h:
                break
            
            if target_mask[py, px] == 1:
                points_in_drusen.append((fx_new, fy_new))
        
        # Направление 2: ВНИЗ (противоположное направление)
        for t in range(1, perp_len):  # начинаем с 1, чтобы не дублировать t=0
            fx_new = fx - nx * t  # обратное направление
            fy_new = fy - ny * t
            
            px = int(round(fx_new))
            py = int(round(fy_new))
            
            if px < 0 or px >= w or py < 0 or py >= h:
                break
            
            if target_mask[py, px] == 1:
                points_in_drusen.append((fx_new, fy_new))
        
        # Если нашли точки в друзе, вычисляем длину перпендикуляра
        if len(points_in_drusen) >= 2:
            # Находим две самые удаленные друг от друга точки
            max_dist = 0
            point1 = None
            point2 = None
            
            for i in range(len(points_in_drusen)):
                for j in range(i + 1, len(points_in_drusen)):
                    dx = points_in_drusen[j][0] - points_in_drusen[i][0]
                    dy = points_in_drusen[j][1] - points_in_drusen[i][1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist > max_dist:
                        max_dist = dist
                        point1 = points_in_drusen[i]
                        point2 = points_in_drusen[j]
            
            if point1 is not None and point2 is not None and max_dist > max_height_px:
                max_height_px = max_dist
                # Сохраняем обе крайние точки
                max_perp_point = (point1[0], point1[1], point2[0], point2[1])
    
    # Преобразуем в микрометры (учитываем наклон перпендикуляра)
    if max_height_px > 0 and max_perp_point is not None:
        if scale_x is not None and scale_y is not None:
            # Для наклонного перпендикуляра нужно учесть оба масштаба
            point1_x, point1_y, point2_x, point2_y = max_perp_point
            dx_px = point2_x - point1_x
            dy_px = point2_y - point1_y
            # Евклидово расстояние в микрометрах
            dx_um = dx_px * scale_x
            dy_um = dy_px * scale_y
            max_height_um = np.sqrt(dx_um**2 + dy_um**2)
        else:
            max_height_um = None
    else:
        max_height_um = None
    
    # Определяем локализацию для этой конкретной друзы
    # Проверяем совпадение X-координат друзы с зонами фовеа
    location = None
    if fovea_mask is not None:
        in_foveola = False
        in_fovea = False
        in_macula = False
        
        # Получаем уникальные X координаты друзы
        ys, xs = np.where(target_mask == 1)
        if len(xs) > 0:
            unique_xs_drusen = set(xs)
            
            # Получаем X координаты для каждой зоны фовеа
            h, w = fovea_mask.shape
            for x in unique_xs_drusen:
                if x >= w:
                    continue
                # Проверяем все Y в этом X в маске фовеа
                column = fovea_mask[:, x]
                if 1 in column:  # Фовеола
                    in_foveola = True
                if 2 in column:  # Фовеа
                    in_fovea = True
                if 0 in column or 3 in column:  # Макула
                    in_macula = True
                
                # Если уже нашли фовеолу, можем выйти
                if in_foveola:
                    break
        
        # Логика определения локализации по приоритету
        if in_foveola:
            location = "2 – Фовеола + фовеа + макула"
        elif in_fovea:
            location = "1 – Фовеа + макула (без фовеолы)"
        elif in_macula:
            location = "3 – Макула (без фовеа и фовеолы)"
        else:
            location = "0 – отсутствует"
    
    print(f"Drusen: width_um={width_um}, height_um={max_height_um}, area_um2={area_um2}, location={location}")
    return width_um, max_height_um, area_um2, left_x, right_x, max_perp_point, location, choroid_segment


def measure_neuroepithelial_detachment(mask, smooth_upper_choroid, spline_choroid, target_class, fovea_mask=None, scale_x=None, scale_y=None, perp_len=400):
    """
    Измеряет отслойку нейроэпителия (класс 6 - СРЖ).
    
    - Ширина = длина кривой сглаженной ВЕРХНЕЙ границы самой отслойки (не хориоидеи!)
    - Высота = максимальный перпендикуляр к линии хориоидеи (как для других отслоек)
    
    Возвращает: (width, height, area, left_x, right_x, max_perp_point, location, upper_boundary_segment)
    """
    h, w = mask.shape
    
    # Извлекаем маску отслойки
    detachment_mask = (mask == target_class).astype(np.uint8)
    
    # Вычисляем площадь
    area_px = np.sum(detachment_mask)
    if area_px == 0:
        return None, None, None, None, None, None, None, None
    
    if scale_x is not None and scale_y is not None:
        area_um2 = area_px * scale_x * scale_y
    else:
        area_um2 = None
    
    # Находим границы отслойки
    ys, xs = np.where(detachment_mask == 1)
    if len(xs) == 0:
        return None, None, None, None, None, None, None, None
    
    left_x = int(np.min(xs))
    right_x = int(np.max(xs))
    
    # Находим ВЕРХНЮЮ границу отслойки (минимальный Y для каждого X)
    upper_boundary = np.full(w, -1, dtype=np.float32)
    
    for x in range(left_x, right_x + 1):
        column_ys = ys[xs == x]
        if len(column_ys) > 0:
            upper_boundary[x] = np.min(column_ys)
    
    # Сглаживаем верхнюю границу отслойки сплайном
    valid_xs = []
    valid_ys = []
    for x in range(left_x, right_x + 1):
        if upper_boundary[x] >= 0:
            valid_xs.append(x)
            valid_ys.append(upper_boundary[x])
    
    if len(valid_xs) < 4:  # нужно минимум 4 точки для сплайна
        return None, None, None, None, None, None, None, None
    
    # Создаем сплайн верхней границы отслойки
    try:
        spline_upper = UnivariateSpline(valid_xs, valid_ys, s=len(valid_xs) * 2, k=3)
        smooth_upper_detachment = np.array([spline_upper(x) for x in range(left_x, right_x + 1)])
    except:
        return None, None, None, None, None, None, None, None
    
    # 1. ШИРИНА: Длина кривой сглаженной верхней границы отслойки
    width_um = 0
    upper_boundary_segment = []
    
    for i, x in enumerate(range(left_x, right_x + 1)):
        y = smooth_upper_detachment[i]
        upper_boundary_segment.append((x, int(y)))
    
    # Вычисляем длину кривой
    if scale_x is not None and scale_y is not None:
        for i in range(len(upper_boundary_segment) - 1):
            x1, y1 = upper_boundary_segment[i]
            x2, y2 = upper_boundary_segment[i + 1]
            
            dx_px = x2 - x1
            dy_px = y2 - y1
            
            dx_um = dx_px * scale_x
            dy_um = dy_px * scale_y
            
            width_um += np.sqrt(dx_um**2 + dy_um**2)
    else:
        width_um = None
    
    # 2. ВЫСОТА: Максимальный перпендикуляр к хориоидее (как для других отслоек)
    max_height_px = 0
    max_perp_point = None
    
    for x in range(left_x, right_x + 1):
        if x < 0 or x >= len(smooth_upper_choroid):
            continue
        
        # Точка на линии хориоидеи
        fx = float(x)
        fy = smooth_upper_choroid[x]
        
        # Вычисляем производную сплайна хориоидеи
        try:
            slope = float(spline_choroid.derivative()(fx))
        except:
            continue
        
        # Перпендикуляр: если касательная (1, slope), то перпендикуляр (slope, -1)
        perp_dir = np.array([slope, -1.0])
        norm = np.linalg.norm(perp_dir)
        if norm < 1e-9:
            continue
        
        nx = perp_dir[0] / norm
        ny = perp_dir[1] / norm
        
        # Ищем точки отслойки вдоль перпендикуляра (в обе стороны)
        points_in_detachment = []
        
        # Направление 1: ВВЕРХ
        for t in range(perp_len):
            fx_new = fx + nx * t
            fy_new = fy + ny * t
            
            px = int(round(fx_new))
            py = int(round(fy_new))
            
            if px < 0 or px >= w or py < 0 or py >= h:
                break
            
            if detachment_mask[py, px] == 1:
                points_in_detachment.append((fx_new, fy_new))
        
        # Направление 2: ВНИЗ
        for t in range(1, perp_len):
            fx_new = fx - nx * t
            fy_new = fy - ny * t
            
            px = int(round(fx_new))
            py = int(round(fy_new))
            
            if px < 0 or px >= w or py < 0 or py >= h:
                break
            
            if detachment_mask[py, px] == 1:
                points_in_detachment.append((fx_new, fy_new))
        
        # Находим две самые удаленные точки
        if len(points_in_detachment) >= 2:
            max_dist = 0
            point1 = None
            point2 = None
            
            for i in range(len(points_in_detachment)):
                for j in range(i + 1, len(points_in_detachment)):
                    dx = points_in_detachment[j][0] - points_in_detachment[i][0]
                    dy = points_in_detachment[j][1] - points_in_detachment[i][1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist > max_dist:
                        max_dist = dist
                        point1 = points_in_detachment[i]
                        point2 = points_in_detachment[j]
            
            if point1 is not None and point2 is not None and max_dist > max_height_px:
                max_height_px = max_dist
                max_perp_point = (point1[0], point1[1], point2[0], point2[1])
    
    # Преобразуем высоту в микрометры
    if max_height_px > 0 and max_perp_point is not None:
        if scale_x is not None and scale_y is not None:
            point1_x, point1_y, point2_x, point2_y = max_perp_point
            dx_px = point2_x - point1_x
            dy_px = point2_y - point1_y
            dx_um = dx_px * scale_x
            dy_um = dy_px * scale_y
            max_height_um = np.sqrt(dx_um**2 + dy_um**2)
        else:
            max_height_um = None
    else:
        max_height_um = None
    
    # Определяем локализацию: проверяем совпадение X-координат отслойки с зонами фовеа
    location = None
    if fovea_mask is not None:
        in_foveola = False
        in_fovea = False
        in_macula = False
        
        # Получаем уникальные X координаты отслойки
        ys_det, xs_det = np.where(detachment_mask == 1)
        if len(xs_det) > 0:
            unique_xs_detachment = set(xs_det)
            
            # Получаем X координаты для каждой зоны фовеа
            h, w = fovea_mask.shape
            for x in unique_xs_detachment:
                if x >= w:
                    continue
                # Проверяем все Y в этом X в маске фовеа
                column = fovea_mask[:, x]
                if 1 in column:  # Фовеола
                    in_foveola = True
                if 2 in column:  # Фовеа
                    in_fovea = True
                if 0 in column or 3 in column:  # Макула
                    in_macula = True
                
                # Если уже нашли фовеолу, можем выйти
                if in_foveola:
                    break
        
        # Логика определения локализации по приоритету
        if in_foveola:
            location = "2 – Фовеола + фовеа + макула"
        elif in_fovea:
            location = "1 – Фовеа + макула (без фовеолы)"
        elif in_macula:
            location = "3 – Макула (без фовеа и фовеолы)"
        else:
            location = "0 – отсутствует"
        if in_foveola:
            location = "0 – Фовеола"
        elif in_fovea:
            location = "1 – Фовеа (без фovеолы)"
        elif in_macula:
            location = "2 – Макула (без фовеолы и фовеа)"
        else:
            location = "3 – Вне макулы"
    
    print(f"Neuroepithelial detachment: width_um={width_um}, height_um={max_height_um}, area_um2={area_um2}, location={location}")
    return width_um, max_height_um, area_um2, left_x, right_x, max_perp_point, location, upper_boundary_segment


# ============================================================
#     ВИЗУАЛИЗАЦИЯ
# ============================================================
def visualize(img, mask, fovea_mask, smooth_upper, spline, out_path):

    vis = img.copy()

    # Хория
    overlay = vis.copy()
    overlay[mask == CHOROID_ID] = CHOROID_COLOR
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

    # Наложение слоев патологий
    for layer_id in RETINA_LAYERS:
        layer_overlay = vis.copy()
        layer_overlay[mask == layer_id] = RETINA_LAYER_COLORS[layer_id]  # Наложение соответствующего цвета для каждого слоя
        vis = cv2.addWeighted(layer_overlay, 0.35, vis, 0.65, 0)

    # Фовеола / фовеа
    fov_overlay = vis.copy()
    fov_overlay[fovea_mask == 1] = FOVEOLA_COLOR
    fov_overlay[fovea_mask == 2] = FOVEA_COLOR
    vis = cv2.addWeighted(fov_overlay, 0.35, vis, 0.65, 0)

    # Сглаженная линия
    for x in range(len(smooth_upper) - 1):
        cv2.line(vis,
                 (x, int(smooth_upper[x])),
                 (x + 1, int(smooth_upper[x + 1])),
                 LINE_COLOR, 1, cv2.LINE_AA)

    # Центр фовеолы
    center = get_foveola_center(fovea_mask)
    if center is None:
        cv2.imwrite(out_path, vis)
        return

    cx, cy = center
    cv2.circle(vis, (cx, cy), 4, (255, 0, 0), -1)

    # Перпендикуляр и ЦТС
    result = perpendicular_through_foveola(smooth_upper, spline, center, mask)
    width = central_width_of_retina(smooth_upper, spline, center, mask)
    left_point, right_point = central_width_near(img, smooth_upper, spline, fovea_mask, mask, 1)
    # Убедитесь, что left_point и right_point всегда кортежи (x, y)
    wl = central_width_of_retina(smooth_upper, spline, left_point, mask)
    wr = central_width_of_retina(smooth_upper, spline, right_point, mask)
    avg_width_near_foveolla = (wl[2] + wr[2]) // 2
    if left_point is not None:
        cv2.circle(vis, left_point, 5, (255, 0, 0), -1)  # Левая точка
    if right_point is not None:
        cv2.circle(vis, right_point, 5, (255, 0, 0), -1)  # Правая точка
    left_point, right_point = central_width_near(img, smooth_upper, spline, fovea_mask, mask, 2)
    # Убедитесь, что left_point и right_point всегда кортежи (x, y)
    wl1 = central_width_of_retina(smooth_upper, spline, left_point, mask)
    wr1 = central_width_of_retina(smooth_upper, spline, right_point, mask)
    avg_width_near_fovea = (wl1[2] + wr1[2]) // 2
    if left_point is not None:
        cv2.circle(vis, left_point, 5, (255, 0, 255), -1)  # Левая точка
    if right_point is not None:
        cv2.circle(vis, right_point, 5, (255, 0, 255), -1)  # Правая точка

    if result is not None:
        x0, y0, dist, pt= result

        # Линия перпендикуляра
        cv2.line(vis, (x0, y0), pt, (0, 0, 255), 2, cv2.LINE_AA)

        # Подпись толщины
        cv2.putText(vis, f"{dist}", (x0 + 5, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    for width in [width, wl, wr, wl1, wr1]:
        if width is not None:
            x0, y0, dist, pt= width

            # Линия перпендикуляра
            cv2.line(vis, (x0, y0), pt, (255, 0, 0), 2, cv2.LINE_AA)

            # Подпись толщины
            cv2.putText(vis, f"{dist}", (x0 + 5, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
        # Анализ состояния РПЭ
    rpe_condition = analyze_rpe(mask, fovea_mask, smooth_upper, spline)
    defects_condition = detect_rpe_defects(mask, fovea_mask, smooth_upper, spline)

    # Выводим состояние РПЭ на изображении
    cv2.putText(vis, f"RPE Condition: {rpe_condition}",
                (int(vis.shape[1]/4), int(vis.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(vis, f"RPE Defects: {defects_condition}",
                (int(vis.shape[1]/4), int(vis.shape[0]/2)+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    

    for target_class, color in zip([2, 16, 10, 11], [(0, 255, 255), (0, 0, 255), (255, 0, 0), (255, 255, 0)]):
        width, height, left_x, right_x, max_point = detect_and_measure_detachments(mask, smooth_upper, spline, target_class)
        print(width, height, left_x, right_x, max_point)

        if width is not None and height is not None:
            print(f"Отслойка класса {target_class}: ширина = {width}, высота = {height}")
            
            # Отображаем прямоугольник, ограничивающий отслойку
            cv2.rectangle(vis, (left_x, max_point[1] - height), (right_x, max_point[1]), color, 2)

            # Добавляем текст для отображения ширины и высоты
            cv2.putText(vis, f"W: {width}, H: {height}", 
                        (left_x, max_point[1] - height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite(out_path, vis)


# ============================================================
#     ОСНОВНОЙ ЦИКЛ
# ============================================================
def process_all():
    files = [f for f in os.listdir(masks_dir) if f.endswith(".png")]

    for fname in files:
        mask_path = os.path.join(masks_dir, fname)
        img_path = os.path.join(images_dir, fname)
        fovea_path = os.path.join(fovea_dir, fname)

        if not os.path.exists(img_path) or not os.path.exists(fovea_path):
            print(f"[skip] {fname}")
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        fovea_mask = cv2.imread(fovea_path, cv2.IMREAD_UNCHANGED)

        xs, upper = extract_upper_boundary(mask)
        smooth_upper, spline = smooth_boundary(xs, upper, smooth_factor=30000)

        out_path = os.path.join(out_dir, fname.replace(".png", "_foveola_thickness.png"))
        visualize(img, mask, fovea_mask, smooth_upper, spline, out_path)

        print("→", out_path)


# ============================================================
#     L-ОБЪЕКТ ДЛЯ МАСШТАБИРОВАНИЯ
# ============================================================
def find_L_in_image(image: np.ndarray) -> np.ndarray | None:
    """
    Находит белый L-образный уголок на темном фоне.
    Сначала ищет самый большой контур, потом левый нижний угол внутри него.
    Возвращает контур L или None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    main_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(main_contour)

    corner_w, corner_h = max(1, int(0.07 * w)), max(1, int(0.2 * h))
    corner_x = x
    corner_y = y + h - corner_h

    corner = gray[corner_y:corner_y+corner_h, corner_x:corner_x+corner_w]

    _, corner_bin = cv2.threshold(corner, 100, 255, cv2.THRESH_BINARY)

    corner_contours, _ = cv2.findContours(corner_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not corner_contours:
        return None

    L_contour = max(corner_contours, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3])

    L_contour += np.array([[[corner_x, corner_y]]], dtype=np.int32)

    return L_contour


def calculate_scale_from_L(L_contour: np.ndarray, L_size_micrometers: float = 200.0) -> float | None:
    """
    Вычисляет масштаб (мкм/пиксель) на основе размера L-контура.
    
    Args:
        L_contour: контур L-объекта
        L_size_micrometers: реальный размер L в микрометрах (по умолчанию 200 мкм)
    
    Returns:
        Масштаб в мкм/пиксель или None если контур невалиден
    """
    if L_contour is None or len(L_contour) < 3:
        return None
    
    rect = cv2.minAreaRect(L_contour)
    if rect is None:
        return None
    
    w, h = rect[1]
    if w <= 0 or h <= 0:
        return None
    
    L_pixel_size = max(w, h)
    scale = L_size_micrometers / L_pixel_size
    
    return scale


def calculate_scales_from_L(L_contour: np.ndarray, L_size_micrometers: float = 200.0) -> tuple | None:
    """
    Вычисляет масштабы (мкм/пиксель) отдельно для X и Y на основе L-объекта.
    L-объект имеет размеры L_size_micrometers x L_size_micrometers мкм.
    
    Args:
        L_contour: контур L-объекта
        L_size_micrometers: размер L по обеим осям в микрометрах (по умолчанию 200 мкм)
    
    Returns:
        Кортеж (scale_x, scale_y, w_px, h_px) где масштабы в мкм/пиксель, w_px и h_px в пиксельях
        или None если контур невалиден
    """
    if L_contour is None or len(L_contour) < 3:
        return None
    
    rect = cv2.minAreaRect(L_contour)
    if rect is None:
        return None
    
    w, h = rect[1]  # ширина (X) и высота (Y) ограничивающего прямоугольника
    if w <= 0 or h <= 0:
        return None
    
    # Масштабы для каждой оси
    scale_x = L_size_micrometers / w
    scale_y = L_size_micrometers / h
    
    return scale_x, scale_y, w, h


# ============================================================
#     ВИЗУАЛИЗАЦИЯ ИЗМЕРЕНИЙ
# ============================================================
def create_measurements_visualization(
    image_shape: tuple,
    mask: np.ndarray,
    fovea_mask: np.ndarray,
    smooth_upper: np.ndarray,
    spline,
    image: np.ndarray = None,
) -> np.ndarray:
    """
    Создает слой с визуализацией всех измерений.
    Возвращает RGB изображение с нарисованными линиями измерений.
    """
    h, w = image_shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Визуализация L-объекта если есть изображение
    if image is not None:
        L_contour = find_L_in_image(image)
        if L_contour is not None:
            # Рисуем контур L-объекта
            cv2.drawContours(vis, [L_contour], -1, (255, 255, 0), 2)
            
            # Получаем bounding box для отображения размеров
            rect = cv2.minAreaRect(L_contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            # Рисуем ограничивающий прямоугольник
            cv2.drawContours(vis, [box], 0, (0, 255, 255), 1)
            
            # Вычисляем масштабы
            result = calculate_scales_from_L(L_contour)
            if result is not None:
                scale_x, scale_y, w_px, h_px = result
                
                # Центр L-объекта для текста
                M = cv2.moments(L_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Отображаем размеры
                    cv2.putText(vis, f"W: {w_px:.1f}px = 200um", 
                                (cx - 80, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    cv2.putText(vis, f"H: {h_px:.1f}px = 200um", 
                                (cx - 80, cy + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Сглаженная линия верхней границы хориоидеи
    for x in range(len(smooth_upper) - 1):
        cv2.line(vis,
                 (x, int(smooth_upper[x])),
                 (x + 1, int(smooth_upper[x + 1])),
                 (0, 255, 0), 2, cv2.LINE_AA)
    
    # Центр фовеолы
    center = get_foveola_center(fovea_mask)
    if center is not None:
        cx, cy = center
        cv2.circle(vis, (cx, cy), 4, (255, 0, 0), -1)
        
        # Перпендикуляр через фовеолу (фиолетовая линия)
        result = perpendicular_through_foveola(smooth_upper, spline, center, mask)
        if result is not None:
            x0, y0, dist, pt = result
            cv2.line(vis, (x0, y0), pt, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"{dist}px", (x0 + 5, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Центральная толщина в фовеоле (синяя линия)
        width = central_width_of_retina(smooth_upper, spline, center, mask)
        if width is not None:
            x0, y0, dist, pt = width
            cv2.line(vis, (x0, y0), pt, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, f"{dist}px", (x0 + 5, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Толщина рядом с фовеолой и фовеей
    for ind, color, label in [(1, (255, 255, 0), "F"), (2, (0, 165, 255), "FA")]:
        pts = central_width_near(
            np.zeros((h, w, 3), dtype=np.uint8),  # dummy image
            smooth_upper, spline, fovea_mask, mask, ind
        )
        if pts is not None:
            left_pt, right_pt = pts
            
            # Левая точка
            cv2.circle(vis, left_pt, 5, color, -1)
            wl = central_width_of_retina(smooth_upper, spline, left_pt, mask)
            if wl is not None:
                x0, y0, dist, pt = wl
                cv2.line(vis, (x0, y0), pt, color, 1, cv2.LINE_AA)
                cv2.putText(vis, f"{dist}px", (x0 + 5, y0 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Правая точка
            cv2.circle(vis, right_pt, 5, color, -1)
            wr = central_width_of_retina(smooth_upper, spline, right_pt, mask)
            if wr is not None:
                x0, y0, dist, pt = wr
                cv2.line(vis, (x0, y0), pt, color, 1, cv2.LINE_AA)
                cv2.putText(vis, f"{dist}px", (x0 + 5, y0 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Отслойки
    DETACHMENTS = {
        2: (0, 255, 255),   # СОПЭ - жёлтый
        16: (0, 0, 255),    # ГОПЭ - красный
        10: (255, 0, 0),    # ФВОПЭ - синий
        11: (255, 255, 0),  # Друзеноидная - голубой
    }
    
    for class_id, color in DETACHMENTS.items():
        width, height, left_x, right_x, max_point = detect_and_measure_detachments(
            mask, smooth_upper, spline, class_id
        )
        if width is not None and height is not None and max_point is not None:
            cv2.rectangle(vis, (left_x, max_point[1] - height), (right_x, max_point[1]), color, 2)
            cv2.putText(vis, f"W:{width} H:{height}", 
                        (left_x, max_point[1] - height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return vis


# ============================================================
#     ТОЛЩИНА ХОРИОИДЕИ В ЦЕНТРЕ (С ПИФАГОРОМ)
# ============================================================
def measure_choroid_thickness_at_center(
    mask: np.ndarray,
    smooth_upper: np.ndarray,
    spline,
    center_x: int,
    scale_x: float = None,
    scale_y: float = None,
    foveola_center: tuple = None,
) -> tuple | None:
    """
    Измеряет толщину хориоидеи в центре.
    Продлевает перпендикуляр ЦТС от центра фовеолы (cy) вниз и считает,
    какая часть этого перпендикуляра пересекается с слоем хориоидеи (CHOROID_ID).
    
    Args:
        mask: маска сегментации
        smooth_upper: сглаженная верхняя граница хориоидеи
        spline: сплайн верхней границы
        center_x: X координата центра (обычно центр фовеолы)
        scale_x: масштаб по X (мкм/пиксель)
        scale_y: масштаб по Y (мкм/пиксель)
        foveola_center: кортеж (x, y) центра фовеолы
    
    Returns:
        Кортеж (верхняя_точка, нижняя_точка, dx_мкм, dy_мкм) или None если масштаб не задан
    """
    if center_x < 0 or center_x >= len(smooth_upper):
        return None
    
    if foveola_center is None:
        return None
    
    cx, cy = foveola_center
    h, w = mask.shape
    
    # Верхняя граница хориоидеи
    y0 = float(smooth_upper[center_x])
    
    slope = float(spline.derivative()(center_x))
    
    # Перпендикуляр к верхней границе (тот же, что для ЦТС)
    nx = -slope
    ny = 1.0
    L = np.sqrt(nx*nx + ny*ny)
    nx /= L
    ny /= L
    
    # Продолжаем перпендикуляр от центра фовеолы (cy)
    # Ищем начало и конец пересечения с CHOROID_ID
    t_start_choroid = None
    t_end_choroid = None
    
    for t in range(0, 400):
        xx = int(cx + nx * t)
        yy = int(cy + ny * t)
        
        if xx < 0 or xx >= w or yy < 0 or yy >= h:
            break
        
        # Когда входим в хориоидею
        if mask[yy, xx] == CHOROID_ID and t_start_choroid is None:
            t_start_choroid = t
        
        # Когда выходим из хориоидеи
        if t_start_choroid is not None and mask[yy, xx] != CHOROID_ID:
            t_end_choroid = t - 1
            break
    
    # Если не вышли из хориоидеи - значит она идёт до конца
    if t_start_choroid is not None and t_end_choroid is None:
        t_end_choroid = 399
    
    if t_start_choroid is None:
        # Не пересекает хориоидею
        return None
    
    # Координаты точек пересечения
    x_start = int(cx + nx * t_start_choroid)
    y_start = int(cy + ny * t_start_choroid)
    x_end = int(cx + nx * t_end_choroid)
    y_end = int(cy + ny * t_end_choroid)
    
    upper_point = (x_start, y_start)
    lower_point = (x_end, y_end)
    
    # Расстояние между точками раздельно по X и Y
    dx_px = abs(lower_point[0] - upper_point[0])
    dy_px = abs(lower_point[1] - upper_point[1])
    
    # Преобразуем в микрометры с учетом масштабов по осям
    if scale_x and scale_y:
        dx_um = dx_px * scale_x
        dy_um = dy_px * scale_y
        # Общее расстояние по теореме Пифагора в мкм
        dist_um = np.sqrt(dx_um**2 + dy_um**2)
    else:
        # Если нет масштаба, возвращаем None
        return None
    
    return upper_point, lower_point, dx_um, dy_um




