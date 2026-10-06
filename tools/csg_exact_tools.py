# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact CSG — надёжные boolean-операции для Blender (Exact solver).

Что делает аддон
----------------
Union / Slice / Difference / Intersection на базе Exact-солвера.

Принципы надёжности
-------------------
* Вся работа с парой объектов идёт в ЛОКАЛЬНОМ пространстве целевого
  объекта: его вершины не трогаются (нет лишнего float-шума), операнд
  переводится в это пространство в double-точности.
* Отрицательный детерминант (зеркальный масштаб) -> корректный разворот граней.
  Нормали «вслепую» не пересчитываются: разворачивается только целиком
  вывернутый замкнутый меш.
* Hole Tolerant включается автоматически только если на входе есть
  открытая геометрия (иначе это лишняя потеря скорости).
* Объекты, чьи bbox не пересекаются, в boolean не участвуют вообще
  (остаются нетронутыми).
* Сцена меняется только после успешного завершения всех вычислений;
  пустой результат не приводит к потере исходников.

Триангуляция
------------
* Включена по умолчанию, отключается галочкой «No Triangulation»
  в панели «Adjust Last Operation» (слева снизу).
* Триангулируются только ЗАТРОНУТЫЕ грани (которых не было на входе),
  и только если это нужно (ngon > 4, неплоские, вогнутые квады).
  Нетронутые грани не меняются.
* Триангуляция — ограниченная Делоне (constrained Delaunay) внутри
  полигона: максимизируется минимальный угол, т.е. минимум «растянутых»
  треугольников. UV / атрибуты петель сохраняются.
"""

bl_info = {
    "name": "Quick Exact CSG",
    "author": "Anthropic Claude",
    "version": (1, 1, 3),
    "blender": (5, 2, 0),
    "location": "Object Mode > Object > CSG",
    "description": (
        "Точные CSG-операции (Exact): Union, Slice, Difference, Intersection. "
        "Опциональная аккуратная триангуляция только затронутых граней."
    ),
    "category": "Object",
    "doc_url": "",
    "tracker_url": "",
}

import math
import traceback
from typing import Dict, List, Optional, Sequence, Set, Tuple

import bpy
import bmesh
import numpy as np
from bpy.props import BoolProperty, EnumProperty
from mathutils import Vector
from mathutils.geometry import tessellate_polygon


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

SOLVER = "EXACT"
TEMP_PREFIX = ".csg_tmp_"
PLANAR_REL = 1e-5      # допуск неплоскостности относительно размера грани
BBOX_REL_MARGIN = 1e-6  # запас при проверке пересечения bbox


class _Opts:
    """Параметры операции (из redo-панели)."""

    __slots__ = ("triangulate", "use_self", "hole")

    def __init__(self, triangulate: bool = True, use_self: bool = True,
                 hole: str = "AUTO"):
        self.triangulate = triangulate
        self.use_self = use_self
        self.hole = hole


# ---------------------------------------------------------------------------
# Чистая геометрия: анализ полигона и триангуляция Делоне (без bpy-типов)
# ---------------------------------------------------------------------------

def _newell(pts: Sequence[Sequence[float]]) -> Tuple[float, float, float]:
    nx = ny = nz = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1, z1 = pts[i]
        x2, y2, z2 = pts[(i + 1) % n]
        nx += (y1 - y2) * (z1 + z2)
        ny += (z1 - z2) * (x1 + x2)
        nz += (x1 - x2) * (y1 + y2)
    return nx, ny, nz


def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _analyze_polygon(pts):
    """
    Проецирует полигон в плоскость наилучшего приближения (правый базис,
    обход против часовой). Возвращает (P2d, nonplanar, concave) либо None
    для вырожденного полигона. P2d нормирован (макс. размер ~1).
    """
    n = len(pts)
    nx, ny, nz = _newell(pts)
    ln = math.sqrt(nx * nx + ny * ny + nz * nz)
    if ln < 1e-30:
        return None
    nx, ny, nz = nx / ln, ny / ln, nz / ln

    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    cz = sum(p[2] for p in pts) / n

    # базис (u, v, n) — правый
    if abs(nx) < 0.9:
        hx, hy, hz = 1.0, 0.0, 0.0
    else:
        hx, hy, hz = 0.0, 1.0, 0.0
    d = hx * nx + hy * ny + hz * nz
    ux, uy, uz = hx - nx * d, hy - ny * d, hz - nz * d
    ul = math.sqrt(ux * ux + uy * uy + uz * uz)
    ux, uy, uz = ux / ul, uy / ul, uz / ul
    vx, vy, vz = ny * uz - nz * uy, nz * ux - nx * uz, nx * uy - ny * ux

    P = []
    dev = 0.0
    for (x, y, z) in pts:
        rx, ry, rz = x - cx, y - cy, z - cz
        P.append((rx * ux + ry * uy + rz * uz, rx * vx + ry * vy + rz * vz))
        dev = max(dev, abs(rx * nx + ry * ny + rz * nz))

    size = max(max(abs(p[0]), abs(p[1])) for p in P)
    if size < 1e-30:
        return None

    nonplanar = dev > PLANAR_REL * size
    P = [(p[0] / size, p[1] / size) for p in P]

    concave = False
    for i in range(n):
        if _orient(P[i - 1], P[i], P[(i + 1) % n]) < -1e-9:
            concave = True
            break
    return P, nonplanar, concave


def _in_tri(p, a, b, c) -> bool:
    return (_orient(a, b, p) >= 0.0 and _orient(b, c, p) >= 0.0
            and _orient(c, a, p) >= 0.0)


def _ear_clip(P) -> List[List[int]]:
    """Запасной ear-clipping для CCW-полигона."""
    n = len(P)
    idx = list(range(n))
    tris: List[List[int]] = []
    guard = 0
    while len(idx) > 3 and guard < n * n + 10:
        guard += 1
        m = len(idx)
        clipped = False
        for k in range(m):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            if _orient(P[i0], P[i1], P[i2]) <= 1e-14:
                continue
            if any(_in_tri(P[j], P[i0], P[i1], P[i2])
                   for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append([i0, i1, i2])
            idx.pop(k)
            clipped = True
            break
        if not clipped:
            # вырожденный случай: срезаем самую «плоскую» вершину
            k = min(range(m), key=lambda q: abs(
                _orient(P[idx[q - 1]], P[idx[q]], P[idx[(q + 1) % m]])))
            tris.append([idx[k - 1], idx[k], idx[(k + 1) % m]])
            idx.pop(k)
    tris.append(list(idx))
    return tris


def _tessellate(P) -> List[List[int]]:
    n = len(P)
    try:
        res = tessellate_polygon([[Vector((x, y, 0.0)) for x, y in P]])
        if len(res) == n - 2:
            return [list(t) for t in res]
    except Exception:
        pass
    return _ear_clip(P)


def _should_flip(a, b, c, d) -> bool:
    """
    Треугольники (a,b,c) и (b,a,d) делят ребро ab. True, если замена
    диагонали на cd допустима (выпуклый четырёхугольник) и улучшает
    триангуляцию по критерию Делоне (max-min угол).
    """
    eps = 1e-12
    if _orient(a, d, c) <= eps or _orient(d, b, c) <= eps:
        return False  # не выпуклый — флип невозможен
    if _orient(a, b, c) <= eps or _orient(b, a, d) <= eps:
        return True   # вырожденный треугольник — флип его убирает
    adx, ady = a[0] - d[0], a[1] - d[1]
    bdx, bdy = b[0] - d[0], b[1] - d[1]
    cdx, cdy = c[0] - d[0], c[1] - d[1]
    det = ((adx * adx + ady * ady) * (bdx * cdy - cdx * bdy)
           + (bdx * bdx + bdy * bdy) * (cdx * ady - adx * cdy)
           + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady))
    return det > 1e-10


def _lawson_flips(tris: List[List[int]], P) -> None:
    """Флипы Лоусона: ограниченная (по границе полигона) триангуляция Делоне."""
    dmap: Dict[Tuple[int, int], int] = {}
    for ti, t in enumerate(tris):
        for e in range(3):
            dmap[(t[e], t[(e + 1) % 3])] = ti

    stack = [(a, b) for (a, b) in dmap if a < b and (b, a) in dmap]
    limit = 30 * len(tris) + 100
    while stack and limit > 0:
        limit -= 1
        a, b = stack.pop()
        t1 = dmap.get((a, b))
        t2 = dmap.get((b, a))
        if t1 is None or t2 is None:
            continue  # граничное ребро полигона — не трогаем
        T1, T2 = tris[t1], tris[t2]
        c = next(v for v in T1 if v != a and v != b)
        d = next(v for v in T2 if v != a and v != b)
        if not _should_flip(P[a], P[b], P[c], P[d]):
            continue
        for t in (T1, T2):
            for e in range(3):
                dmap.pop((t[e], t[(e + 1) % 3]), None)
        tris[t1] = [a, d, c]
        tris[t2] = [d, b, c]
        for ti in (t1, t2):
            t = tris[ti]
            for e in range(3):
                dmap[(t[e], t[(e + 1) % 3])] = ti
        for p, q in ((a, d), (d, b), (b, c), (c, a)):
            stack.append((p, q) if p < q else (q, p))


def _best_triangulation(P) -> Optional[List[List[int]]]:
    """Список CCW-треугольников (индексы) с минимумом вытянутых."""
    n = len(P)
    if n < 3:
        return None
    if n == 3:
        return [[0, 1, 2]]
    tris = _tessellate(P)
    for t in tris:
        if _orient(P[t[0]], P[t[1]], P[t[2]]) < 0.0:
            t[1], t[2] = t[2], t[1]
    if len(tris) != n - 2:
        return None
    _lawson_flips(tris, P)
    return tris


def _face_needs_triangulation(n: int, nonplanar: bool, concave: bool) -> bool:
    """Для результата: ngon > 4, неплоские и вогнутые квады."""
    return n > 4 or (n == 4 and (nonplanar or concave))


# ---------------------------------------------------------------------------
# bmesh: триангуляция выбранных граней, ключи граней
# ---------------------------------------------------------------------------

def _split_face(bm: bmesh.types.BMesh, face: bmesh.types.BMFace) -> bool:
    """Заменяет грань оптимальной триангуляцией. Сохраняет атрибуты/UV."""
    verts = list(face.verts)
    n = len(verts)
    if n < 4:
        return False
    if len(set(verts)) != n:
        return False  # самокасающийся полигон — оставляем как есть

    info = _analyze_polygon([tuple(v.co) for v in verts])
    if info is None:
        return False
    tris = _best_triangulation(info[0])
    if tris is None:
        # Запасной вариант — штатная триангуляция bmesh
        bmesh.ops.triangulate(bm, faces=[face],
                              quad_method="BEAUTY", ngon_method="BEAUTY")
        return True

    loops = {l.vert: l for l in face.loops}
    created = []
    try:
        for (i, j, k) in tris:
            f = bm.faces.new((verts[i], verts[j], verts[k]), face)
            for l in f.loops:
                l.copy_from(loops[l.vert])
            created.append(f)
    except ValueError:
        for f in created:
            bm.faces.remove(f)
        return False
    bm.faces.remove(face)
    return True


def _triangulate_nonplanar(bm: bmesh.types.BMesh) -> int:
    """Вход: триангулируем только неплоские грани (они ломают Exact)."""
    count = 0
    for f in list(bm.faces):
        if len(f.verts) < 4:
            continue
        info = _analyze_polygon([tuple(v.co) for v in f.verts])
        if info is not None and info[1]:
            if _split_face(bm, f):
                count += 1
    return count


def _face_key(face: bmesh.types.BMFace):
    return tuple(sorted(tuple(v.co) for v in face.verts))


def _face_keys(me: bpy.types.Mesh) -> Set[tuple]:
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        return {_face_key(f) for f in bm.faces}
    finally:
        bm.free()


def _finalize_triangulation(me: bpy.types.Mesh, input_keys: Set[tuple]) -> int:
    """
    Триангулирует только затронутые грани результата (нет среди входных)
    и только если это нужно. Нетронутые грани не меняются.
    """
    bm = bmesh.new()
    count = 0
    try:
        bm.from_mesh(me)
        todo = [f for f in bm.faces
                if len(f.verts) > 3 and _face_key(f) not in input_keys]
        for f in todo:
            info = _analyze_polygon([tuple(v.co) for v in f.verts])
            if info is None:
                continue
            _, nonplanar, concave = info
            if not _face_needs_triangulation(len(f.verts), nonplanar, concave):
                continue
            if _split_face(bm, f):
                count += 1
        if count:
            bm.to_mesh(me)
            me.update()
    finally:
        bm.free()
    return count


# ---------------------------------------------------------------------------
# Временные данные
# ---------------------------------------------------------------------------

class _Temp:
    """Учёт временных мешей; всё лишнее удаляется при выходе."""

    def __init__(self):
        self._meshes: List[bpy.types.Mesh] = []

    def add(self, me: bpy.types.Mesh) -> bpy.types.Mesh:
        self._meshes.append(me)
        return me

    def keep(self, me: bpy.types.Mesh) -> bpy.types.Mesh:
        self._meshes = [m for m in self._meshes if m != me]
        return me

    def free(self, me: bpy.types.Mesh) -> None:
        self.keep(me)
        try:
            bpy.data.meshes.remove(me)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for me in self._meshes:
            try:
                bpy.data.meshes.remove(me)
            except Exception:
                pass
        self._meshes = []
        return False


class _Part:
    """Подготовленный меш + флаги."""

    __slots__ = ("mesh", "is_open", "base")

    def __init__(self, mesh: bpy.types.Mesh, is_open: bool, base: bool = False):
        self.mesh = mesh
        self.is_open = is_open
        self.base = base  # True — не промежуточный результат


# ---------------------------------------------------------------------------
# Утилиты объектов / мешей
# ---------------------------------------------------------------------------

def _is_mesh_object(obj) -> bool:
    return obj is not None and obj.type == "MESH"


def _selected_mesh_objects(context) -> List[bpy.types.Object]:
    return [o for o in context.selected_objects if _is_mesh_object(o)]


def _ordered_selection(context) -> List[bpy.types.Object]:
    """Выделенные меши, активный — первым."""
    objs = _selected_mesh_objects(context)
    act = context.active_object
    if act is not None and act in objs:
        objs.remove(act)
        objs.insert(0, act)
    return objs


def _eval_mesh(context, obj) -> bpy.types.Mesh:
    """Независимая копия evaluated-меша (с модификаторами), локальное пространство."""
    dg = context.evaluated_depsgraph_get()
    return bpy.data.meshes.new_from_object(
        obj.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)


def _matrix(obj) -> np.ndarray:
    """Мировая матрица объекта (double) из актуального evaluated-состояния."""
    try:
        dg = bpy.context.evaluated_depsgraph_get()
        return np.array(obj.evaluated_get(dg).matrix_world, dtype=np.float64)
    except Exception:
        return np.array(obj.matrix_world, dtype=np.float64)


def _rel_matrix(m_target: np.ndarray, m_other: np.ndarray) -> Optional[np.ndarray]:
    """Матрица перевода локального пространства other -> локальное target (double)."""
    if np.array_equal(m_target, m_other):
        return None
    if abs(np.linalg.det(m_target[:3, :3])) < 1e-30:
        raise ValueError("Объект с нулевым масштабом — CSG невозможен")
    rel = np.linalg.inv(m_target) @ m_other
    if np.allclose(rel, np.eye(4), rtol=0.0, atol=1e-15):
        return None
    return rel


def _transform_mesh(me: bpy.types.Mesh, rel: np.ndarray) -> None:
    n = len(me.vertices)
    if n == 0:
        return
    co = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    c = co.reshape(-1, 3).astype(np.float64)
    c = c @ rel[:3, :3].T + rel[:3, 3]
    me.vertices.foreach_set("co", c.astype(np.float32).ravel())
    me.update()


def _bbox(me: bpy.types.Mesh):
    n = len(me.vertices)
    if n == 0:
        return None
    co = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    return co.min(axis=0).astype(np.float64), co.max(axis=0).astype(np.float64)


def _bbox_overlap(a: bpy.types.Mesh, b: bpy.types.Mesh) -> bool:
    ba, bb = _bbox(a), _bbox(b)
    if ba is None or bb is None:
        return False
    diag = max(np.linalg.norm(ba[1] - ba[0]), np.linalg.norm(bb[1] - bb[0]), 1e-12)
    m = BBOX_REL_MARGIN * diag
    return bool(np.all(ba[0] <= bb[1] + m) and np.all(bb[0] <= ba[1] + m))


def _is_empty(me: Optional[bpy.types.Mesh]) -> bool:
    return me is None or len(me.polygons) == 0 or len(me.vertices) == 0


def _prepare(me: bpy.types.Mesh, rel: Optional[np.ndarray], opts: _Opts) -> _Part:
    """
    Подготовка меша НА МЕСТЕ (me — собственная копия):
    перевод в пространство цели, коррекция ориентации, триангуляция
    только неплоских граней (если включена триангуляция).
    """
    flip = False
    if rel is not None:
        _transform_mesh(me, rel)
        flip = np.linalg.det(rel[:3, :3]) < 0.0

    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        modified = False

        if flip and bm.faces:
            bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
            modified = True

        closed = bool(bm.faces) and all(
            e.is_manifold and e.is_contiguous for e in bm.edges)

        # Целиком вывернутый замкнутый меш (отрицательный объём) — разворачиваем.
        # Полости/вложенные оболочки при этом не портятся (глобальный знак).
        if closed and bm.calc_volume(signed=True) < 0.0:
            bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
            modified = True

        if opts.triangulate and _triangulate_nonplanar(bm):
            modified = True

        if modified:
            bm.to_mesh(me)
            me.update()
    finally:
        bm.free()
    return _Part(me, not closed, base=True)


def _exact_boolean(context, a: bpy.types.Mesh, b: bpy.types.Mesh,
                   operation: str, opts: _Opts, open_inputs: bool
                   ) -> bpy.types.Mesh:
    """
    Exact boolean двух мешей, лежащих в ОДНОМ пространстве.
    Возвращает новый меш. Входные меши не изменяются.
    Оба временных объекта имеют identity-трансформ, поэтому
    относительная матрица модификатора точно единичная.
    """
    col = context.scene.collection
    obj_a = bpy.data.objects.new(TEMP_PREFIX + "a", a)
    obj_b = bpy.data.objects.new(TEMP_PREFIX + "b", b)
    col.objects.link(obj_a)
    col.objects.link(obj_b)
    try:
        mod = obj_a.modifiers.new(name="CSG_Exact", type="BOOLEAN")
        mod.operation = operation
        mod.solver = SOLVER
        mod.operand_type = "OBJECT"
        mod.object = obj_b
        try:
            mod.material_mode = "TRANSFER"
        except Exception:
            pass
        if hasattr(mod, "use_self"):
            mod.use_self = bool(opts.use_self)
        if hasattr(mod, "use_hole_tolerant"):
            mod.use_hole_tolerant = (
                opts.hole == "ON" or (opts.hole == "AUTO" and open_inputs))

        context.view_layer.update()
        dg = context.evaluated_depsgraph_get()
        return bpy.data.meshes.new_from_object(
            obj_a.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    finally:
        bpy.data.objects.remove(obj_a, do_unlink=True)
        bpy.data.objects.remove(obj_b, do_unlink=True)


# ---------------------------------------------------------------------------
# CSG алгоритмы. Возвращают (items, consumed, message)
#   items    — [(mesh, ref_obj, name)]: новые объекты
#   consumed — исходные объекты, которые заменяются
# ---------------------------------------------------------------------------

Result = Tuple[List[Tuple[bpy.types.Mesh, bpy.types.Object, str]],
               List[bpy.types.Object], str]


def csg_union(context, objs: Sequence[bpy.types.Object], opts: _Opts,
              tmp: _Temp) -> Result:
    """Union всех объектов; пространство — локальное первого (активного)."""
    ref = objs[0]
    m_ref = _matrix(ref)

    parts: List[_Part] = []
    for i, o in enumerate(objs):
        me = tmp.add(_eval_mesh(context, o))
        rel = None if i == 0 else _rel_matrix(m_ref, _matrix(o))
        parts.append(_prepare(me, rel, opts))

    keys: Set[tuple] = set()
    if opts.triangulate:
        for p in parts:
            keys |= _face_keys(p.mesh)

    acc = parts[0]
    for p in parts[1:]:
        res = tmp.add(_exact_boolean(
            context, acc.mesh, p.mesh, "UNION", opts, acc.is_open or p.is_open))
        if not acc.base:
            tmp.free(acc.mesh)
        acc = _Part(res, acc.is_open or p.is_open, base=False)

    if _is_empty(acc.mesh):
        return [], [], "Union дал пустой результат"
    if opts.triangulate:
        _finalize_triangulation(acc.mesh, keys)
    tmp.keep(acc.mesh)
    return [(acc.mesh, ref, ref.name + "_Union")], list(objs), \
        f"Union: {len(objs)} → 1"


def csg_intersection(context, objs: Sequence[bpy.types.Object], opts: _Opts,
                     tmp: _Temp) -> Result:
    """Пересечение всех объектов; пространство — локальное первого."""
    ref = objs[0]
    m_ref = _matrix(ref)

    parts: List[_Part] = []
    for i, o in enumerate(objs):
        me = tmp.add(_eval_mesh(context, o))
        rel = None if i == 0 else _rel_matrix(m_ref, _matrix(o))
        parts.append(_prepare(me, rel, opts))

    # Быстрый выход: любая пара с непересекающимися bbox -> пусто
    for p in parts[1:]:
        if not _bbox_overlap(parts[0].mesh, p.mesh):
            return [], [], "Intersection пустое — объекты не пересекаются"

    keys: Set[tuple] = set()
    if opts.triangulate:
        for p in parts:
            keys |= _face_keys(p.mesh)

    acc = parts[0]
    for p in parts[1:]:
        res = tmp.add(_exact_boolean(
            context, acc.mesh, p.mesh, "INTERSECT", opts,
            acc.is_open or p.is_open))
        if not acc.base:
            tmp.free(acc.mesh)
        acc = _Part(res, acc.is_open or p.is_open, base=False)
        if _is_empty(acc.mesh):
            return [], [], "Intersection пустое — объекты не пересекаются"

    if opts.triangulate:
        _finalize_triangulation(acc.mesh, keys)
    tmp.keep(acc.mesh)
    return [(acc.mesh, ref, ref.name + "_Intersect")], list(objs), \
        f"Intersection: {len(objs)} → 1"


def csg_difference(context, targets: Sequence[bpy.types.Object],
                   cutter: bpy.types.Object, opts: _Opts, tmp: _Temp) -> Result:
    """
    Активный объект (cutter) вычитается из каждого target.
    Каждый результат живёт в локальном пространстве своего target.
    Не пересекающиеся с резаком объекты остаются нетронутыми.
    """
    m_cut = _matrix(cutter)
    cutter_base = tmp.add(_eval_mesh(context, cutter))

    items = []
    consumed: List[bpy.types.Object] = []
    skipped = 0
    emptied = 0

    for t in targets:
        m_t = _matrix(t)
        t_part = _prepare(tmp.add(_eval_mesh(context, t)), None, opts)
        c_part = _prepare(tmp.add(cutter_base.copy()),
                          _rel_matrix(m_t, m_cut), opts)

        if not _bbox_overlap(t_part.mesh, c_part.mesh):
            skipped += 1
            continue

        res = tmp.add(_exact_boolean(
            context, t_part.mesh, c_part.mesh, "DIFFERENCE", opts,
            t_part.is_open or c_part.is_open))
        consumed.append(t)
        if _is_empty(res):
            emptied += 1
            tmp.free(res)
            continue
        if opts.triangulate:
            _finalize_triangulation(
                res, _face_keys(t_part.mesh) | _face_keys(c_part.mesh))
        tmp.keep(res)
        items.append((res, t, t.name + "_Diff"))

    if not consumed:
        return [], [], "Резак не пересекается ни с одним объектом"
    if not items:
        return [], [], "Результат пуст: объекты целиком внутри резака"

    consumed.append(cutter)
    msg = f"Difference: вырезано из {len(items)} объект(ов)"
    if skipped:
        msg += f", не затронуто: {skipped}"
    if emptied:
        msg += f", полностью удалено: {emptied}"
    return items, consumed, msg


def csg_slice(context, objs: Sequence[bpy.types.Object], opts: _Opts,
              tmp: _Temp) -> Result:
    """
    Каждый объект режется каждым из остальных (INTERSECT + DIFFERENCE).
    Общая часть пары достаётся объекту с меньшим индексом (активный — 0).
    Фрагменты объекта живут в его локальном пространстве. Ножи берутся
    всегда из исходных данных; непересекающиеся фрагменты не обрабатываются.
    """
    n = len(objs)
    mats = [_matrix(o) for o in objs]
    base = [tmp.add(_eval_mesh(context, o)) for o in objs]
    own = [_prepare(tmp.add(base[i].copy()), None, opts) for i in range(n)]

    cache: Dict[Tuple[int, int], _Part] = {}

    def knife(i: int, j: int) -> _Part:
        key = (i, j)
        if key not in cache:
            cache[key] = _prepare(tmp.add(base[j].copy()),
                                  _rel_matrix(mats[i], mats[j]), opts)
        return cache[key]

    frags: List[Tuple[_Part, int]] = [(own[i], i) for i in range(n)]
    touched: Set[int] = set()

    for k in range(n):
        nxt: List[Tuple[_Part, int]] = []
        for part, origin in frags:
            if origin == k:
                nxt.append((part, origin))
                continue
            kn = knife(origin, k)
            if not _bbox_overlap(part.mesh, kn.mesh):
                nxt.append((part, origin))
                continue

            touched.add(origin)
            open_flag = part.is_open or kn.is_open
            keep_intersect = origin < k

            pieces: List[bpy.types.Mesh] = []
            diff = tmp.add(_exact_boolean(
                context, part.mesh, kn.mesh, "DIFFERENCE", opts, open_flag))
            pieces.append(diff)
            if keep_intersect:
                pieces.append(tmp.add(_exact_boolean(
                    context, part.mesh, kn.mesh, "INTERSECT", opts, open_flag)))

            if not part.base:
                tmp.free(part.mesh)
            for me in pieces:
                if _is_empty(me):
                    tmp.free(me)
                else:
                    nxt.append((_Part(me, open_flag, base=False), origin))
        frags = nxt

    if not touched:
        return [], [], "Slice: объекты не пересекаются"

    # Ключи входных граней — до любых изменений мешей
    keys: Dict[int, Set[tuple]] = {}
    if opts.triangulate:
        for i in touched:
            ks = _face_keys(own[i].mesh)
            for j in range(n):
                if j != i and (i, j) in cache:
                    ks |= _face_keys(cache[(i, j)].mesh)
            keys[i] = ks

    items = []
    consumed: List[bpy.types.Object] = []
    counters = {i: 0 for i in range(n)}
    for part, origin in frags:
        if origin not in touched:
            continue
        if opts.triangulate:
            _finalize_triangulation(part.mesh, keys[origin])
        counters[origin] += 1
        name = f"{objs[origin].name}_Slice_{counters[origin]:02d}"
        tmp.keep(part.mesh)
        items.append((part.mesh, objs[origin], name))
    for i in sorted(touched):
        consumed.append(objs[i])

    return items, consumed, \
        f"Slice: {len(consumed)} объект(ов) → {len(items)} фрагмент(ов)"


# ---------------------------------------------------------------------------
# Применение результата к сцене
# ---------------------------------------------------------------------------

_TRANSFORM_PROPS = (
    "rotation_mode", "location", "rotation_euler", "rotation_quaternion",
    "rotation_axis_angle", "scale", "delta_location", "delta_rotation_euler",
    "delta_rotation_quaternion", "delta_scale",
)


def _make_object(context, mesh: bpy.types.Mesh, ref: bpy.types.Object,
                 name: str) -> bpy.types.Object:
    """Новый объект с точной копией трансформа/родителя/коллекций ref."""
    mesh.name = name
    obj = bpy.data.objects.new(name, mesh)
    cols = list(ref.users_collection) or [context.scene.collection]
    for c in cols:
        c.objects.link(obj)
    obj.parent = ref.parent
    obj.parent_type = ref.parent_type
    if ref.parent is not None:
        obj.matrix_parent_inverse = ref.matrix_parent_inverse.copy()
    for prop in _TRANSFORM_PROPS:
        setattr(obj, prop, getattr(ref, prop))
    return obj


def _remove_object(obj: bpy.types.Object) -> None:
    me = obj.data if obj.type == "MESH" else None
    bpy.data.objects.remove(obj, do_unlink=True)
    if me is not None and me.users == 0:
        bpy.data.meshes.remove(me)


def _deselect_all(context) -> None:
    """Снять выделение, не спотыкаясь о «битые» записи view layer.

    После bpy.data.objects.remove() (в т.ч. для временных объектов boolean)
    view_layer.objects может отдавать None, пока слои не пересинхронизированы.
    """
    vl = context.view_layer
    vl.update()
    for o in list(vl.objects):
        if o is None:
            continue
        try:
            o.select_set(False)
        except (ReferenceError, RuntimeError):
            pass


def _commit(context, items, consumed) -> List[bpy.types.Object]:
    vl = context.view_layer

    # 1) Снимаем выделение, пока все исходные объекты ещё валидны
    _deselect_all(context)

    # 2) Создаём новые объекты, затем удаляем заменённые
    new_objs = [_make_object(context, me, ref, name) for me, ref, name in items]
    for o in consumed:
        _remove_object(o)

    # 3) Синхронизируем view layer (иначе новые объекты ещё не имеют Base,
    #    а удалённые оставляют None в vl.objects)
    vl.update()

    # 4) Выделяем результат и делаем активным
    for o in new_objs:
        try:
            o.select_set(True)
        except (ReferenceError, RuntimeError):
            pass
    if new_objs:
        vl.objects.active = new_objs[0]
    return new_objs


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class CSG_OT_base(bpy.types.Operator):
    bl_options = {"REGISTER", "UNDO"}

    no_triangulation: BoolProperty(
        name="No Triangulation",
        description=(
            "Не триангулировать грани: результат остаётся таким, как его "
            "выдаёт Exact-солвер (n-gon'ы сохраняются). Выключено: "
            "затронутые грани-ngon/неплоские/вогнутые триангулируются "
            "(минимум вытянутых треугольников), нетронутые не меняются"
        ),
        default=False,
    )
    use_self: BoolProperty(
        name="Self Intersection",
        description="Учитывать самопересечения внутри операндов (надёжнее, медленнее)",
        default=True,
    )
    hole_tolerant: EnumProperty(
        name="Hole Tolerant",
        description="Режим для открытых мешей (медленнее)",
        items=(
            ("AUTO", "Auto", "Включать только если на входе есть открытая геометрия"),
            ("ON", "On", "Всегда включено"),
            ("OFF", "Off", "Всегда выключено"),
        ),
        default="AUTO",
    )

    @classmethod
    def poll(cls, context) -> bool:
        # ВАЖНО: poll вызывается Blender'ом и ПОСЛЕ выполнения оператора
        # (чтобы решить, показывать ли панель «Adjust Last Operation» и
        # можно ли делать redo). К этому моменту выделение уже заменено
        # результатом (часто — один объект), поэтому poll не должен требовать
        # «>= 2 объектов», иначе панель исчезает/redo перестаёт работать.
        # Проверка количества объектов — в _selection_ok() / execute().
        return context.mode == "OBJECT" and bool(_selected_mesh_objects(context))

    @staticmethod
    def selection_ok(context) -> bool:
        """Достаточно ли выделено объектов для операции (для меню и execute)."""
        return len(_selected_mesh_objects(context)) >= 2

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(self, "no_triangulation")
        layout.separator()
        layout.prop(self, "use_self")
        layout.prop(self, "hole_tolerant")

    def _opts(self) -> _Opts:
        return _Opts(triangulate=not self.no_triangulation,
                     use_self=self.use_self, hole=self.hole_tolerant)

    def run(self, context, opts: _Opts, tmp: _Temp) -> Result:
        raise NotImplementedError

    def execute(self, context):
        if not self.selection_ok(context):
            self.report({"WARNING"}, "Выделите минимум 2 mesh-объекта")
            return {"CANCELLED"}
        # После redo (undo + повторный execute) матрицы объектов могут быть
        # устаревшими/единичными, пока depsgraph не пересчитан. Тогда все
        # операнды считались бы лежащими в одном пространстве и «слипались»
        # в origin активного объекта. Принудительно синхронизируем сцену.
        context.view_layer.update()
        opts = self._opts()
        try:
            with _Temp() as tmp:
                items, consumed, msg = self.run(context, opts, tmp)
                if not items:
                    # сцена не меняется — ничего не потеряно
                    self.report({"WARNING"}, msg)
                    return {"CANCELLED"}
                _commit(context, items, consumed)
        except Exception as exc:
            traceback.print_exc()
            self.report({"ERROR"}, f"CSG failed: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class CSG_OT_union(CSG_OT_base):
    """Соединяет все выделенные mesh-объекты (Exact UNION)."""
    bl_idname = "csg.union"
    bl_label = "Union"
    bl_description = "Exact Union: объединяет все выделенные объекты в один"

    def run(self, context, opts, tmp):
        return csg_union(context, _ordered_selection(context), opts, tmp)


class CSG_OT_intersection(CSG_OT_base):
    """Оставляет общую часть всех выделенных объектов (Exact INTERSECT)."""
    bl_idname = "csg.intersection"
    bl_label = "Intersection"
    bl_description = "Exact Intersection: общая часть всех выделенных объектов"

    def run(self, context, opts, tmp):
        return csg_intersection(context, _ordered_selection(context), opts, tmp)


class CSG_OT_difference(CSG_OT_base):
    """Активный объект — резак; вырезается из всех остальных выделенных."""
    bl_idname = "csg.difference"
    bl_label = "Difference"
    bl_description = (
        "Exact Difference: активный объект — резак, "
        "из остальных выделенных всё вырезается"
    )

    @classmethod
    def poll(cls, context) -> bool:
        # см. комментарий в CSG_OT_base.poll
        return context.mode == "OBJECT" and _is_mesh_object(context.active_object)

    @staticmethod
    def selection_ok(context) -> bool:
        active = context.active_object
        if not _is_mesh_object(active):
            return False
        return any(o != active for o in _selected_mesh_objects(context))

    def run(self, context, opts, tmp):
        cutter = context.active_object
        targets = [o for o in _selected_mesh_objects(context) if o != cutter]
        return csg_difference(context, targets, cutter, opts, tmp)


class CSG_OT_slice(CSG_OT_base):
    """Режет все выделенные объекты друг другом, сохраняя все фрагменты."""
    bl_idname = "csg.slice"
    bl_label = "Slice"
    bl_description = (
        "Exact Slice: все выделенные объекты режут друг друга; "
        "сохраняются все непустые фрагменты"
    )

    def run(self, context, opts, tmp):
        return csg_slice(context, _ordered_selection(context), opts, tmp)


# ---------------------------------------------------------------------------
# UI: Object → CSG
# ---------------------------------------------------------------------------

class VIEW3D_MT_object_csg(bpy.types.Menu):
    bl_label = "CSG"
    bl_idname = "VIEW3D_MT_object_csg"

    def draw(self, context):
        layout = self.layout
        in_object_mode = context.mode == "OBJECT"

        def item(cls, text, icon):
            row = layout.row()
            row.enabled = in_object_mode and cls.selection_ok(context)
            row.operator(cls.bl_idname, text=text, icon=icon)

        item(CSG_OT_union, "Union", "SELECT_EXTEND")
        item(CSG_OT_slice, "Slice", "MOD_BOOLEAN")
        item(CSG_OT_difference, "Difference", "SELECT_SUBTRACT")
        item(CSG_OT_intersection, "Intersection", "SELECT_INTERSECT")


def menu_object_csg(self, context):
    self.layout.separator()
    self.layout.menu(VIEW3D_MT_object_csg.bl_idname, icon="MOD_BOOLEAN")


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

CLASSES = (
    CSG_OT_union,
    CSG_OT_slice,
    CSG_OT_difference,
    CSG_OT_intersection,
    VIEW3D_MT_object_csg,
)


def _remove_menu_entries() -> None:
    """Убрать ВСЕ пункты меню CSG, добавленные этим файлом.

    Если аддон был зарегистрирован дважды (запуск из Text Editor + установка,
    перезагрузка скриптов, дубликат файла), в меню остаются «осиротевшие»
    функции с другим id — удаляем их по имени, а не по ссылке.
    """
    menu = bpy.types.VIEW3D_MT_object
    funcs = getattr(menu.draw, "_draw_funcs", [])
    for f in list(funcs):
        if getattr(f, "__name__", "") == "menu_object_csg":
            try:
                menu.remove(f)
            except Exception:
                pass


def register():
    _remove_menu_entries()
    for cls in CLASSES:
        try:
            bpy.utils.unregister_class(cls)   # на случай «старой» регистрации
        except Exception:
            pass
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_object.append(menu_object_csg)


def unregister():
    _remove_menu_entries()
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        unregister()
    except Exception:
        pass
    register()
