# SPDX-License-Identifier: GPL-3.0-or-later
"""Face Edge Cut — разрез выделенных граней по кратчайшему пути между двумя рёбрами."""

bl_info = {
    "name": "Face Edge Cut",
    "author": "Anysphere Cursor",
    "version": (1, 2, 0),
    "blender": (5, 2, 0),
    "location": "Edit Mode > Toolbar (рядом с Loop Cut)",
    "description": (
        "Инструмент Edit Mode: разрез только по выделенным граням — "
        "кратчайший путь от точки на одном ребре до точки на другом. "
        "Gizmo и Snap to Grid для точной настройки."
    ),
    "category": "Mesh",
    "doc_url": "",
    "tracker_url": "",
}

import math
from collections import deque
from typing import Iterable, List, Optional, Sequence, Tuple

import bpy
import bmesh
from bpy.props import (
    BoolProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    StringProperty,
)
from mathutils import Matrix, Vector


# ---------------------------------------------------------------------------
# Константы solver / подготовка
# ---------------------------------------------------------------------------

SOLVER = "EXACT"
EPS_PLANAR = 1e-5
EPS_EMPTY = 1e-12
TEMP_PREFIX = ".csg_tmp_"


# ---------------------------------------------------------------------------
# Утилиты выбора / контекста
# ---------------------------------------------------------------------------

def _is_mesh_object(obj: bpy.types.Object) -> bool:
    return obj is not None and obj.type == "MESH"


def _selected_mesh_objects(context: bpy.types.Context) -> List[bpy.types.Object]:
    return [obj for obj in context.selected_objects if _is_mesh_object(obj)]


def _ensure_object_mode(context: bpy.types.Context) -> None:
    if context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def _set_active(context: bpy.types.Context, obj: bpy.types.Object) -> None:
    context.view_layer.objects.active = obj


def _select_only(context: bpy.types.Context, objects: Sequence[bpy.types.Object],
                 active: Optional[bpy.types.Object] = None) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        if obj is not None:
            obj.select_set(True)
    if active is not None:
        _set_active(context, active)
    elif objects:
        _set_active(context, objects[0])


def _depsgraph(context: bpy.types.Context):
    return context.evaluated_depsgraph_get()


# ---------------------------------------------------------------------------
# Анализ / триангуляция
# ---------------------------------------------------------------------------

def _face_is_nonplanar(coords: Sequence[Vector], eps: float = EPS_PLANAR) -> bool:
    """True, если полигон заметно неплоскостной (искажает boolean / shading)."""
    n = len(coords)
    if n <= 3:
        return False

    origin = coords[0]
    # Нормаль по Newell
    normal = Vector((0.0, 0.0, 0.0))
    for i in range(n):
        a = coords[i]
        b = coords[(i + 1) % n]
        normal.x += (a.y - b.y) * (a.z + b.z)
        normal.y += (a.z - b.z) * (a.x + b.x)
        normal.z += (a.x - b.x) * (a.y + b.y)

    if normal.length_squared <= eps * eps:
        return True

    normal.normalize()
    # Максимальное отклонение вершин от плоскости
    max_dist = 0.0
    for v in coords[1:]:
        max_dist = max(max_dist, abs((v - origin).dot(normal)))
    return max_dist > eps


def mesh_needs_triangulation(me: bpy.types.Mesh) -> bool:
    """Триангулировать, если есть ngon или неплоскостные quad/ngon."""
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bm.faces.ensure_lookup_table()
        for face in bm.faces:
            n = len(face.verts)
            if n > 4:
                return True
            if n == 4:
                coords = [v.co.copy() for v in face.verts]
                if _face_is_nonplanar(coords):
                    return True
        return False
    finally:
        bm.free()


def triangulate_mesh(me: bpy.types.Mesh) -> None:
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bmesh.ops.triangulate(
            bm,
            faces=bm.faces[:],
            quad_method="BEAUTY",
            ngon_method="BEAUTY",
        )
        bm.to_mesh(me)
        me.update()
    finally:
        bm.free()


def ensure_reliable_topology(obj: bpy.types.Object) -> None:
    """Подготовка меша под Exact boolean: normals + условная триангуляция."""
    me = obj.data
    if me is None:
        return

    # Пересчёт нормалей в object-space данных
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        bm.to_mesh(me)
    finally:
        bm.free()

    if mesh_needs_triangulation(me):
        triangulate_mesh(me)

    me.calc_loop_triangles()
    me.update()


def mesh_is_empty(obj: bpy.types.Object) -> bool:
    me = obj.data
    if me is None:
        return True
    if len(me.vertices) == 0 or len(me.polygons) == 0:
        return True
    # Нулевой объём / вырожденность — эвристика по bbox
    if me.vertices:
        coords = [v.co for v in me.vertices]
        mn = Vector((
            min(c.x for c in coords),
            min(c.y for c in coords),
            min(c.z for c in coords),
        ))
        mx = Vector((
            max(c.x for c in coords),
            max(c.y for c in coords),
            max(c.z for c in coords),
        ))
        if (mx - mn).length <= EPS_EMPTY:
            return True
    return False


# ---------------------------------------------------------------------------
# Копии / удаление
# ---------------------------------------------------------------------------

def duplicate_as_mesh(context: bpy.types.Context, obj: bpy.types.Object,
                      name: Optional[str] = None) -> bpy.types.Object:
    """Создаёт независимый mesh-объект с evaluated-геометрией и applied transforms."""
    dg = _depsgraph(context)
    eval_obj = obj.evaluated_get(dg)
    mesh = bpy.data.meshes.new_from_object(
        eval_obj,
        preserve_all_data_layers=True,
        depsgraph=dg,
    )

    new_obj = bpy.data.objects.new(name or (TEMP_PREFIX + obj.name), mesh)
    new_obj.matrix_world = obj.matrix_world.copy()

    # Линкуем в ту же коллекцию, что и исходник (или в scene collection)
    linked = False
    for col in obj.users_collection:
        col.objects.link(new_obj)
        linked = True
        break
    if not linked:
        context.scene.collection.objects.link(new_obj)

    # Применяем transform в данные меша (boolean надёжнее в world-space identity)
    _apply_object_transform(new_obj)
    ensure_reliable_topology(new_obj)
    return new_obj


def _apply_object_transform(obj: bpy.types.Object) -> None:
    """Bake matrix_world в вершины, сброс transform."""
    me = obj.data
    mw = obj.matrix_world.copy()
    me.transform(mw)
    me.update()
    obj.matrix_world.identity()


def delete_objects(context: bpy.types.Context, objects: Iterable[bpy.types.Object]) -> None:
    to_remove = [obj for obj in objects if obj is not None and obj.name in bpy.data.objects]
    if not to_remove:
        return
    _ensure_object_mode(context)
    _select_only(context, to_remove, active=to_remove[0])
    bpy.ops.object.delete(use_global=False)


def rename_keep_data(obj: bpy.types.Object, name: str) -> None:
    obj.name = name
    if obj.data is not None:
        obj.data.name = name


# ---------------------------------------------------------------------------
# Exact boolean
# ---------------------------------------------------------------------------

def boolean_exact(
    context: bpy.types.Context,
    target: bpy.types.Object,
    operand: bpy.types.Object,
    operation: str,
    *,
    use_self: bool = True,
    use_hole_tolerant: bool = True,
) -> bpy.types.Object:
    """
    Применяет Exact Boolean modifier к target с operand.
    operation: 'UNION' | 'DIFFERENCE' | 'INTERSECT'
    Возвращает target (изменённый in-place).
    """
    _ensure_object_mode(context)
    _select_only(context, [target], active=target)

    mod = target.modifiers.new(name="CSG_Exact", type="BOOLEAN")
    mod.operation = operation
    mod.solver = SOLVER
    mod.operand_type = "OBJECT"
    mod.object = operand
    mod.material_mode = "TRANSFER"

    # Exact-опции (есть в 5.x)
    if hasattr(mod, "use_self"):
        mod.use_self = use_self
    if hasattr(mod, "use_hole_tolerant"):
        mod.use_hole_tolerant = use_hole_tolerant

    with context.temp_override(
        object=target,
        active_object=target,
        selected_objects=[target],
        selected_editable_objects=[target],
    ):
        bpy.ops.object.modifier_apply(modifier=mod.name)

    # На всякий случай убрать незакрытый модификатор
    if "CSG_Exact" in target.modifiers:
        target.modifiers.remove(target.modifiers["CSG_Exact"])

    ensure_reliable_topology(target)
    return target


def copy_object_mesh(context: bpy.types.Context, src: bpy.types.Object,
                     name: Optional[str] = None) -> bpy.types.Object:
    """Полная копия уже подготовленного mesh-объекта (данные + transform)."""
    me = src.data.copy()
    obj = bpy.data.objects.new(name or (TEMP_PREFIX + "copy"), me)
    obj.matrix_world = src.matrix_world.copy()
    linked = False
    for col in src.users_collection:
        col.objects.link(obj)
        linked = True
        break
    if not linked:
        context.scene.collection.objects.link(obj)
    return obj


# ---------------------------------------------------------------------------
# CSG алгоритмы
# ---------------------------------------------------------------------------

def csg_union(context: bpy.types.Context, objects: Sequence[bpy.types.Object]
              ) -> bpy.types.Object:
    """Соединяет все выделенные объекты в один (Exact UNION)."""
    prepared = [duplicate_as_mesh(context, obj) for obj in objects]
    try:
        result = prepared[0]
        rest = prepared[1:]
        if rest:
            # Попарно — максимально предсказуемо для Exact
            for other in rest:
                boolean_exact(context, result, other, "UNION")
                delete_objects(context, [other])
        rename_keep_data(result, objects[0].name + "_Union")
        return result
    except Exception:
        delete_objects(context, prepared)
        raise


def csg_intersection(context: bpy.types.Context, objects: Sequence[bpy.types.Object]
                     ) -> bpy.types.Object:
    """Пересечение всех выделенных объектов (Exact INTERSECT)."""
    prepared = [duplicate_as_mesh(context, obj) for obj in objects]
    try:
        result = prepared[0]
        for other in prepared[1:]:
            boolean_exact(context, result, other, "INTERSECT")
            delete_objects(context, [other])
            if mesh_is_empty(result):
                break
        rename_keep_data(result, objects[0].name + "_Intersect")
        return result
    except Exception:
        delete_objects(context, prepared)
        raise


def csg_difference(context: bpy.types.Context, targets: Sequence[bpy.types.Object],
                   cutter: bpy.types.Object) -> List[bpy.types.Object]:
    """
    Active (cutter) вырезается из всех остальных.
    Возвращает список результатов (по одному на каждый target).
    """
    cutter_prep = duplicate_as_mesh(context, cutter, name=TEMP_PREFIX + "cutter")
    results: List[bpy.types.Object] = []
    try:
        for target in targets:
            t = duplicate_as_mesh(context, target)
            # Копия резака на каждый target — исходный cutter_prep не портится
            c = copy_object_mesh(context, cutter_prep, name=TEMP_PREFIX + "cutter_use")
            try:
                boolean_exact(context, t, c, "DIFFERENCE")
            finally:
                delete_objects(context, [c])
            rename_keep_data(t, target.name + "_Diff")
            results.append(t)
        return results
    except Exception:
        delete_objects(context, results + [cutter_prep])
        raise
    finally:
        delete_objects(context, [cutter_prep])


def csg_slice(context: bpy.types.Context, objects: Sequence[bpy.types.Object]
              ) -> List[bpy.types.Object]:
    """
    Slice по всем выделенным объектам.

    Каждый объект режется каждым из остальных Exact INTERSECT + DIFFERENCE.
    Пересечение пары (A∩B) сохраняется один раз (из объекта с меньшим индексом),
    чтобы не дублировать одинаковый объём B∩A.
    """
    # Исходные «ножи» — неизменяемые копии
    knives = [duplicate_as_mesh(context, obj, name=TEMP_PREFIX + f"knife_{i}")
              for i, obj in enumerate(objects)]

    # Рабочие фрагменты стартуют как копии исходников
    fragments: List[Tuple[bpy.types.Object, int]] = [
        (copy_object_mesh(context, knives[i], name=TEMP_PREFIX + f"frag_{i}"), i)
        for i in range(len(knives))
    ]

    try:
        for knife_idx, knife in enumerate(knives):
            next_fragments: List[Tuple[bpy.types.Object, int]] = []
            for frag, origin_idx in fragments:
                # Самим собой не режем — иначе бессмысленный self-slice
                if origin_idx == knife_idx:
                    next_fragments.append((frag, origin_idx))
                    continue

                # A∩B == B∩A — пересечение берём только с одной стороны пары
                keep_intersect = origin_idx < knife_idx

                frag_diff = frag
                knife_diff = copy_object_mesh(context, knife, name=TEMP_PREFIX + "k_diff")
                frag_isect = None
                knife_isect = None

                if keep_intersect:
                    frag_isect = copy_object_mesh(context, frag, name=TEMP_PREFIX + "frag_isect")
                    knife_isect = copy_object_mesh(context, knife, name=TEMP_PREFIX + "k_isect")

                try:
                    boolean_exact(context, frag_diff, knife_diff, "DIFFERENCE")
                    if keep_intersect:
                        boolean_exact(context, frag_isect, knife_isect, "INTERSECT")
                finally:
                    delete_objects(context, [knife_diff] + ([knife_isect] if knife_isect else []))

                for piece in (frag_diff, frag_isect):
                    if piece is None:
                        continue
                    if mesh_is_empty(piece):
                        delete_objects(context, [piece])
                    else:
                        next_fragments.append((piece, origin_idx))

            fragments = next_fragments

        # Финальные объекты
        results: List[bpy.types.Object] = []
        counters = {i: 0 for i in range(len(objects))}
        for frag, origin_idx in fragments:
            counters[origin_idx] += 1
            base = objects[origin_idx].name
            rename_keep_data(frag, f"{base}_Slice_{counters[origin_idx]:02d}")
            results.append(frag)

        return results
    except Exception:
        delete_objects(context, [f for f, _ in fragments] + knives)
        raise
    finally:
        delete_objects(context, knives)


def replace_selection_with(
    context: bpy.types.Context,
    originals: Sequence[bpy.types.Object],
    results: Sequence[bpy.types.Object],
    *,
    delete_originals: bool = True,
) -> None:
    if delete_originals:
        delete_objects(context, originals)
    _select_only(context, list(results), active=results[0] if results else None)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class CSG_OT_base(bpy.types.Operator):
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        if context.mode != "OBJECT":
            return False
        meshes = _selected_mesh_objects(context)
        return len(meshes) >= cls._min_objects()

    @classmethod
    def _min_objects(cls) -> int:
        return 2

    def _report_fail(self, exc: BaseException):
        self.report({"ERROR"}, f"CSG failed: {exc}")
        return {"CANCELLED"}


class CSG_OT_union(CSG_OT_base):
    """Соединяет все выделенные mesh-объекты (Exact UNION)."""
    bl_idname = "csg.union"
    bl_label = "Union"
    bl_description = "Exact Union: объединяет все выделенные объекты в один"

    def execute(self, context):
        objs = _selected_mesh_objects(context)
        try:
            result = csg_union(context, objs)
            replace_selection_with(context, objs, [result])
            self.report({"INFO"}, f"Union: {len(objs)} → 1")
            return {"FINISHED"}
        except Exception as exc:
            return self._report_fail(exc)


class CSG_OT_intersection(CSG_OT_base):
    """Оставляет общую часть всех выделенных объектов (Exact INTERSECT)."""
    bl_idname = "csg.intersection"
    bl_label = "Intersection"
    bl_description = "Exact Intersection: общая часть всех выделенных объектов"

    def execute(self, context):
        objs = _selected_mesh_objects(context)
        try:
            result = csg_intersection(context, objs)
            replace_selection_with(context, objs, [result])
            if mesh_is_empty(result):
                self.report({"WARNING"}, "Intersection пустое — объекты не пересекаются")
            else:
                self.report({"INFO"}, f"Intersection: {len(objs)} → 1")
            return {"FINISHED"}
        except Exception as exc:
            return self._report_fail(exc)


class CSG_OT_difference(CSG_OT_base):
    """Активный объект — резак; вырезается из всех остальных выделенных."""
    bl_idname = "csg.difference"
    bl_label = "Difference"
    bl_description = (
        "Exact Difference: активный объект — резак, "
        "из остальных выделенных всё вырезается"
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        if context.mode != "OBJECT":
            return False
        active = context.active_object
        if not _is_mesh_object(active):
            return False
        others = [o for o in _selected_mesh_objects(context) if o != active]
        return len(others) >= 1

    def execute(self, context):
        cutter = context.active_object
        targets = [o for o in _selected_mesh_objects(context) if o != cutter]
        try:
            results = csg_difference(context, targets, cutter)
            # Удаляем targets + cutter, оставляем результаты
            replace_selection_with(context, list(targets) + [cutter], results)
            self.report({"INFO"}, f"Difference: вырезано из {len(results)} объект(ов)")
            return {"FINISHED"}
        except Exception as exc:
            return self._report_fail(exc)


class CSG_OT_slice(CSG_OT_base):
    """Режет все выделенные объекты друг другом, сохраняя все фрагменты."""
    bl_idname = "csg.slice"
    bl_label = "Slice"
    bl_description = (
        "Exact Slice: все выделенные объекты режут друг друга; "
        "сохраняются все непустые фрагменты"
    )

    def execute(self, context):
        objs = _selected_mesh_objects(context)
        try:
            results = csg_slice(context, objs)
            replace_selection_with(context, objs, results)
            self.report({"INFO"}, f"Slice: {len(objs)} → {len(results)} фрагмент(ов)")
            return {"FINISHED"}
        except Exception as exc:
            return self._report_fail(exc)


# ---------------------------------------------------------------------------
# UI: Object → CSG
# ---------------------------------------------------------------------------

class VIEW3D_MT_object_csg(bpy.types.Menu):
    bl_label = "CSG"
    bl_idname = "VIEW3D_MT_object_csg"

    def draw(self, context):
        layout = self.layout
        layout.operator(CSG_OT_union.bl_idname, text="Union", icon="SELECT_EXTEND")
        layout.operator(CSG_OT_slice.bl_idname, text="Slice", icon="MOD_BOOLEAN")
        layout.operator(CSG_OT_difference.bl_idname, text="Difference", icon="SELECT_SUBTRACT")
        layout.operator(CSG_OT_intersection.bl_idname, text="Intersection", icon="SELECT_INTERSECT")


def menu_object_csg(self, context):
    self.layout.separator()
    self.layout.menu(VIEW3D_MT_object_csg.bl_idname, icon="MOD_BOOLEAN")


# ---------------------------------------------------------------------------
# Face Edge Cut (Edit Mode tool — группа Loop Cut)
# ---------------------------------------------------------------------------

_EDGE_CUT_MODAL = None  # type: Optional["MESH_OT_csg_face_edge_cut"]
_FACTOR_UPDATE_LOCK = False


def _viewport_grid_step(context: bpy.types.Context) -> float:
    """Шаг мелкой сетки viewport (major / subdivisions), с учётом unit scale."""
    space = getattr(context, "space_data", None)
    if space is None or not hasattr(space, "overlay"):
        return 1.0
    overlay = space.overlay
    major = float(overlay.grid_scale_unit) if overlay.grid_scale_unit > 1e-12 else float(overlay.grid_scale)
    if major <= 1e-12:
        major = 1.0
    subdiv = max(1, int(overlay.grid_subdivisions))
    step = major / float(subdiv)

    # В axis-ortho Blender уплотняет сетку при приближении — приближаем поведение
    rv3d = getattr(context, "region_data", None)
    region = getattr(context, "region", None)
    if rv3d is not None and region is not None and rv3d.view_perspective == "ORTHO":
        view = getattr(rv3d, "view", "USER")
        if view in {"FRONT", "BACK", "LEFT", "RIGHT", "TOP", "BOTTOM"}:
            # Примерный world-extent по высоте вида
            extent = max(rv3d.view_distance * 2.0, 1e-8)
            # Держим шаг не крупнее ~1/8 видимой области и не мельче pixel-ish clamp
            while step > extent / 4.0:
                step *= 0.5
            while step < extent / 512.0 and step > 1e-12:
                step *= 2.0
    return max(step, 1e-12)


def _edge_grid_factors(p0: Vector, p1: Vector, step: float) -> List[float]:
    """Factor'ы вдоль ребра в точках пересечения с плоскостями сетки X/Y/Z = n*step."""
    factors = [0.0, 1.0]
    delta = p1 - p0
    for axis in range(3):
        da = delta[axis]
        if abs(da) < 1e-12:
            continue
        a0 = p0[axis]
        a1 = p1[axis]
        amin, amax = (a0, a1) if a0 <= a1 else (a1, a0)
        n_start = math.ceil((amin - 1e-9) / step)
        n_end = math.floor((amax + 1e-9) / step)
        for n in range(n_start, n_end + 1):
            plane = n * step
            t = (plane - a0) / da
            if 0.0 <= t <= 1.0:
                factors.append(float(t))
    # Уникальные с допуском
    factors.sort()
    uniq: List[float] = []
    for t in factors:
        if not uniq or abs(uniq[-1] - t) > 1e-8:
            uniq.append(t)
    return uniq


def _snap_factor_to_grid(p0: Vector, p1: Vector, factor: float, step: float) -> float:
    candidates = _edge_grid_factors(p0, p1, step)
    return min(candidates, key=lambda t: abs(t - factor))


def _parse_faces_csv(csv: str) -> List[int]:
    if not csv.strip():
        return []
    out: List[int] = []
    for part in csv.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _find_edge(bm: bmesh.types.BMesh, i0: int, i1: int):
    bm.verts.ensure_lookup_table()
    if i0 >= len(bm.verts) or i1 >= len(bm.verts):
        return None
    v0 = bm.verts[i0]
    v1 = bm.verts[i1]
    for e in v0.link_edges:
        if e.other_vert(v0) is v1:
            return e
    return None


def _split_edge_at_factor(edge, factor: float):
    """Разрез ребра: factor 0 → verts[0], 1 → verts[1]. Возвращает BMVert."""
    v0, v1 = edge.verts[0], edge.verts[1]
    f = max(0.0, min(1.0, float(factor)))
    if f <= 1e-8:
        return v0
    if f >= 1.0 - 1e-8:
        return v1
    _new_edge, new_vert = bmesh.utils.edge_split(edge, v0, f)
    return new_vert


def _closest_factor_on_edge_to_segment(
    ea: Vector, eb: Vector, p0: Vector, p1: Vector
) -> float:
    """Factor на ребре ea–eb для точки, ближайшей к отрезку p0–p1."""
    u = eb - ea
    a = u.length_squared
    if a < 1e-18:
        return 0.0

    v = p1 - p0
    w0 = ea - p0
    b = u.dot(v)
    c = v.dot(v)
    d = u.dot(w0)
    e = v.dot(w0)
    denom = a * c - b * b

    if c < 1e-18:
        # p0≈p1 — проекция точки на ребро
        return max(0.0, min(1.0, -d / a))

    if abs(denom) < 1e-18:
        # Параллельны — проекция середины отрезка
        mid = (p0 + p1) * 0.5
        return max(0.0, min(1.0, (mid - ea).dot(u) / a))

    # Ближайшие точки на бесконечных прямых, затем clamp к отрезкам
    sn = b * e - c * d
    tn = a * e - b * d
    if sn < 0.0:
        sn = 0.0
        tn = e
        # t = e/c when s=0, but e = v·w0; use projection of ea onto p0p1 then back
        t = max(0.0, min(1.0, e / c))
        closest = p0 + v * t
        return max(0.0, min(1.0, (closest - ea).dot(u) / a))
    if sn > denom:
        sn = denom
        t = max(0.0, min(1.0, (e + b) / c))
        closest = p0 + v * t
        return max(0.0, min(1.0, (closest - ea).dot(u) / a))

    s = sn / denom
    t = tn / denom
    if t < 0.0:
        closest = p0
        return max(0.0, min(1.0, (closest - ea).dot(u) / a))
    if t > 1.0:
        closest = p1
        return max(0.0, min(1.0, (closest - ea).dot(u) / a))
    return max(0.0, min(1.0, s))


def _face_neighbors_in_allowed(face, allowed: set):
    """Соседние грани из allowed и общее ребро."""
    for edge in face.edges:
        for other in edge.link_faces:
            if other is not face and other in allowed:
                yield other, edge


def _shortest_face_path(allowed: set, starts: Sequence, goals: Sequence):
    """
    BFS по dual-графу выделенных граней.
    Возвращает (faces, shared_edges) или None.
    shared_edges[i] — ребро между faces[i] и faces[i+1].
    """
    if not starts or not goals:
        return None
    goal_set = set(goals)
    start_list = list(starts)

    for s in start_list:
        if s in goal_set:
            return [s], []

    parent = {}  # face -> (prev_face, shared_edge)
    visited = set()
    q = deque()
    for s in start_list:
        if s in visited:
            continue
        visited.add(s)
        parent[s] = (None, None)
        q.append(s)

    found = None
    while q:
        cur = q.popleft()
        for nb, edge in _face_neighbors_in_allowed(cur, allowed):
            if nb in visited:
                continue
            visited.add(nb)
            parent[nb] = (cur, edge)
            if nb in goal_set:
                found = nb
                q.clear()
                break
            q.append(nb)

    if found is None:
        return None

    faces = []
    edges_between = []
    cur = found
    while cur is not None:
        faces.append(cur)
        prev, edge = parent[cur]
        if edge is not None:
            edges_between.append(edge)
        cur = prev
    faces.reverse()
    edges_between.reverse()
    return faces, edges_between


def _cut_along_selected_faces(bm: bmesh.types.BMesh, allowed: set, va, vb) -> None:
    """
    Разрез по кратчайшему пути граней (только allowed) между вершинами va и vb.
    На общих рёбрах пути ставит точки ближе к отрезку va–vb, затем face_split.
    """
    if va is vb:
        return

    # Уже на одной выделенной грани
    shared = [f for f in va.link_faces if f in vb.link_faces and f in allowed]
    if shared:
        bmesh.utils.face_split(shared[0], va, vb)
        return

    starts = [f for f in va.link_faces if f in allowed]
    goals = [f for f in vb.link_faces if f in allowed]
    path = _shortest_face_path(allowed, starts, goals)
    if path is None:
        raise RuntimeError(
            "Нет пути по выделенным граням между выбранными рёбрами"
        )

    faces, shared_edges = path
    if not shared_edges:
        if faces:
            bmesh.utils.face_split(faces[0], va, vb)
        return

    p0 = va.co.copy()
    p1 = vb.co.copy()

    cut_verts = [va]
    for edge in shared_edges:
        ev0, ev1 = edge.verts[0], edge.verts[1]
        if va in edge.verts:
            cut_verts.append(va)
            continue
        if vb in edge.verts:
            cut_verts.append(vb)
            continue
        factor = _closest_factor_on_edge_to_segment(ev0.co, ev1.co, p0, p1)
        cut_verts.append(_split_edge_at_factor(edge, factor))
    cut_verts.append(vb)

    for i, face in enumerate(faces):
        v_from = cut_verts[i]
        v_to = cut_verts[i + 1]
        if v_from is v_to:
            continue
        common = [
            f for f in v_from.link_faces
            if f in v_to.link_faces and (f in allowed or f is face)
        ]
        if not common and face.is_valid and v_from in face.verts and v_to in face.verts:
            common = [face]
        if not common:
            raise RuntimeError("Не удалось разрезать грань на пути")
        # Предпочитаем грань из исходного пути / allowed
        target = next((f for f in common if f in allowed), common[0])
        if not target.is_valid:
            target = common[0]
        bmesh.utils.face_split(target, v_from, v_to)


def _cut_between_edge_points(
    bm: bmesh.types.BMesh,
    face_indices: Sequence[int],
    e0_v0: int,
    e0_v1: int,
    e1_v0: int,
    e1_v1: int,
    factor_a: float,
    factor_b: float,
) -> None:
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    allowed = set()
    for idx in face_indices:
        if 0 <= idx < len(bm.faces):
            allowed.add(bm.faces[idx])
    if not allowed:
        raise RuntimeError("Сохранённые грани не найдены")

    edge_a = _find_edge(bm, e0_v0, e0_v1)
    edge_b = _find_edge(bm, e1_v0, e1_v1)
    if edge_a is None or edge_b is None:
        raise RuntimeError("Сохранённые рёбра не найдены")

    def edge_ok(edge) -> bool:
        return any(f in allowed for f in edge.link_faces)

    if not edge_ok(edge_a) or not edge_ok(edge_b):
        raise RuntimeError("Рёбра должны принадлежать ранее выбранным граням")

    # Ориентация factor: от сохранённого v0 к v1
    if edge_a.verts[0].index == e0_v0:
        fa = factor_a
    else:
        fa = 1.0 - factor_a
    if edge_b.verts[0].index == e1_v0:
        fb = factor_b
    else:
        fb = 1.0 - factor_b

    va = _split_edge_at_factor(edge_a, fa)
    vb = _split_edge_at_factor(edge_b, fb)
    if va is vb:
        return

    _cut_along_selected_faces(bm, allowed, va, vb)


class MESH_OT_csg_face_edge_cut(bpy.types.Operator):
    """Разрез только по выделенным граням: кратчайший путь между двумя рёбрами."""

    bl_idname = "mesh.csg_face_edge_cut"
    bl_label = "Face Edge Cut"
    bl_options = {"REGISTER", "UNDO"}
    bl_description = (
        "Выделите грани → Enter → 2 ребра → Enter. "
        "Разрез идёт по кратчайшему пути только по этим граням; "
        "gizmo и панель — для factor'ов"
    )

    faces_csv: StringProperty(name="Faces", options={"HIDDEN", "SKIP_SAVE"})
    edge_a0: IntProperty(name="Edge A v0", default=-1, options={"HIDDEN", "SKIP_SAVE"})
    edge_a1: IntProperty(name="Edge A v1", default=-1, options={"HIDDEN", "SKIP_SAVE"})
    edge_b0: IntProperty(name="Edge B v0", default=-1, options={"HIDDEN", "SKIP_SAVE"})
    edge_b1: IntProperty(name="Edge B v1", default=-1, options={"HIDDEN", "SKIP_SAVE"})

    edge_a_co0: FloatVectorProperty(name="A0", size=3, subtype="TRANSLATION", options={"HIDDEN", "SKIP_SAVE"})
    edge_a_co1: FloatVectorProperty(name="A1", size=3, subtype="TRANSLATION", options={"HIDDEN", "SKIP_SAVE"})
    edge_b_co0: FloatVectorProperty(name="B0", size=3, subtype="TRANSLATION", options={"HIDDEN", "SKIP_SAVE"})
    edge_b_co1: FloatVectorProperty(name="B1", size=3, subtype="TRANSLATION", options={"HIDDEN", "SKIP_SAVE"})

    def _update_factor_a(self, context):
        self._on_factor_changed(context, which="a")

    def _update_factor_b(self, context):
        self._on_factor_changed(context, which="b")

    def _update_snap(self, context):
        global _FACTOR_UPDATE_LOCK
        if _FACTOR_UPDATE_LOCK:
            return
        if self.use_snap_grid and getattr(self, "_phase", None) == "ADJUST":
            _FACTOR_UPDATE_LOCK = True
            try:
                step = _viewport_grid_step(context)
                p0 = Vector(self.edge_a_co0)
                p1 = Vector(self.edge_a_co1)
                self.factor_a = _snap_factor_to_grid(p0, p1, self.factor_a, step)
                p0 = Vector(self.edge_b_co0)
                p1 = Vector(self.edge_b_co1)
                self.factor_b = _snap_factor_to_grid(p0, p1, self.factor_b, step)
            finally:
                _FACTOR_UPDATE_LOCK = False
            self._reapply_from_backup(context)
        elif getattr(self, "_phase", None) == "ADJUST":
            self._reapply_from_backup(context)

    factor_a: FloatProperty(
        name="Factor A",
        description="Точка разреза на первом ребре (0–1 по длине)",
        default=0.5,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        update=_update_factor_a,
    )
    factor_b: FloatProperty(
        name="Factor B",
        description="Точка разреза на втором ребре (0–1 по длине)",
        default=0.5,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        update=_update_factor_b,
    )
    use_snap_grid: BoolProperty(
        name="Snap to Grid",
        description="Привязка factor/gizmo к пересечениям ребра с сеткой viewport",
        default=False,
        update=_update_snap,
    )

    def _on_factor_changed(self, context, *, which: str):
        global _FACTOR_UPDATE_LOCK
        if _FACTOR_UPDATE_LOCK:
            return
        if getattr(self, "_phase", None) != "ADJUST":
            return
        if self.use_snap_grid:
            _FACTOR_UPDATE_LOCK = True
            try:
                step = _viewport_grid_step(context)
                if which == "a":
                    p0 = Vector(self.edge_a_co0)
                    p1 = Vector(self.edge_a_co1)
                    snapped = _snap_factor_to_grid(p0, p1, self.factor_a, step)
                    if abs(snapped - self.factor_a) > 1e-10:
                        self.factor_a = snapped
                        return
                else:
                    p0 = Vector(self.edge_b_co0)
                    p1 = Vector(self.edge_b_co1)
                    snapped = _snap_factor_to_grid(p0, p1, self.factor_b, step)
                    if abs(snapped - self.factor_b) > 1e-10:
                        self.factor_b = snapped
                        return
            finally:
                _FACTOR_UPDATE_LOCK = False
        self._reapply_from_backup(context)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "factor_a")
        layout.prop(self, "factor_b")
        layout.prop(self, "use_snap_grid")

    @classmethod
    def poll(cls, context):
        ob = context.edit_object
        return context.mode == "EDIT_MESH" and ob is not None and ob.type == "MESH"

    def invoke(self, context, event):
        # Redo / вызов с уже заполненными рёбрами → сразу execute
        if self.edge_a0 >= 0 and self.edge_b0 >= 0 and self.faces_csv:
            return self.execute(context)

        ob = context.edit_object
        me = ob.data
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()
        faces = [f for f in bm.faces if f.select]
        if not faces:
            self.report({"ERROR"}, "Сначала выделите грани, затем нажмите Enter")
            return {"CANCELLED"}

        self.faces_csv = ",".join(str(f.index) for f in faces)
        for f in bm.faces:
            f.select = False
        for e in bm.edges:
            e.select = False
        for v in bm.verts:
            v.select = False
        bmesh.update_edit_mesh(me)

        self._phase = "EDGES"
        self._backup_mesh = None
        global _EDGE_CUT_MODAL
        _EDGE_CUT_MODAL = self
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set(
            "Face Edge Cut: выделите ровно 2 ребра на выбранных гранях, Enter — разрез"
        )
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        global _EDGE_CUT_MODAL

        if self._phase == "EDGES":
            if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
                return self._confirm_edges_and_cut(context)
            if event.type in {"ESC"} and event.value == "PRESS":
                context.workspace.status_text_set(None)
                _EDGE_CUT_MODAL = None
                return {"CANCELLED"}
            return {"PASS_THROUGH"}

        if self._phase == "ADJUST":
            if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
                self._cleanup_backup()
                context.workspace.status_text_set(None)
                _EDGE_CUT_MODAL = None
                self._phase = "DONE"
                return {"FINISHED"}
            if event.type in {"ESC"} and event.value == "PRESS":
                self._restore_backup(context)
                self._cleanup_backup()
                context.workspace.status_text_set(None)
                _EDGE_CUT_MODAL = None
                return {"CANCELLED"}
            # Камера, gizmo, UI — не блокируем события
            return {"PASS_THROUGH"}

        return {"CANCELLED"}

    def _confirm_edges_and_cut(self, context):
        ob = context.edit_object
        me = ob.data
        bm = bmesh.from_edit_mesh(me)
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()

        face_indices = _parse_faces_csv(self.faces_csv)
        allowed = {bm.faces[i] for i in face_indices if 0 <= i < len(bm.faces)}
        edges = [e for e in bm.edges if e.select]
        if len(edges) != 2:
            self.report({"ERROR"}, "Нужно выделить ровно 2 ребра")
            return {"RUNNING_MODAL"}

        for e in edges:
            if not any(f in allowed for f in e.link_faces):
                self.report({"ERROR"}, "Оба ребра должны принадлежать ранее выбранным граням")
                return {"RUNNING_MODAL"}

        ea, eb = edges[0], edges[1]
        self.edge_a0, self.edge_a1 = ea.verts[0].index, ea.verts[1].index
        self.edge_b0, self.edge_b1 = eb.verts[0].index, eb.verts[1].index

        mw = ob.matrix_world
        self.edge_a_co0 = (mw @ ea.verts[0].co).to_tuple()
        self.edge_a_co1 = (mw @ ea.verts[1].co).to_tuple()
        self.edge_b_co0 = (mw @ eb.verts[0].co).to_tuple()
        self.edge_b_co1 = (mw @ eb.verts[1].co).to_tuple()

        # Сброс выделения рёбер
        for e in bm.edges:
            e.select = False
        bmesh.update_edit_mesh(me)

        if self.use_snap_grid:
            step = _viewport_grid_step(context)
            self.factor_a = _snap_factor_to_grid(
                Vector(self.edge_a_co0), Vector(self.edge_a_co1), self.factor_a, step
            )
            self.factor_b = _snap_factor_to_grid(
                Vector(self.edge_b_co0), Vector(self.edge_b_co1), self.factor_b, step
            )

        self._save_backup(context)
        try:
            self._apply_cut(context)
        except Exception as exc:
            self._restore_backup(context)
            self._cleanup_backup()
            self.report({"ERROR"}, f"Cut failed: {exc}")
            global _EDGE_CUT_MODAL
            _EDGE_CUT_MODAL = None
            context.workspace.status_text_set(None)
            return {"CANCELLED"}

        self._phase = "ADJUST"
        context.workspace.status_text_set(
            "Face Edge Cut: крутите gizmo / factor в панели снизу слева, Enter — подтвердить"
        )
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        return {"RUNNING_MODAL"}

    def _save_backup(self, context):
        self._cleanup_backup()
        ob = context.edit_object
        bm = bmesh.from_edit_mesh(ob.data)
        self._backup_mesh = bpy.data.meshes.new(TEMP_PREFIX + "edge_cut_backup")
        tmp = bm.copy()
        tmp.to_mesh(self._backup_mesh)
        tmp.free()

    def _restore_backup(self, context):
        if not getattr(self, "_backup_mesh", None):
            return
        ob = context.edit_object
        bm = bmesh.from_edit_mesh(ob.data)
        bm.clear()
        bm.from_mesh(self._backup_mesh)
        bmesh.update_edit_mesh(ob.data, loop_triangles=True, destructive=True)

    def _cleanup_backup(self):
        mesh = getattr(self, "_backup_mesh", None)
        if mesh is not None:
            if mesh.users == 0 or mesh.name.startswith(TEMP_PREFIX):
                bpy.data.meshes.remove(mesh, do_unlink=True)
            self._backup_mesh = None

    def _reapply_from_backup(self, context):
        if getattr(self, "_phase", None) != "ADJUST":
            return
        if not getattr(self, "_backup_mesh", None):
            return
        try:
            self._restore_backup(context)
            self._apply_cut(context)
        except Exception as exc:
            self.report({"ERROR"}, f"Cut failed: {exc}")

    def _apply_cut(self, context):
        ob = context.edit_object
        me = ob.data
        bm = bmesh.from_edit_mesh(me)
        fa, fb = self.factor_a, self.factor_b
        if self.use_snap_grid:
            step = _viewport_grid_step(context)
            fa = _snap_factor_to_grid(Vector(self.edge_a_co0), Vector(self.edge_a_co1), fa, step)
            fb = _snap_factor_to_grid(Vector(self.edge_b_co0), Vector(self.edge_b_co1), fb, step)
        _cut_between_edge_points(
            bm,
            _parse_faces_csv(self.faces_csv),
            self.edge_a0,
            self.edge_a1,
            self.edge_b0,
            self.edge_b1,
            fa,
            fb,
        )
        bmesh.update_edit_mesh(me, loop_triangles=True, destructive=True)
        me.update()

    def execute(self, context):
        # Путь Redo (Adjust Last Operation): меш уже откатан Blender'ом
        if self.edge_a0 < 0 or self.edge_b0 < 0 or not self.faces_csv:
            self.report({"ERROR"}, "Нет данных разреза")
            return {"CANCELLED"}
        try:
            fa, fb = self.factor_a, self.factor_b
            if self.use_snap_grid:
                global _FACTOR_UPDATE_LOCK
                step = _viewport_grid_step(context)
                fa = _snap_factor_to_grid(Vector(self.edge_a_co0), Vector(self.edge_a_co1), fa, step)
                fb = _snap_factor_to_grid(Vector(self.edge_b_co0), Vector(self.edge_b_co1), fb, step)
                if abs(fa - self.factor_a) > 1e-10 or abs(fb - self.factor_b) > 1e-10:
                    _FACTOR_UPDATE_LOCK = True
                    try:
                        self.factor_a = fa
                        self.factor_b = fb
                    finally:
                        _FACTOR_UPDATE_LOCK = False
            self._apply_cut(context)
        except Exception as exc:
            self.report({"ERROR"}, f"Cut failed: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}

    def cancel(self, context):
        global _EDGE_CUT_MODAL
        if getattr(self, "_phase", None) == "ADJUST":
            self._restore_backup(context)
        self._cleanup_backup()
        context.workspace.status_text_set(None)
        _EDGE_CUT_MODAL = None


def _edge_cut_active_op(context):
    global _EDGE_CUT_MODAL
    if _EDGE_CUT_MODAL is not None and getattr(_EDGE_CUT_MODAL, "_phase", None) == "ADJUST":
        return _EDGE_CUT_MODAL
    ops = context.window_manager.operators
    if ops and ops[-1].bl_idname == MESH_OT_csg_face_edge_cut.bl_idname:
        return ops[-1]
    return None


def _op_prop(op, name):
    """Свойство модального Operator или RNA-оператора из undo-стека."""
    if hasattr(op, "properties"):
        props = op.properties
        if hasattr(props, name):
            return getattr(props, name)
    return getattr(op, name)


def _op_set_prop(op, name, value):
    if hasattr(op, "properties") and hasattr(op.properties, name):
        setattr(op.properties, name, value)
    else:
        setattr(op, name, value)


def _matrix_along_edge(p0: Vector, p1: Vector) -> Matrix:
    z_axis = p1 - p0
    length = z_axis.length
    if length < 1e-12:
        return Matrix.Translation(p0)
    z_axis.normalize()
    x_axis = z_axis.orthogonal().normalized()
    y_axis = z_axis.cross(x_axis).normalized()
    mat = Matrix.Identity(4)
    mat.col[0].xyz = x_axis
    mat.col[1].xyz = y_axis
    mat.col[2].xyz = z_axis
    mat.col[3].xyz = p0
    return mat


class CSG_GGT_face_edge_cut(bpy.types.GizmoGroup):
    bl_idname = "CSG_GGT_face_edge_cut"
    bl_label = "Face Edge Cut Gizmos"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options = {"3D", "PERSISTENT", "SHOW_MODAL_ALL"}

    @classmethod
    def poll(cls, context):
        if context.mode != "EDIT_MESH":
            return False
        return _edge_cut_active_op(context) is not None

    def setup(self, context):
        def make_arrow(which: str):
            gz = self.gizmos.new("GIZMO_GT_arrow_3d")
            gz.draw_style = "BOX"
            gz.scale_basis = 0.7
            gz.color = (0.2, 0.85, 1.0) if which == "a" else (1.0, 0.55, 0.15)
            gz.alpha = 0.65
            gz.color_highlight = (1.0, 1.0, 1.0)
            gz.alpha_highlight = 1.0
            gz.use_draw_modal = True
            gz.use_draw_value = True

            def get_offset():
                op = _edge_cut_active_op(context)
                if op is None:
                    return 0.0
                if which == "a":
                    p0 = Vector(_op_prop(op, "edge_a_co0"))
                    p1 = Vector(_op_prop(op, "edge_a_co1"))
                    fac = float(_op_prop(op, "factor_a"))
                else:
                    p0 = Vector(_op_prop(op, "edge_b_co0"))
                    p1 = Vector(_op_prop(op, "edge_b_co1"))
                    fac = float(_op_prop(op, "factor_b"))
                return fac * (p1 - p0).length

            def set_offset(value):
                global _FACTOR_UPDATE_LOCK
                op = _edge_cut_active_op(context)
                if op is None:
                    return
                if which == "a":
                    p0 = Vector(_op_prop(op, "edge_a_co0"))
                    p1 = Vector(_op_prop(op, "edge_a_co1"))
                else:
                    p0 = Vector(_op_prop(op, "edge_b_co0"))
                    p1 = Vector(_op_prop(op, "edge_b_co1"))
                length = (p1 - p0).length
                if length < 1e-12:
                    return
                factor = max(0.0, min(1.0, float(value) / length))
                if bool(_op_prop(op, "use_snap_grid")):
                    factor = _snap_factor_to_grid(p0, p1, factor, _viewport_grid_step(context))

                _FACTOR_UPDATE_LOCK = True
                try:
                    _op_set_prop(op, "factor_a" if which == "a" else "factor_b", factor)
                finally:
                    _FACTOR_UPDATE_LOCK = False

                if getattr(op, "_phase", None) == "ADJUST":
                    op._reapply_from_backup(context)
                else:
                    props = {
                        "faces_csv": _op_prop(op, "faces_csv"),
                        "edge_a0": int(_op_prop(op, "edge_a0")),
                        "edge_a1": int(_op_prop(op, "edge_a1")),
                        "edge_b0": int(_op_prop(op, "edge_b0")),
                        "edge_b1": int(_op_prop(op, "edge_b1")),
                        "edge_a_co0": tuple(_op_prop(op, "edge_a_co0")),
                        "edge_a_co1": tuple(_op_prop(op, "edge_a_co1")),
                        "edge_b_co0": tuple(_op_prop(op, "edge_b_co0")),
                        "edge_b_co1": tuple(_op_prop(op, "edge_b_co1")),
                        "factor_a": float(_op_prop(op, "factor_a")),
                        "factor_b": float(_op_prop(op, "factor_b")),
                        "use_snap_grid": bool(_op_prop(op, "use_snap_grid")),
                    }
                    if which == "a":
                        props["factor_a"] = factor
                    else:
                        props["factor_b"] = factor
                    bpy.ops.ed.undo()
                    bpy.ops.mesh.csg_face_edge_cut(**props)

            def get_range():
                op = _edge_cut_active_op(context)
                if op is None:
                    return (0.0, 1.0)
                if which == "a":
                    length = (
                        Vector(_op_prop(op, "edge_a_co1")) - Vector(_op_prop(op, "edge_a_co0"))
                    ).length
                else:
                    length = (
                        Vector(_op_prop(op, "edge_b_co1")) - Vector(_op_prop(op, "edge_b_co0"))
                    ).length
                return (0.0, max(length, 1e-8))

            gz.target_set_handler("offset", get=get_offset, set=set_offset, range=get_range)
            return gz

        self.gizmo_a = make_arrow("a")
        self.gizmo_b = make_arrow("b")

    def refresh(self, context):
        op = _edge_cut_active_op(context)
        if op is None:
            return
        self.gizmo_a.matrix_basis = _matrix_along_edge(
            Vector(_op_prop(op, "edge_a_co0")),
            Vector(_op_prop(op, "edge_a_co1")),
        )
        self.gizmo_b.matrix_basis = _matrix_along_edge(
            Vector(_op_prop(op, "edge_b_co0")),
            Vector(_op_prop(op, "edge_b_co1")),
        )


class CSG_WT_face_edge_cut(bpy.types.WorkSpaceTool):
    bl_space_type = "VIEW_3D"
    bl_context_mode = "EDIT_MESH"
    bl_idname = "csg.face_edge_cut"
    bl_label = "Face Edge Cut"
    bl_description = (
        "Разрез выделенных граней по кратчайшему пути между двумя рёбрами. "
        "Enter → 2 ребра → Enter; gizmo / factor в панели"
    )
    bl_icon = "ops.mesh.loopcut_slide"
    bl_widget = "CSG_GGT_face_edge_cut"
    bl_options = {"KEYMAP_FALLBACK"}
    bl_keymap = (
        ("mesh.csg_face_edge_cut", {"type": "RET", "value": "PRESS"}, None),
        ("mesh.csg_face_edge_cut", {"type": "NUMPAD_ENTER", "value": "PRESS"}, None),
    )

    def draw_settings(context, layout, tool):
        props = tool.operator_properties("mesh.csg_face_edge_cut")
        layout.prop(props, "factor_a")
        layout.prop(props, "factor_b")
        layout.prop(props, "use_snap_grid")


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

CLASSES = (
    CSG_OT_union,
    CSG_OT_slice,
    CSG_OT_difference,
    CSG_OT_intersection,
    VIEW3D_MT_object_csg,
    MESH_OT_csg_face_edge_cut,
    CSG_GGT_face_edge_cut,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_object.append(menu_object_csg)
    bpy.utils.register_tool(
        CSG_WT_face_edge_cut,
        after={"builtin.loop_cut"},
        separator=False,
        group=False,
    )


def unregister():
    try:
        bpy.utils.unregister_tool(CSG_WT_face_edge_cut)
    except Exception:
        pass
    bpy.types.VIEW3D_MT_object.remove(menu_object_csg)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    try:
        unregister()
    except Exception:
        pass
    register()
