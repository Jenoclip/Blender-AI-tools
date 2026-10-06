bl_info = {
    "name": "Polyline Carve (Advanced)",
    "author": "Anthropic Claude",
    "version": (2, 1, 0),
    "blender": (5, 2, 0),
    "location": "3D Viewport > Toolbar (Object Mode and Edit Mode) / Sidebar > Tool",
    "description": "Polyline carve (как в Bool Tool) с привязкой точек: сетка, угол, вершины, рёбра",
    "category": "Object",
}

import base64
import math
import os
import bpy
import numpy as np
import bmesh
import gpu
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
from mathutils import Vector, geometry
from bpy.props import BoolProperty, EnumProperty, FloatProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup, WorkSpaceTool

EPS = 1e-12
CLOSE_PIXELS = 12.0
NAV_EVENTS = {
    'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'WHEELINMOUSE',
    'WHEELOUTMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE',
    'NDOF_MOTION',
}


# ----------------------------------------------------------------------------
# Размер сетки Blender (space.overlay.grid_scale)
# ----------------------------------------------------------------------------
def get_grid_size(space, scene):
    g = 1.0
    try:
        g = float(space.overlay.grid_scale)
    except Exception:
        pass
    us = scene.unit_settings
    if us.system != 'NONE':
        # так же, как делает сам Blender при отрисовке сетки
        if us.system == 'IMPERIAL':
            g *= 0.3048
        if us.scale_length > 0:
            g /= us.scale_length
    return max(g, 1e-6)


# ----------------------------------------------------------------------------
# Настройки
# ----------------------------------------------------------------------------
def _upd_snap_grid(self, context):
    if self.snap_grid and self.snap_angle:
        self.snap_angle = False  # сетка и угол взаимоисключают друг друга


def _upd_snap_angle(self, context):
    if self.snap_angle and self.snap_grid:
        self.snap_grid = False


class PCS_Settings(PropertyGroup):
    snap_grid: BoolProperty(
        name="Snap to Grid",
        description="Привязка точек к сетке вьюпорта (отключает привязку к углу)",
        default=True, update=_upd_snap_grid,
    )
    snap_angle: BoolProperty(
        name="Snap to Angle",
        description="Привязка направления от предыдущей точки к углу, кратному шагу "
                    "(отключает привязку к сетке)",
        default=False, update=_upd_snap_angle,
    )
    angle_step: FloatProperty(
        name="Angle Step",
        description="Кратность угла для привязки к углу",
        default=math.radians(15.0), min=math.radians(0.5), max=math.radians(90.0),
        subtype='ANGLE', unit='ROTATION',
    )
    snap_vertices: BoolProperty(
        name="Snap to Vertices",
        description="Привязка точек к вершинам объекта (в приоритете над сеткой и углом)",
        default=False,
    )
    snap_edges: BoolProperty(
        name="Snap to Edges",
        description="Привязка точек к оси ребра объекта; вместе с сеткой - к пересечению "
                    "сетки и оси ребра, вместе с углом - к пересечению линии угла и оси ребра",
        default=False,
    )
    cut_method: EnumProperty(
        name="Cut Method",
        description="Чем вырезать: выдавленной фигурой или объёмом, который образуют точки",
        items=(
            ('EXTRUDE', "Extrude Shape",
             "Фигура выдавливается в направлении Cut Direction на заданную глубину"),
            ('VOLUME', "Volume",
             "Вырезается объём - выпуклая оболочка точек (например, тетраэдр из угла "
             "параллелепипеда). Cut Direction и глубина игнорируются; нужно минимум 4 точки "
             "не в одной плоскости"),
        ),
        default='EXTRUDE',
    )
    cut_mode: EnumProperty(
        name="Cut Direction",
        description="Относительно чего вычисляется фигура выреза",
        items=(
            ('VIEW', "Camera",
             "Фигура проецируется вдоль направления камеры (в перспективе - из камеры); "
             "точки ставятся на плоскости через 3D-курсор"),
            ('NORMAL', "Face Normals",
             "Каждая точка ставится на поверхность под курсором и получает нормаль своей грани; "
             "вырез идёт вдоль нормали каждой точки"),
        ),
        default='VIEW',
    )
    auto_depth: BoolProperty(
        name="Auto Depth",
        description="Вырезать на всю глубину выделенных объектов. "
                    "Выключите, чтобы задать глубину выреза вручную",
        default=True,
    )
    depth: FloatProperty(
        name="Depth",
        description="Глубина выреза, считая от поверхности объекта (если Auto Depth выключен)",
        default=1.0, min=1e-4, subtype='DISTANCE', unit='LENGTH',
    )
    backward_depth: FloatProperty(
        name="Backward Depth",
        description="Рез обратной стороной нормалей: насколько вырез продолжается в сторону, "
                    "противоположную основному резу (к камере / наружу от поверхности). "
                    "Нужен, когда часть геометрии находится за фигурой (если Auto Depth выключен)",
        default=0.0, min=0.0, subtype='DISTANCE', unit='LENGTH',
    )
    depth_margin: FloatProperty(
        name="Margin",
        description="Запас глубины с каждой стороны при Auto Depth",
        default=0.1, min=0.0, subtype='DISTANCE', unit='LENGTH',
    )


def draw_settings_ui(layout, s, header=False):
    volume = s.cut_method == 'VOLUME'
    if header:
        # верхняя панель инструмента: все настройки в одну строку
        layout.prop(s, "cut_method", text="")
        row = layout.row()
        row.active = not volume
        row.prop(s, "cut_mode", text="")
        row = layout.row(align=True)
        row.prop(s, "snap_grid", text="Grid", toggle=True)
        row.prop(s, "snap_angle", text="Angle", toggle=True)
        sub = row.row(align=True)
        sub.active = s.snap_angle
        sub.prop(s, "angle_step", text="")
        row = layout.row(align=True)
        row.prop(s, "snap_vertices", text="Vertices", toggle=True)
        row.prop(s, "snap_edges", text="Edges", toggle=True)
        row = layout.row()
        row.active = not volume
        row.prop(s, "auto_depth")
        row = layout.row(align=True)
        row.active = not s.auto_depth and not volume
        row.prop(s, "depth")
        row.prop(s, "backward_depth")
        row = layout.row(align=True)
        row.active = s.auto_depth and not volume
        row.prop(s, "depth_margin")
        return
    col = layout.column(align=True)
    col.prop(s, "cut_method", text="")
    sub = col.column(align=True)
    sub.active = not volume
    sub.prop(s, "cut_mode", text="")
    col.separator()
    col.prop(s, "snap_vertices")
    col.prop(s, "snap_edges")
    col.prop(s, "snap_grid")
    col.prop(s, "snap_angle")
    sub = col.column(align=True)
    sub.active = s.snap_angle
    sub.prop(s, "angle_step")
    col.separator()
    sub = col.column(align=True)
    sub.active = not volume
    sub.prop(s, "auto_depth")
    row = sub.column(align=True)
    row.active = not s.auto_depth and not volume
    row.prop(s, "depth")
    row.prop(s, "backward_depth")
    row = sub.column(align=True)
    row.active = s.auto_depth and not volume
    row.prop(s, "depth_margin")


# ----------------------------------------------------------------------------
# Геометрия
# ----------------------------------------------------------------------------
def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_seg(a, b, c):
    return (min(a[0], b[0]) - EPS <= c[0] <= max(a[0], b[0]) + EPS and
            min(a[1], b[1]) - EPS <= c[1] <= max(a[1], b[1]) + EPS)


def segments_intersect(a, b, c, d):
    """Пересечение/касание отрезков ab и cd в 2D."""
    d1, d2 = _ccw(a, b, c), _ccw(a, b, d)
    d3, d4 = _ccw(c, d, a), _ccw(c, d, b)
    if ((d1 > EPS and d2 < -EPS) or (d1 < -EPS and d2 > EPS)) and \
       ((d3 > EPS and d4 < -EPS) or (d3 < -EPS and d4 > EPS)):
        return True
    if abs(d1) <= EPS and _on_seg(a, b, c):
        return True
    if abs(d2) <= EPS and _on_seg(a, b, d):
        return True
    if abs(d3) <= EPS and _on_seg(c, d, a):
        return True
    if abs(d4) <= EPS and _on_seg(c, d, b):
        return True
    return False


def triangulate_shape(pts):
    """Триангуляция фигуры (ear clipping) в плоскости средней нормали.
    Возвращает список троек индексов точек."""
    n = len(pts)
    if n < 3:
        return []
    area = Vector((0.0, 0.0, 0.0))
    for i in range(n):
        area += pts[i].cross(pts[(i + 1) % n])
    if area.length < 1e-12:
        return []
    nav = area.normalized()
    ref = Vector((0.0, 0.0, 1.0)) if abs(nav.z) < 0.9 else Vector((1.0, 0.0, 0.0))
    bu = nav.cross(ref).normalized()
    bv = nav.cross(bu)
    poly = [Vector((p.dot(bu), p.dot(bv), 0.0)) for p in pts]
    try:
        tess = geometry.tessellate_polygon([poly])
    except Exception:
        return []
    return [tuple(t) for t in tess]


def shape_vertex_normals(pts, tris, hints):
    """Нормали реза в точках фигуры - по граням (треугольникам) самой фигуры.

    hints - приблизительные внешние нормали точек: нужны только чтобы повернуть грани
    фигуры наружу. Нормаль точки = среднее нормалей смежных граней (с весом по площади)."""
    acc = [Vector((0.0, 0.0, 0.0)) for _ in pts]
    for a, b, c in tris:
        fn = (pts[b] - pts[a]).cross(pts[c] - pts[a])  # длина = удвоенная площадь
        if fn.length < 1e-12:
            continue
        if fn.dot(hints[a] + hints[b] + hints[c]) < 0:
            fn = -fn
        acc[a] += fn
        acc[b] += fn
        acc[c] += fn
    out = []
    for i, v in enumerate(acc):
        out.append(v.normalized() if v.length > 1e-9 else hints[i].normalized())
    return out


def _hull_into(bm, pts):
    """Строит в bm выпуклую оболочку точек, убирает внутренние и лишние вершины."""
    for p in pts:
        bm.verts.new(p)
    res = bmesh.ops.convex_hull(bm, input=bm.verts[:], use_existing_faces=False)
    drop = [g for g in res['geom_interior'] + res['geom_unused']
            if isinstance(g, bmesh.types.BMVert)]
    if drop:
        bmesh.ops.delete(bm, geom=drop, context='VERTS')
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context='VERTS')


def volume_hull(pts):
    """Оболочка точек для показа и проверки: (вершины, треугольники, объём)."""
    n = len(pts)
    if n < 3:
        return [], [], 0.0
    if n == 3:
        return [p.copy() for p in pts], [(0, 1, 2)], 0.0
    bm = bmesh.new()
    try:
        _hull_into(bm, pts)
        bm.verts.ensure_lookup_table()
        bm.verts.index_update()
        verts = [v.co.copy() for v in bm.verts]
        tris = []
        for f in bm.faces:
            idx = [v.index for v in f.verts]
            for k in range(1, len(idx) - 1):
                tris.append((idx[0], idx[k], idx[k + 1]))
        volume = abs(bm.calc_volume()) if len(bm.faces) >= 4 else 0.0
        return verts, tris, volume
    except Exception:
        return [], [], 0.0
    finally:
        bm.free()


def add_prism(bm, bottom_pts, top_pts):
    """Добавляет в bmesh замкнутый резак между нижним и верхним контуром.
    Для параллельных контуров это призма, для контуров из камеры - усечённая пирамида."""
    n = len(bottom_pts)
    bottom = [p.copy() for p in bottom_pts]
    top = [p.copy() for p in top_pts]

    # обход контура должен быть против часовой стрелки, если смотреть в сторону выдавливания
    s = Vector((0.0, 0.0, 0.0))
    for i in range(n):
        s += bottom[i].cross(bottom[(i + 1) % n])
    cb = sum(bottom, Vector((0.0, 0.0, 0.0))) / n
    ct = sum(top, Vector((0.0, 0.0, 0.0))) / n
    if s.dot(ct - cb) < 0:
        bottom.reverse()
        top.reverse()

    bv = [bm.verts.new(p) for p in bottom]
    tv = [bm.verts.new(p) for p in top]
    bm.faces.new(tv)
    bm.faces.new(list(reversed(bv)))
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((bv[i], bv[j], tv[j], tv[i]))


def build_cutter_mesh(bottom_pts, top_pts):
    bm = bmesh.new()
    add_prism(bm, bottom_pts, top_pts)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    mesh = bpy.data.meshes.new("PolylineCarveCutter")
    bm.to_mesh(mesh)
    bm.free()
    return mesh


# ----------------------------------------------------------------------------
# Отрисовка в вьюпорте
# ----------------------------------------------------------------------------
def draw_callback(op):
    try:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')

        def draw(kind, coords, color, width=None, size=None):
            if not coords:
                return
            if width is not None:
                gpu.state.line_width_set(width)
            if size is not None:
                gpu.state.point_size_set(size)
            batch = batch_for_shader(shader, kind, {"pos": [tuple(c) for c in coords]})
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)

        hover = op.hover
        g = op.settings_grid()
        editing = op.editing
        show_hover = hover is not None and (not editing or op.drag_idx is not None)

        # маленький фрагмент сетки вокруг курсора
        if show_hover and op.scene.polyline_carve_snap.snap_grid:
            eu = op.plane_u * g
            ev = op.plane_v * g
            k = 4
            lines = []
            for i in range(-k, k + 1):
                lines += [hover + eu * i - ev * k, hover + eu * i + ev * k]
                lines += [hover + ev * i - eu * k, hover + ev * i + eu * k]
            draw('LINES', lines, (0.8, 0.8, 0.8, 0.25), width=1.0)

        # фигура, триангулированная сеткой (грани, по которым считаются нормали реза)
        spts, stris = op.shape_pts, op.shape_tris
        if stris:
            fill, wire = [], []
            for a, b, c in stris:
                fill += [spts[a], spts[b], spts[c]]
                wire += [spts[a], spts[b], spts[b], spts[c], spts[c], spts[a]]
            draw('TRIS', fill, (1.0, 0.6, 0.1, 0.15))
            draw('LINES', wire, (1.0, 0.8, 0.4, 0.6), width=1.0)

        pts = op.points
        if pts:
            if editing:
                if not op.volume:
                    draw('LINE_LOOP', pts, (1.0, 0.6, 0.1, 1.0), width=2.5)
            else:
                draw('LINE_STRIP', pts, (1.0, 0.6, 0.1, 1.0), width=2.5)
                if hover is not None:
                    draw('LINES', [pts[-1], hover], (1.0, 0.6, 0.1, 0.7), width=1.5)
                if len(pts) >= 3 and not op.volume:
                    draw('LINES', [pts[-1], pts[0]], (0.3, 1.0, 0.3, 0.35), width=1.5)

            # нормали реза (по граням фигуры) в режиме Face Normals
            if op.cut_mode == 'NORMAL' and op.shape_normals:
                nl = []
                for p, n in zip(spts, op.shape_normals):
                    nl += [p, p + n * (g * 0.6)]
                draw('LINES', nl, (0.3, 0.9, 1.0, 0.9), width=1.5)

            draw('POINTS', pts, (1.0, 1.0, 1.0, 1.0), size=8.0)
            if not editing:
                draw('POINTS', [pts[0]], (0.3, 1.0, 0.3, 1.0), size=11.0)
            else:
                if op.hover_idx is not None and op.hover_idx < len(pts):
                    draw('POINTS', [pts[op.hover_idx]], (1.0, 0.9, 0.1, 1.0), size=12.0)
                if op.drag_idx is not None and op.drag_idx < len(pts):
                    draw('POINTS', [pts[op.drag_idx]], (1.0, 0.4, 0.1, 1.0), size=13.0)
        if show_hover:
            if op.angle_guide is not None:
                draw('LINES', list(op.angle_guide), (0.6, 0.8, 1.0, 0.6), width=1.5)
            if op.snap_kind == 'EDGE' and op.snap_edge is not None:
                draw('LINES', list(op.snap_edge), (0.2, 1.0, 1.0, 1.0), width=3.0)
                draw('POINTS', [hover], (0.2, 1.0, 1.0, 1.0), size=12.0)
            elif op.snap_kind == 'VERT':
                draw('POINTS', [hover], (1.0, 0.2, 0.8, 1.0), size=14.0)
            else:
                draw('POINTS', [hover], (1.0, 0.9, 0.1, 1.0), size=10.0)

        gpu.state.blend_set('NONE')
        gpu.state.line_width_set(1.0)
        gpu.state.point_size_set(1.0)
    except Exception as e:  # отрисовка не должна ломать оператор
        print("PolylineCarve draw error:", e)


# ----------------------------------------------------------------------------
# Оператор
# ----------------------------------------------------------------------------
class OBJECT_OT_polyline_carve_snap(Operator):
    bl_idname = "object.polyline_carve_snap"
    bl_label = "Polyline Carve (Advanced)"
    bl_description = ("Рисует многоугольник по точкам (с привязкой к сетке) и вычитает его "
                      "из выделенных мешей (Boolean Difference)")
    bl_options = {'REGISTER', 'UNDO'}

    from_tool: BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == 'VIEW_3D' \
            and context.mode in {'OBJECT', 'EDIT_MESH'}

    # --- плоскость рисования ------------------------------------------------
    def settings_grid(self):
        return get_grid_size(self.space, self.scene)

    def set_plane(self, co, n, label):
        """Задаёт плоскость и базис сетки на ней.

        co - точка на плоскости, n - внешняя нормаль."""
        n = n.normalized()
        axis = None
        for i in range(3):
            if abs(n[i]) > 0.99999:  # плоскость параллельна мировой - сетка точно по мировой
                axis = i
                break
        if axis is not None:
            sign = 1.0 if n[axis] > 0 else -1.0
            n = Vector((0.0, 0.0, 0.0))
            n[axis] = sign
            pn = Vector((0.0, 0.0, 0.0))
            pn[axis] = 1.0
            iu, iv = [k for k in range(3) if k != axis]
            u = Vector((0.0, 0.0, 0.0))
            v = Vector((0.0, 0.0, 0.0))
            u[iu] = 1.0
            v[iv] = 1.0
            self.plane_origin = pn * co[axis]
            self.plane_no = pn
        else:
            if abs(n.z) < 0.999:
                u = Vector((0.0, 0.0, 1.0)).cross(n).normalized()
            else:
                u = (Vector((1.0, 0.0, 0.0)) - n * n.x).normalized()
            v = n.cross(u)
            self.plane_origin = n * co.dot(n)
            self.plane_no = n
        self.plane_u, self.plane_v = u, v
        self.plane_co = co.copy()
        self.surface_no = n
        self.plane_label = label

    def resolve_plane(self, context):
        """Плоскость для режима Camera: через 3D-курсор, перпендикулярно оси."""
        if self.axis_mode == 'AUTO':
            vd = self.rv3d.view_rotation @ Vector((0.0, 0.0, -1.0))
            i = max(range(3), key=lambda k: abs(vd[k]))
            axis = i if abs(vd[i]) > 0.999 else 2
            label = f"{'XYZ'[axis]} (auto)"
        else:
            axis = {'X': 0, 'Y': 1, 'Z': 2}[self.axis_mode]
            label = 'XYZ'[axis]
        n = Vector((0.0, 0.0, 0.0))
        n[axis] = 1.0
        self.set_plane(self.scene.cursor.location, n, label)

    def snap(self, p):
        g = self.settings_grid()
        d = p - self.plane_origin
        a = round(d.dot(self.plane_u) / g) * g
        b = round(d.dot(self.plane_v) / g) * g
        return self.plane_origin + self.plane_u * a + self.plane_v * b

    def mouse_to_plane(self, coord):
        origin = view3d_utils.region_2d_to_origin_3d(self.region, self.rv3d, coord)
        vec = view3d_utils.region_2d_to_vector_3d(self.region, self.rv3d, coord)
        n = self.plane_no
        if abs(vec.dot(n)) < 0.02:
            return None
        hit = geometry.intersect_line_plane(origin, origin + vec * 1000.0,
                                            self.plane_co, n, False)
        if hit is None:
            return None
        if self.rv3d.is_perspective and (hit - origin).dot(vec) <= 0:
            return None
        return hit

    # --- лучи по целевым мешам ----------------------------------------------
    def cast_targets(self, context, origin, direction):
        """Ближайшее попадание луча в целевые меши: (расстояние, точка, нормаль) или None."""
        direction = direction.normalized()
        depsgraph = context.evaluated_depsgraph_get()
        best = None
        for obj in self.targets:
            mw = obj.matrix_world
            inv = mw.inverted_safe()
            try:
                ok, loc, nor, _idx = obj.ray_cast(
                    inv @ origin, (inv.to_3x3() @ direction).normalized(), depsgraph=depsgraph)
            except Exception:
                continue
            if not ok:
                continue
            w = mw @ loc
            dist = (w - origin).dot(direction)
            if best is None or dist < best[0]:
                nw = (mw.to_3x3().inverted_safe().transposed() @ nor).normalized()
                if nw.dot(direction) > 0:  # попали в обратную сторону грани
                    nw = -nw
                best = (dist, w, nw)
        return best

    def coord_of(self, event):
        return (event.mouse_region_x, event.mouse_region_y)

    def ray_at(self, coord):
        origin = view3d_utils.region_2d_to_origin_3d(self.region, self.rv3d, coord)
        vec = view3d_utils.region_2d_to_vector_3d(self.region, self.rv3d, coord)
        return origin, vec.normalized()

    def surface_hits(self, context, coord, limit):
        """Поверхности вдоль луча из точки экрана: [(точка, нормаль)] от ближней к дальней."""
        origin, vec = self.ray_at(coord)
        step = max(self.settings_grid() * 1e-3, 1e-4)
        hits = []
        o = origin
        for _ in range(limit):
            res = self.cast_targets(context, o, vec)
            if res is None:
                break
            hits.append((res[1], res[2]))
            o = res[1] + vec * step
        return hits

    def pick_surface(self, context, coord):
        """Поверхность под курсором. При hit_index > 0 (клавиша Tab) - следующие по глубине."""
        if self.hit_index <= 0:
            hits = self.surface_hits(context, coord, 1)
            self.hit_count = 0
            return hits[0] if hits else None
        hits = self.surface_hits(context, coord, 8)
        self.hit_count = len(hits)
        if not hits:
            return None
        return hits[self.hit_index % len(hits)]

    def reproject(self, context, p, n):
        """Опускает привязанную к сетке точку вдоль нормали на реальную поверхность."""
        up = max(self.settings_grid() * 2.0, 1e-3)
        res = self.cast_targets(context, p + n * up, -n)
        if res is None:
            return p, n
        return res[1], res[2]

    def update_hover(self, context, event):
        self.resolve_hover(context, self.coord_of(event))

    # --- привязка: угол, вершины, рёбра -------------------------------------
    def angle_reference(self):
        """Предыдущая точка фигуры - от неё отсчитывается угол."""
        n = len(self.points)
        if self.drag_idx is not None and n >= 2:
            return self.points[(self.drag_idx - 1) % n]
        if not self.editing and n >= 1:
            return self.points[-1]
        return None

    def angle_dir(self, raw, prev):
        """Единичное направление от prev к raw в плоскости рисования, округлённое до шага угла."""
        step = self.scene.polyline_carve_snap.angle_step
        n = self.plane_no
        d = raw - prev
        d = d - n * d.dot(n)
        length = d.length
        if length < 1e-9:
            return None, 0.0
        ang = math.atan2(d.dot(self.plane_v), d.dot(self.plane_u))
        a = round(ang / step) * step
        return self.plane_u * math.cos(a) + self.plane_v * math.sin(a), length

    def angle_snap(self, raw, prev):
        dirv, length = self.angle_dir(raw, prev)
        if dirv is None:
            return raw
        n = self.plane_no
        p = prev + dirv * length + n * (raw - prev).dot(n)
        self.angle_guide = (prev, prev + dirv * max(length * 1.5, self.settings_grid() * 3.0))
        return p

    def ensure_snap_cache(self):
        """Мировые координаты вершин и индексы рёбер всех целевых мешей."""
        if self.snap_V is not None:
            return
        verts, edges, offset = [], [], 0
        for obj in self.targets:
            try:
                if obj.mode == 'EDIT':
                    obj.update_from_editmode()
                me = obj.data
                nv, ne = len(me.vertices), len(me.edges)
                if nv == 0:
                    continue
                co = np.empty(nv * 3, dtype=np.float32)
                me.vertices.foreach_get('co', co)
                mw = np.array(obj.matrix_world, dtype=np.float64)
                verts.append(co.reshape(nv, 3).astype(np.float64) @ mw[:3, :3].T + mw[:3, 3])
                ed = np.empty(ne * 2, dtype=np.int32)
                me.edges.foreach_get('vertices', ed)
                edges.append(ed.reshape(ne, 2).astype(np.int64) + offset)
                offset += nv
            except Exception:
                continue
        self.snap_V = np.vstack(verts) if verts else np.zeros((0, 3))
        self.snap_E = np.vstack(edges) if edges else np.zeros((0, 2), dtype=np.int64)

    def project_np(self, pts):
        """Экранные координаты (пиксели региона) для массива мировых точек + маска видимости."""
        m = np.array(self.rv3d.perspective_matrix, dtype=np.float64)
        w = pts @ m[3, :3] + m[3, 3]
        ok = w > 1e-6
        ws = np.where(ok, w, 1.0)
        x = (pts @ m[0, :3] + m[0, 3]) / ws
        y = (pts @ m[1, :3] + m[1, 3]) / ws
        sx = (x + 1.0) * 0.5 * self.region.width
        sy = (y + 1.0) * 0.5 * self.region.height
        return np.stack([sx, sy], axis=1), ok

    def visible_at(self, context, p):
        """Не закрыта ли точка поверхностью. Возвращает (видима, нормаль поверхности или None)."""
        sc = self.screen(p)
        if sc is None:
            return False, None
        origin, vec = self.ray_at((sc.x, sc.y))
        dist_p = (p - origin).dot(vec)
        tol = 5e-4 * abs(dist_p) + 1e-4
        res = self.cast_targets(context, origin, vec)
        if res is None:
            return True, None
        if res[0] < dist_p - tol:
            return False, None
        return True, (res[2] if abs(res[0] - dist_p) <= tol * 4 else None)

    def edge_grid_t(self, p0, d, t_c):
        """Параметр на ребре, где ось ребра пересекает линию сетки (ближайший к t_c)."""
        g = self.settings_grid()
        best = None
        for axis in (self.plane_u, self.plane_v):
            a0 = (p0 - self.plane_origin).dot(axis)
            da = d.dot(axis)
            if abs(da) < 1e-9:
                continue
            k0 = round((a0 + da * t_c) / g)
            for k in (k0 - 1, k0, k0 + 1):
                t = (k * g - a0) / da
                if 0.0 <= t <= 1.0 and (best is None or abs(t - t_c) < best[0]):
                    best = (abs(t - t_c), t)
        return best[1] if best else None

    def edge_point(self, p0, p1, coord, prev, s):
        """Точка на оси ребра под курсором; с сеткой - пересечение сетки и оси ребра,
        с углом - пересечение линии угла и оси ребра."""
        d = p1 - p0
        l2 = d.length_squared
        if l2 < 1e-18:
            return None
        origin, vec = self.ray_at(coord)
        r = geometry.intersect_line_line(p0, p1, origin, origin + vec)
        if r is None:
            return None
        t_c = max(0.0, min(1.0, (r[0] - p0).dot(d) / l2))
        t = t_c
        if s.snap_grid:
            tg = self.edge_grid_t(p0, d, t_c)
            if tg is not None:
                t = tg
        elif s.snap_angle and prev is not None:
            dirv, _length = self.angle_dir(p0 + d * t_c, prev)
            if dirv is not None:
                r2 = geometry.intersect_line_line(p0, p1, prev, prev + dirv)
                if r2 is not None:
                    t2 = (r2[0] - p0).dot(d) / l2
                    if 0.0 <= t2 <= 1.0:
                        t = t2
        return p0 + d * t

    def find_snap_target(self, context, coord, prev):
        """Вершина или ребро рядом с курсором (вершины в приоритете). None, если нет."""
        s = self.scene.polyline_carve_snap
        if not (s.snap_vertices or s.snap_edges):
            return None
        self.ensure_snap_cache()
        if len(self.snap_V) == 0:
            return None
        try:
            thr = 14.0 * bpy.context.preferences.system.ui_scale
        except Exception:
            thr = 14.0
        scr, ok = self.project_np(self.snap_V)
        m = np.array(coord, dtype=np.float64)

        if s.snap_vertices:
            d = np.where(ok, np.hypot(scr[:, 0] - m[0], scr[:, 1] - m[1]), np.inf)
            idx = np.nonzero(d < thr)[0]
            if len(idx):
                for i in idx[np.argsort(d[idx])][:4]:
                    p = Vector(self.snap_V[i].tolist())
                    vis, n = self.visible_at(context, p)
                    if vis:
                        return {'kind': 'VERT', 'p': p, 'n': n, 'edge': None}

        if s.snap_edges and len(self.snap_E):
            e = self.snap_E
            a, b = scr[e[:, 0]], scr[e[:, 1]]
            ab = b - a
            ll = (ab * ab).sum(1)
            t = np.where(ll > 1e-9, ((m - a) * ab).sum(1) / np.where(ll > 1e-9, ll, 1.0), 0.0)
            c = a + ab * np.clip(t, 0.0, 1.0)[:, None]
            d = np.hypot(c[:, 0] - m[0], c[:, 1] - m[1])
            d = np.where(ok[e[:, 0]] & ok[e[:, 1]], d, np.inf)
            idx = np.nonzero(d < thr)[0]
            if len(idx):
                for i in idx[np.argsort(d[idx])][:4]:
                    p0 = Vector(self.snap_V[e[i, 0]].tolist())
                    p1 = Vector(self.snap_V[e[i, 1]].tolist())
                    p = self.edge_point(p0, p1, coord, prev, s)
                    if p is None:
                        continue
                    vis, n = self.visible_at(context, p)
                    if vis:
                        return {'kind': 'EDGE', 'p': p, 'n': n, 'edge': (p0, p1)}
        return None

    def resolve_hover(self, context, coord):
        """Позиция точки под экранной координатой и её нормаль.

        Приоритет привязки: вершины -> рёбра -> сетка или угол."""
        s = self.scene.polyline_carve_snap
        self.hover_normal = None
        self.snap_kind = None
        self.snap_edge = None
        self.angle_guide = None

        # 1. сырая позиция: поверхность (Face Normals) или плоскость рисования (Camera)
        raw, raw_n, on_surface = None, None, False
        if self.surface_mode():
            hit = self.pick_surface(context, coord)
            if hit is not None:
                raw, raw_n = hit
                on_surface = True
                self.set_plane(raw, raw_n, "face")  # плоскость и сетка - по грани под курсором
            elif self.points:  # курсор вне объекта - продолжаем плоскость точки/соседа
                ref = self.drag_idx if self.drag_idx is not None else len(self.points) - 1
                self.set_plane(self.points[ref], self.point_normals[ref], "face")
                raw = self.mouse_to_plane(coord)
                raw_n = self.surface_no.copy()
        else:
            if not self.points and self.axis_mode == 'AUTO':
                self.resolve_plane(None)
            raw = self.mouse_to_plane(coord)

        prev = self.angle_reference()

        # 2. вершины и рёбра - в первую очередь
        target = self.find_snap_target(context, coord, prev)
        if target is not None:
            self.hover = target['p']
            self.snap_kind = target['kind']
            self.snap_edge = target['edge']
            self.hover_normal = target['n'] if target['n'] is not None else (
                raw_n.copy() if raw_n is not None else None)
            if self.surface_mode() and self.hover_normal is not None:
                self.set_plane(self.hover, self.hover_normal, "face")
            return

        if raw is None:
            self.hover = None
            return

        # 3. сетка либо угол (взаимоисключающие), либо без привязки
        p = raw
        if s.snap_grid:
            p = self.snap(raw)
        elif s.snap_angle and prev is not None:
            p = self.angle_snap(raw, prev)
        n = raw_n
        if on_surface and (s.snap_grid or (s.snap_angle and prev is not None)):
            p, n = self.reproject(context, p, self.surface_no.copy())  # обратно на поверхность
        self.hover = p
        self.hover_normal = n.copy() if n is not None else None

    def surface_mode(self):
        """Точки ставятся на поверхности объектов (Face Normals и Volume)."""
        return self.volume or self.cut_mode == 'NORMAL'

    def min_points(self):
        return 4 if self.volume else 3

    def refresh_shape(self):
        """Пересчитывает триангуляцию фигуры (с учётом точки под курсором при рисовании)
        и нормали реза по её граням."""
        pts = list(self.points)
        hints = list(self.point_normals)
        h = self.hover
        if (not self.editing and h is not None and len(pts) >= (1 if self.volume else 2)
                and not any((h - q).length < 1e-9 for q in pts)
                and not self.would_intersect(h)):
            pts.append(h.copy())
            hints.append(self.hover_point_normal())
        if self.volume:
            # показываем выпуклую оболочку точек - это и есть вырезаемый объём
            self.shape_pts, self.shape_tris, _vol = volume_hull(pts)
            self.shape_normals = []
            return
        tris = triangulate_shape(pts) if len(pts) >= 3 else []
        self.shape_pts, self.shape_tris = pts, tris
        if tris and self.cut_mode == 'NORMAL':
            self.shape_normals = shape_vertex_normals(pts, tris, hints)
        else:
            self.shape_normals = []

    def in_nav_gizmo(self, event):
        """Курсор над навигационным гизмо (справа сверху) - его клики пропускаем в Blender."""
        try:
            prefs = bpy.context.preferences
            gizmo = getattr(prefs.view, "gizmo_size_navigate_v3d", 80)
            ui = prefs.system.ui_scale
            width, height = (gizmo + 60) * ui, (gizmo + 260) * ui
        except Exception:
            width, height = 140.0, 340.0
        # справа сверху: оси, кнопки зума/панорамирования/камеры/перспективы (под заголовками)
        return (event.mouse_region_x > self.region.width - width and
                event.mouse_region_y > self.region.height - height)

    def in_region(self, event):
        return (0 <= event.mouse_region_x < self.region.width and
                0 <= event.mouse_region_y < self.region.height)

    # --- проверки фигуры -----------------------------------------------------
    def check_basis(self):
        """Базис проверки самопересечений. В Face Normals - плоскость средней нормали фигуры
        (как при триангуляции), поэтому результат не зависит от положения камеры."""
        if self.cut_mode == 'NORMAL':
            n = sum(self.point_normals, Vector((0.0, 0.0, 0.0)))
            if n.length < 1e-9:
                n = self.surface_no.copy()
            n = n.normalized()
            ref = Vector((0.0, 0.0, 1.0)) if abs(n.z) < 0.9 else Vector((1.0, 0.0, 0.0))
            u = n.cross(ref).normalized()
            return u, n.cross(u)
        return self.plane_u, self.plane_v

    def p2(self, p):
        u, v = self._basis
        return (p.dot(u), p.dot(v))

    def would_intersect(self, p):
        if self.volume:
            return False
        self._basis = self.check_basis()
        pts = self.points
        if len(pts) < 2:
            return False
        a, b = self.p2(pts[-1]), self.p2(p)
        # возврат назад по предыдущему отрезку
        prev = self.p2(pts[-2])
        if abs(_ccw(prev, a, b)) <= EPS:
            if (a[0] - prev[0]) * (b[0] - a[0]) + (a[1] - prev[1]) * (b[1] - a[1]) < 0:
                return True
        for i in range(len(pts) - 2):  # последний отрезок смежный - пропускаем
            if segments_intersect(a, b, self.p2(pts[i]), self.p2(pts[i + 1])):
                return True
        return False

    def polygon_intersects(self, pts):
        """Самопересекается ли замкнутая фигура."""
        if self.volume:
            return False
        self._basis = self.check_basis()
        n = len(pts)
        q = [self.p2(p) for p in pts]
        for i in range(n):
            a, b = q[i], q[(i + 1) % n]
            for j in range(i + 1, n):
                if j == i + 1 or (i == 0 and j == n - 1):
                    continue  # смежные рёбра
                if segments_intersect(a, b, q[j], q[(j + 1) % n]):
                    return True
        return False

    # --- выбор точек и рёбер мышью ------------------------------------------
    def screen(self, p):
        return view3d_utils.location_3d_to_region_2d(self.region, self.rv3d, p)

    def mouse_vec(self, event):
        return Vector((event.mouse_region_x, event.mouse_region_y))

    def pick_point(self, event, radius=CLOSE_PIXELS):
        m = self.mouse_vec(event)
        best = None
        for i, p in enumerate(self.points):
            sc = self.screen(p)
            if sc is None:
                continue
            d = (sc - m).length
            if d < radius and (best is None or d < best[0]):
                best = (d, i)
        return best[1] if best else None

    def pick_edge(self, event, radius=8.0):
        m = self.mouse_vec(event)
        n = len(self.points)
        best = None
        for i in range(n):
            a = self.screen(self.points[i])
            b = self.screen(self.points[(i + 1) % n])
            if a is None or b is None:
                continue
            ab = b - a
            ll = ab.length_squared
            t = 0.0 if ll < 1e-9 else max(0.0, min(1.0, (m - a).dot(ab) / ll))
            d = (m - (a + ab * t)).length
            if d < radius and (best is None or d < best[0]):
                best = (d, i)
        return best[1] if best else None

    def near_first(self, event):
        if not self.points:
            return False
        sc = self.screen(self.points[0])
        if sc is None:
            return False
        return (sc - self.mouse_vec(event)).length < CLOSE_PIXELS

    # --- статус и очистка ---------------------------------------------------
    def snap_summary(self):
        s = self.scene.polyline_carve_snap
        parts = []
        if s.snap_vertices:
            parts.append("verts")
        if s.snap_edges:
            parts.append("edges")
        if s.snap_grid:
            parts.append(f"grid {self.settings_grid():.4g}")
        if s.snap_angle:
            parts.append(f"angle {math.degrees(s.angle_step):.4g}")
        return "+".join(parts) if parts else "off"

    def update_status(self, context):
        if self.volume:
            mode = "Volume (convex hull of points)"
        elif self.cut_mode == 'NORMAL':
            mode = "Face Normals (normal per point)"
        else:
            mode = f"Camera (plane normal: {self.plane_label})"
        if self.editing:
            txt = ("Polyline Carve | EDIT (camera is free: MMB/wheel/numpad/Shift+` fly/gizmo) | "
                   "drag points (LMB) | Tab: next surface for point | click edge: add point | "
                   "Backspace/Del on point: remove | Enter: CUT | Esc/RMB: cancel | "
                   f"Snap: {self.snap_summary()} | Mode: {mode} | Points: {len(self.points)}")
        else:
            txt = ("Polyline Carve | LMB: point | Tab: next surface under cursor | "
                   "Enter or click first point: edit shape | "
                   "Backspace: undo point | V: volume/shape | N: camera/normals | X/Y/Z/A: plane | "
                   f"Snap: {self.snap_summary()} | Mode: {mode} | "
                   f"Points: {len(self.points)} | Esc/RMB: cancel")
        context.workspace.status_text_set(txt)

    def cleanup(self, context):
        if self._handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None
        context.workspace.status_text_set(None)
        if context.area:
            context.area.tag_redraw()

    # --- жизненный цикл -----------------------------------------------------
    def invoke(self, context, event):
        if context.area.type != 'VIEW_3D' or context.region.type != 'WINDOW' \
                or context.region_data is None:
            self.report({'WARNING'}, "Запустите оператор из 3D Viewport")
            return {'CANCELLED'}
        if context.mode == 'EDIT_MESH':
            targets = [o for o in context.objects_in_mode_unique_data if o.type == 'MESH']
            msg = "Нет мешей в Edit Mode"
        else:
            targets = [o for o in context.selected_objects if o.type == 'MESH']
            msg = "Выделите хотя бы один меш-объект"
        if not targets:
            self.report({'WARNING'}, msg)
            return {'CANCELLED'}

        # второй запуск (например, клик по гизмо, прошедший в keymap инструмента) не нужен
        try:
            if any(o.bl_idname == self.bl_idname for o in context.window.modal_operators):
                return {'CANCELLED'}
        except Exception:
            pass

        self.targets = set(targets)
        self.scene = context.scene
        self.space = context.space_data
        self.region = context.region
        self.rv3d = context.region_data
        self.cut_mode = self.scene.polyline_carve_snap.cut_mode
        self.volume = self.scene.polyline_carve_snap.cut_method == 'VOLUME'
        self.hit_index = 0
        self.hit_count = 0
        self.cycle_anchor = None
        self.snap_V = None
        self.snap_E = None
        self.snap_kind = None
        self.snap_edge = None
        self.angle_guide = None
        self.points = []
        self.point_normals = []
        self.shape_pts = []
        self.shape_tris = []
        self.shape_normals = []
        self.hover = None
        self.hover_normal = None
        self.hover_idx = None
        self.drag_idx = None
        self.editing = False
        self.axis_mode = 'AUTO'
        self._handle = None
        self.resolve_plane(context)

        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            draw_callback, (self,), 'WINDOW', 'POST_VIEW')
        context.window_manager.modal_handler_add(self)

        self.update_hover(context, event)
        if self.from_tool and self.in_region(event):
            self.add_point_at_hover()
        self.update_status(context)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def hover_point_normal(self):
        n = self.hover_normal if self.hover_normal is not None else self.surface_no
        return n.copy()

    def add_point_at_hover(self):
        p = self.hover
        if p is None:
            return
        if any((p - q).length < 1e-9 for q in self.points):
            self.report({'INFO'}, "Такая точка уже есть")
            return
        if self.would_intersect(p):
            self.report({'WARNING'}, "Сегмент не должен пересекать фигуру")
            return
        self.points.append(p.copy())
        self.point_normals.append(self.hover_point_normal())

    def begin_edit(self, context):
        """Первое нажатие Enter: фигура замыкается и переходит в режим редактирования точек."""
        if len(self.points) < self.min_points():
            self.report({'WARNING'}, f"Нужно минимум {self.min_points()} точки")
            return
        if self.volume and volume_hull(self.points)[2] < 1e-12:
            self.report({'WARNING'}, "Точки лежат в одной плоскости - объёма нет")
            return
        if self.polygon_intersects(self.points):
            self.report({'WARNING'}, "Замыкающий сегмент пересекает фигуру")
            return
        self.editing = True
        self.drag_idx = None
        self.hover_idx = None
        self.update_status(context)

    def move_point_to_hover(self, i):
        """Переносит точку i в позицию hover (на поверхность под курсором). Возвращает успех."""
        if i is None or self.hover is None or i >= len(self.points):
            return False
        if any(j != i and (self.hover - q).length < 1e-9 for j, q in enumerate(self.points)):
            return False
        old_p, old_n = self.points[i], self.point_normals[i]
        self.points[i] = self.hover.copy()
        self.point_normals[i] = self.hover_point_normal()
        if self.polygon_intersects(self.points):  # самопересечение - не двигаем
            self.points[i], self.point_normals[i] = old_p, old_n
            return False
        return True

    def drag_to_hover(self):
        self.move_point_to_hover(self.drag_idx)

    def cycle_surface(self, context, event):
        """Tab: следующая поверхность вдоль луча (грань, на которой лежит точка)."""
        i = self.drag_idx if self.drag_idx is not None else (self.hover_idx if self.editing else None)
        coord = self.coord_of(event)
        if i is not None and i < len(self.points):
            sc = self.screen(self.points[i])
            if sc is not None:
                coord = (sc.x, sc.y)  # луч через саму точку, а не через курсор
        self.hit_index += 1
        self.cycle_anchor = Vector(self.coord_of(event))
        self.resolve_hover(context, coord)
        if self.hit_count > 0:
            self.report({'INFO'}, f"Поверхность {self.hit_index % self.hit_count + 1} из {self.hit_count}")
        if self.editing and i is not None:
            if not self.move_point_to_hover(i):
                self.report({'WARNING'}, "Нельзя перенести точку на эту поверхность")

    def edit_press(self, context, event):
        i = self.pick_point(event)
        if i is not None:
            self.drag_idx = i
            return
        e = self.pick_edge(event)
        if e is not None and self.hover is not None:
            if any((self.hover - q).length < 1e-9 for q in self.points):
                return
            self.points.insert(e + 1, self.hover.copy())
            self.point_normals.insert(e + 1, self.hover_point_normal())
            if self.polygon_intersects(self.points):
                del self.points[e + 1]
                del self.point_normals[e + 1]
                self.report({'WARNING'}, "Точка пересекает фигуру")
            else:
                self.drag_idx = e + 1

    def modal(self, context, event):
        try:
            result = self.handle_event(context, event)
        except Exception:
            self.cleanup(context)
            raise
        if 'RUNNING_MODAL' in result and event.type not in NAV_EVENTS:
            self.refresh_shape()
        return result

    def handle_event(self, context, event):
        if context.area is None:
            self.cleanup(context)
            return {'CANCELLED'}
        context.area.tag_redraw()

        # Камера свободно двигается и при рисовании, и при редактировании фигуры
        # (включая режим полёта Shift+` / Shift+F, который запускается штатными средствами Blender)
        if event.type in NAV_EVENTS or event.type in {'HOME', 'ACCENT_GRAVE'} or \
                (event.type == 'F' and event.shift) or \
                (event.type.startswith('NUMPAD_') and event.type != 'NUMPAD_ENTER'):
            return {'PASS_THROUGH'}
        if event.type == 'LEFTMOUSE' and (event.alt or self.in_nav_gizmo(event)):
            return {'PASS_THROUGH'}  # Alt+ЛКМ (эмуляция СКМ) и навигационное гизмо

        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            if self.hit_index and self.cycle_anchor is not None and \
                    (Vector(self.coord_of(event)) - self.cycle_anchor).length > 6.0:
                self.hit_index = 0  # курсор ушёл - снова ближайшая поверхность
                self.cycle_anchor = None
            self.update_hover(context, event)
            if self.editing:
                if self.drag_idx is not None:
                    self.drag_to_hover()
                else:
                    self.hover_idx = self.pick_point(event)
            if self.in_nav_gizmo(event):
                return {'RUNNING_MODAL', 'PASS_THROUGH'}  # подсветка гизмо
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                if not self.in_region(event):
                    return {'PASS_THROUGH'}
                self.update_hover(context, event)
                if self.editing:
                    self.edit_press(context, event)
                elif len(self.points) >= self.min_points() and self.near_first(event):
                    self.begin_edit(context)
                else:
                    self.add_point_at_hover()
                    self.update_status(context)
                self.hit_index = 0
                self.cycle_anchor = None
                return {'RUNNING_MODAL'}
            if event.value == 'RELEASE' and self.drag_idx is not None:
                self.drag_idx = None
                return {'RUNNING_MODAL'}

        if event.value == 'PRESS':
            if event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'}:
                if not self.editing:
                    self.begin_edit(context)
                    return {'RUNNING_MODAL'}
                return self.finish(context)
            if event.type in {'ESC', 'RIGHTMOUSE'}:
                self.cleanup(context)
                return {'CANCELLED'}
            if event.type in {'BACK_SPACE', 'DEL'} or (event.type == 'Z' and event.ctrl):
                if self.editing:
                    i = self.hover_idx
                    if i is not None and len(self.points) > self.min_points():
                        old = (self.points.pop(i), self.point_normals.pop(i))
                        if self.polygon_intersects(self.points):
                            self.points.insert(i, old[0])
                            self.point_normals.insert(i, old[1])
                            self.report({'WARNING'}, "Удаление создаст пересечение")
                        else:
                            self.hover_idx = None
                    elif i is not None:
                        self.report({'INFO'}, f"Нужно минимум {self.min_points()} точки")
                elif self.points:
                    self.points.pop()
                    self.point_normals.pop()
                self.update_status(context)
                return {'RUNNING_MODAL'}
            if event.type == 'TAB':
                if self.surface_mode():
                    self.cycle_surface(context, event)
                else:
                    self.report({'INFO'}, "Tab работает в режимах Face Normals и Volume")
                return {'RUNNING_MODAL'}
            if self.editing:
                return {'RUNNING_MODAL'}
            if event.type == 'V':
                if self.points:
                    self.report({'INFO'}, "Способ выреза можно сменить, пока нет точек")
                else:
                    self.volume = not self.volume
                    self.scene.polyline_carve_snap.cut_method = 'VOLUME' if self.volume else 'EXTRUDE'
                    if not self.surface_mode():
                        self.resolve_plane(context)
                    self.update_hover(context, event)
                    self.update_status(context)
                return {'RUNNING_MODAL'}
            if event.type == 'N':
                if self.volume:
                    self.report({'INFO'}, "В режиме Volume направление реза не используется")
                elif self.points:
                    self.report({'INFO'}, "Режим можно сменить, пока нет точек")
                else:
                    self.cut_mode = 'NORMAL' if self.cut_mode == 'VIEW' else 'VIEW'
                    self.scene.polyline_carve_snap.cut_mode = self.cut_mode
                    if self.cut_mode == 'VIEW':
                        self.resolve_plane(context)
                    self.update_hover(context, event)
                    self.update_status(context)
                return {'RUNNING_MODAL'}
            if event.type in {'X', 'Y', 'Z', 'A'}:
                if self.surface_mode():
                    self.report({'INFO'}, "Плоскость берётся из грани под курсором")
                elif self.points:
                    self.report({'INFO'}, "Смена плоскости возможна только пока нет точек")
                else:
                    self.axis_mode = 'AUTO' if event.type == 'A' else event.type
                    self.resolve_plane(context)
                    self.update_hover(context, event)
                    self.update_status(context)
                return {'RUNNING_MODAL'}

        return {'RUNNING_MODAL'}

    def finish(self, context):
        if len(self.points) < self.min_points() or self.polygon_intersects(self.points) or \
                (self.volume and volume_hull(self.points)[2] < 1e-12):
            self.report({'WARNING'}, "Фигура некорректна")
            return {'RUNNING_MODAL'}
        self.cleanup(context)
        return {'FINISHED'} if self.carve(context) else {'CANCELLED'}

    # --- булева операция ----------------------------------------------------
    def carve(self, context):
        s = context.scene.polyline_carve_snap
        view_layer = context.view_layer
        edit = context.mode == 'EDIT_MESH'

        if edit:
            targets = [o for o in context.objects_in_mode_unique_data if o.type == 'MESH']
        else:
            targets = [o for o in context.selected_objects if o.type == 'MESH']
        if not targets:
            self.report({'WARNING'}, "Нет подходящих мешей")
            return False

        active_before = view_layer.objects.active
        selected_before = [o for o in view_layer.objects if o.select_get()]

        if edit:
            # Boolean-модификатор работает с мешем объекта, а не с edit-mesh,
            # поэтому на время операции выходим из Edit Mode.
            bpy.ops.object.mode_set(mode='OBJECT')

        try:
            self.targets = set(targets)
            done = self.carve_objects(context, s, targets)
        finally:
            if edit:
                for o in view_layer.objects:
                    try:
                        o.select_set(False)
                    except Exception:
                        pass
                for o in targets:
                    o.select_set(True)
                view_layer.objects.active = active_before if active_before in targets else targets[0]
                bpy.ops.object.mode_set(mode='EDIT')
                for o in selected_before:
                    if o not in targets:
                        try:
                            o.select_set(True)
                        except Exception:
                            pass

        if done:
            self.report({'INFO'}, f"Вырезано из объектов: {done}")
        return done > 0

    # --- Face Normals: резак по граням самой фигуры --------------------------
    def build_surface_cutter(self, context, s, targets, overshoot):
        """Фигура триангулируется; нормаль реза в каждой точке - среднее нормалей граней фигуры.
        Из-за этого на сгибах (выпуклых и вогнутых) пол и стенки выреза ломаются по граням фигуры,
        а не остаются одной плоскостью на всю фигуру."""
        pts = [p.copy() for p in self.points]
        hints = [n.copy() for n in self.point_normals]
        cnt = len(pts)
        tris = triangulate_shape(pts)
        if not tris:
            return None
        nrm = shape_vertex_normals(pts, tris, hints)
        navg = sum(nrm, Vector((0.0, 0.0, 0.0)))
        navg = navg.normalized() if navg.length > 1e-9 else self.surface_no.copy()

        corners = [o.matrix_world @ Vector(c) for o in targets for c in o.bound_box]
        lo = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
        hi = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
        big = (hi - lo).length + 1.0
        cmin = min(c.dot(navg) for c in corners)

        # Крышка - это хорды между точками; на выпуклой поверхности она уходит под неё.
        # Поднимаем верхний контур настолько, чтобы вся крышка была над поверхностью.
        samples = [(pts[a] + pts[b] + pts[c]) / 3.0 for a, b, c in tris]
        samples += [(pts[i] + pts[(i + 1) % cnt]) * 0.5 for i in range(cnt)]
        lift = 0.0
        for c in samples:
            res = self.cast_targets(context, c + navg * big, -navg)
            if res is not None:
                lift = max(lift, big - res[0])

        back = max(s.backward_depth, overshoot)
        out = max(s.depth_margin, back) if s.auto_depth else back
        top = [p + n * (out + lift) for p, n in zip(pts, nrm)]
        if s.auto_depth:
            bottom = [p - navg * (p.dot(navg) - cmin + s.depth_margin) for p in pts]
        else:
            bottom = [p - n * s.depth for p, n in zip(pts, nrm)]

        bm = bmesh.new()
        tv = [bm.verts.new(p) for p in top]
        bv_ = [bm.verts.new(p) for p in bottom]

        def face(*verts):
            try:
                bm.faces.new(verts)
            except ValueError:
                pass  # дубликат или вырожденная грань

        for a, b, c in tris:
            face(tv[a], tv[b], tv[c])      # верх - те же грани, что и у фигуры
            face(bv_[a], bv_[c], bv_[b])   # низ - те же грани, с обратной ориентацией
        for i in range(cnt):               # стенки по контуру фигуры
            j = (i + 1) % cnt
            face(bv_[i], bv_[j], tv[j])
            face(bv_[i], tv[j], tv[i])

        if not bm.faces:
            bm.free()
            return None
        # согласованная ориентация граней и нормали наружу
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        mesh = bpy.data.meshes.new("PolylineCarveCutter")
        bm.to_mesh(mesh)
        bm.free()
        return mesh

    # --- Volume: выпуклая оболочка точек ------------------------------------
    def build_volume_cutter(self):
        bm = bmesh.new()
        try:
            _hull_into(bm, self.points)
            if len(bm.faces) < 4 or abs(bm.calc_volume()) < 1e-12:
                return None
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
            mesh = bpy.data.meshes.new("PolylineCarveCutter")
            bm.to_mesh(mesh)
            return mesh
        except Exception:
            return None
        finally:
            bm.free()

    # --- Camera: глубина от видимой поверхности ------------------------------
    def front_depth(self, context, targets, depth_of, ray_for):
        """Глубины ближней и дальней видимой поверхности под фигурой: (near, far) или None.
        Лучи идут через точки фигуры, середины рёбер и центр."""
        pts = list(self.points)
        n = len(pts)
        if n == 0:
            return None
        centroid = sum(pts, Vector((0.0, 0.0, 0.0))) / n
        samples = pts + [centroid] + [(pts[i] + pts[(i + 1) % n]) * 0.5 for i in range(n)]
        near_d, far_d = None, None
        for pt in samples:
            try:
                origin, direction = ray_for(pt)
                res = self.cast_targets(context, origin, direction)
            except Exception:
                continue
            if res is None:
                continue
            dep = depth_of(res[1])
            near_d = dep if near_d is None else min(near_d, dep)
            far_d = dep if far_d is None else max(far_d, dep)
        return None if near_d is None else (near_d, far_d)

    # --- применение резака ---------------------------------------------------
    def apply_cutter(self, context, targets, mesh):
        cutter = bpy.data.objects.new("PolylineCarveCutter", mesh)
        context.scene.collection.objects.link(cutter)
        done = 0
        try:
            for obj in targets:
                mod = obj.modifiers.new("PolylineCarve", 'BOOLEAN')
                mod.operation = 'DIFFERENCE'
                mod.operand_type = 'OBJECT'
                mod.object = cutter
                try:
                    mod.solver = 'EXACT'
                except Exception:
                    pass
                # применяем к исходному мешу - ставим модификатор первым
                if len(obj.modifiers) > 1:
                    obj.modifiers.move(len(obj.modifiers) - 1, 0)
                try:
                    with context.temp_override(object=obj, active_object=obj):
                        bpy.ops.object.modifier_apply(modifier=mod.name, single_user=True)
                    done += 1
                except Exception as e:
                    obj.modifiers.remove(mod)
                    self.report({'WARNING'}, f"{obj.name}: не удалось применить Boolean ({e})")
        finally:
            bpy.data.objects.remove(cutter, do_unlink=True)
            bpy.data.meshes.remove(mesh)
        return done

    def carve_objects(self, context, s, targets):
        context.view_layer.update()
        overshoot = max(self.settings_grid() * 1e-3, 1e-5)  # старт чуть над поверхностью

        if self.volume:
            mesh = self.build_volume_cutter()
            if mesh is None:
                self.report({'WARNING'}, "Не удалось построить объём: точки лежат в одной плоскости")
                return 0
            return self.apply_cutter(context, targets, mesh)

        if self.cut_mode == 'NORMAL':
            mesh = self.build_surface_cutter(context, s, targets, overshoot)
            if mesh is None:
                self.report({'WARNING'}, "Не удалось построить резак")
                return 0
            return self.apply_cutter(context, targets, mesh)

        # Camera: вдоль лучей вида - в перспективе усечённая пирамида из камеры через точки
        # фигуры, в ортографике - призма вдоль направления взгляда.
        near = 1e-3
        rv3d = self.rv3d
        persp = rv3d.is_perspective
        eye = rv3d.view_matrix.inverted().translation.copy()
        fwd = (rv3d.view_rotation @ Vector((0.0, 0.0, -1.0))).normalized()
        if persp:
            def depth_of(x):
                return (x - eye).dot(fwd)

            def at_depth(pt, d):
                return eye + (pt - eye) * (d / depth_of(pt))

            if any(depth_of(pt) <= 1e-6 for pt in self.points):
                self.report({'WARNING'}, "Точки фигуры находятся за камерой")
                return 0
        else:
            def depth_of(x):
                return x.dot(fwd)

            def at_depth(pt, d):
                return pt + fwd * (d - pt.dot(fwd))

        # габариты целей по глубине
        dmin, dmax = float('inf'), float('-inf')
        for o in targets:
            for c in o.bound_box:
                d = depth_of(o.matrix_world @ Vector(c))
                dmin, dmax = min(dmin, d), max(dmax, d)

        if s.auto_depth:
            d0 = dmin - max(s.depth_margin, s.backward_depth)
            d1 = dmax + s.depth_margin
        else:
            # глубина считается от ближайшей к камере поверхности под фигурой
            if persp:
                def ray_for(pt):
                    return eye, (pt - eye).normalized()
            else:
                def ray_for(pt):
                    return at_depth(pt, dmin - 1.0), fwd
            hits = self.front_depth(context, targets, depth_of, ray_for)
            if hits is None:
                hits = (dmin, dmin)  # лучи ничего не задели - от передней границы объектов
            # Начало - у ближайшей точки поверхности, конец - Depth от самой дальней точки
            # поверхности под фигурой: так вырез гарантированно проходит всю фигуру на
            # наклонной или неровной поверхности (раньше глубина считалась только от ближней).
            d0 = hits[0] - s.backward_depth - overshoot
            d1 = hits[1] + s.depth

        if persp:
            d0 = max(d0, near)
            if d1 <= near:
                self.report({'WARNING'}, "Область выреза находится за камерой")
                return 0
        if d1 - d0 < 1e-6:
            d0, d1 = d0 - 0.5, d1 + 0.5
            if persp:
                d0 = max(d0, near)

        mesh = build_cutter_mesh([at_depth(pt, d0) for pt in self.points],
                                 [at_depth(pt, d1) for pt in self.points])
        return self.apply_cutter(context, targets, mesh)


# ----------------------------------------------------------------------------
# Инструмент в тулбаре, панель, меню
# ----------------------------------------------------------------------------
ICON_NAME = "polyline_carve_snap_icon"
ICON_B64 = (
    "VkNPAP//AAAtoSyiJJfWudm+0rnWTdxM3FK2LdZG1k2/o9a50rlJUEQmSSy/n9a01rnWudy63L7WtNxS3LqxLKomtiaKJkks"
    "RCaqLIomqibLvtK52b6xVLYttllu0GHRYM5Y3lroVt5Yq6KWlaSilqKXn5yVpKKWm6GfnJuhopaVpFu2WKtbtjqsSKk6rDmq"
    "SKk5qjinRqJIqTmqRqJrZGxkZWdsZFirXXA4pzekQ5w3pDWiPJZsZF1wYGtlZ2xkYGtGojinQ5w1ojOhPJZNp11wWKs3pDyW"
    "Q5xIqU2nW7ZNp1irW7YnqCepG6Ybph6eKKYeniSXK6MbpiimJ6gklyyULaEslDSTMKExoTSTM6E0kzyWM6E0kzGhMKEoph6e"
    "KKUopR6eKaQppB6eK6MslDChLqEslC6hLaEsoiujJJfWudy+2b7WTdZG3Ey2LbYm1ka/o7+f1rlJUERWRCa/n76b1rTWuda0"
    "3LrWtNZN3FK2JrYtsSyxLKosqiaKJoosSSyqLIosiibLvse50rmxVLEsti0brhumJ6kbriepJ6slvB+2KrAfthuuKK4lvC2y"
    "LL8ywCy/MLMvsyy/LbJGvDfAOLE3wDLAM7NA0EHLTc9By0a8T8tA0E3PTdFN00HVTdFB1UDQTdEywDCzM7Msvy+zMLNNz0HL"
    "Ts1OzUHLT8stsiW8K7IbrierJ6wbriesKK4orimvH7YpryqwH7YqsCuyJbxQyk/LRrxSyVDKRrxUyFLJWrpSyUa8Wro3wDOz"
    "NbNWyFTIWrpZyFbIWro3wDWzNrI3wDayOLFcyVnIYL1GvDixObBZyFq6YL1GvDmwOa45rjqsRrw6rFu2Rrxauka8W7ZgvWbB"
    "XMlcyWbBXsxmwWvIXsxryG7QYM5u0G3YYdRu0GHUYdFryGDOXsxE3UHVTdVB1U3TTdVR50rjU91K40TdT9lO10TdTdVi5Vro"
    "W91a6FHnVt5i5VvdXNxt2GngYNdp4GLlXtte22LlXNxg1mHUbdhg12DWbdhE3U7XT9lf2GDXaeBf2V/YaeBK40/ZUNtK41Db"
    "Utxe21/ZaeBK41LcU91b3VroWt1a3VroWN5R51PdVd1R51XdVt63yITKtL1Ac1pCY0qtX7a+omEudjRmRGyiVLJbq2u3tsa8"
    "wMyBxZHMi9uoWWNMZUBVS1s7a0K3yIbVhMpAczdsWkKtX8G8tr49fDt9OX05fTd9NHw0fDJ7LnYyezB6LnYudi1zLXEtcS1v"
    "LmwubC9qMGkwaTJnLmwyZzRmLmw0ZjdlOWU5ZTtlPWY9ZkBnRGxBaUNqRGxEbEVvRXFFcUVzRHZEdkN4QXpBekB7PXw9fDl9"
    "LnY5fTR8LnYudi1xLmw0ZjllRGw5ZT1mRGxEbEVxPXxFcUR2PXxEdkF6PXwudi5sNGY9fC52RGyia6BpnGSgaZ5onGScZJti"
    "m1+bX5tdnFueV6BVolSrVK5Vr1evV7FZq1SxWbJbq1SyW7Nds1+zX7NisluzYrJksluyZLFmq2uxZq9oq2uvaK5pq2ucZJtf"
    "nFucW55XnGSeV6JUnGSiVKdTslunU6tUslura6JrolSia5xkolSyW7Jkq2vAzL7Nu827zbnNt8y3zLXLsMa1y7PKsMazyrHI"
    "sMawxq/DsLyvw6/BsLyvwa+/sLywvLG6s7m1t7e2sLy3trm1u7W7tb61t7a+tcC2t7bAtsK3xrzCt8S5xrzEucW6xrzGvMe/"
    "x8HHwcfDxsbEysLLxsbCy8DMxsbAzLvNt8zGvMfBwMzHwcbGwMzAzLfMsMawxrC8wMywvLe2wMy3tsC2xryL24jchtyG3ITc"
    "gduB23/ae9V/2n7Ze9V+2XzXe9V71XrTetB60HrOe8x7zHzKfsh/xoHFe8yBxYTEhsSGxIjEi8WLxY3GkcyNxo/IkcyRzJLO"
    "ktCS0JLTkdWR1ZDXi9uQ14/Zi9uL24bcgdt71XrQe8yBxYbEkcyGxIvFkcyRzJLQi9uS0JHVi9uL24Hbe9V71XvMi9t7zIHF"
    "i9uoWaRkY0xkUWJSYFJgUl1SW1FbUVlQVUtZUFdPVUtVS1RJVEZURlREVUJXPlk8VUJZPFs7VUJgOmI7WztkO2Y8aD5oPmpA"
    "a0JrS2pNaE9kUWBSW1FVS1RGVUJkO2g+a0JrQmxGZFFsRmtLZFFrS2hPZFFkUVtRVUtVS1VCWztbO2Q7a0JkUVVLa0LY2Nj/"
    "2NjY/9jY2P91/6//df+v/3X/r/91/6//df+v/3X/r/91/6//df+v/3X/r/91/6//df+v/3X/r/91/6//df+v/3X/r/91/6//"
    "df+v/3X/r/91/6//df+v/3X/r/91/6//df+v/3X/r/91/6//df+v/3X/r/91/6//df+v/3X/r/91/6//df+v/3X/r/91/6//"
    "df+v/3X/r/91/6//df+v/3X/r//Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/"
    "2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/"
    "2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/"
    "2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/"
    "2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/"
    "2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/"
    "2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P91/6//df+v/3X/r/91/6//df+v/3X/r/91/6//"
    "df+v/3X/r/91/6//df+v/3X/r/91/6//df+v/3X/r/91/6//df+v/3X/r/91/6//df+v/3X/r/91/6//df+v/3X/r/91/6//"
    "df+v/3X/r/91/6//df+v/3X/r/91/6//df+v/3X/r/91/6//df+v/3X/r/91/6//df+v/3X/r/91/6//df+v/3X/r//Y2Nj/"
    "2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/"
    "2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/"
    "2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/"
    "2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/"
    "2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/"
    "2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/"
    "2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/"
    "2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/"
    "2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/"
    "2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/"
    "2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/"
    "2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/Y2Nj/2NjY/9jY2P/1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/"
    "9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW//1a1v/9Wtb//VrW/8="
)


def resolve_icon():
    """Записывает встроенную иконку (.dat) в папку конфигурации Blender и возвращает путь без расширения."""
    try:
        folder = bpy.utils.user_resource('CONFIG', path="polyline_carve_snap", create=True)
        if not folder:
            folder = bpy.app.tempdir
        path = os.path.join(folder, ICON_NAME)
        with open(path + ".dat", "wb") as f:
            f.write(base64.b64decode("".join(ICON_B64)))
        return path
    except Exception as e:
        print("Polyline Carve: не удалось записать иконку:", e)
        return "ops.generic.select_lasso"


_TOOL_KEYMAP = (
    ("object.polyline_carve_snap",
     {"type": 'LEFTMOUSE', "value": 'PRESS'},
     {"properties": [("from_tool", True)]}),
)


def _tool_draw_settings(context, layout, tool):
    region = getattr(context, "region", None)
    header = region is not None and region.type == 'TOOL_HEADER'
    draw_settings_ui(layout, context.scene.polyline_carve_snap, header=header)


class PCS_Tool_Object(WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'OBJECT'
    bl_idname = "object.polyline_carve_snap_tool"
    bl_label = "Polyline Carve (Advanced)"
    bl_description = "Вырезание многоугольной полости в выделенных объектах с привязкой точек к сетке"
    bl_icon = "ops.generic.select_lasso"
    bl_widget = None
    bl_keymap = _TOOL_KEYMAP
    draw_settings = _tool_draw_settings


class PCS_Tool_Edit(WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'EDIT_MESH'
    bl_idname = "mesh.polyline_carve_snap_tool"
    bl_label = "Polyline Carve (Advanced)"
    bl_description = "Вырезание многоугольной полости в редактируемых мешах с привязкой точек к сетке"
    bl_icon = "ops.generic.select_lasso"
    bl_widget = None
    bl_keymap = _TOOL_KEYMAP
    draw_settings = _tool_draw_settings


class VIEW3D_PT_polyline_carve_snap(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tool"
    bl_label = "Polyline Carve (Advanced)"

    def draw(self, context):
        layout = self.layout
        layout.operator(OBJECT_OT_polyline_carve_snap.bl_idname, icon='MOD_BOOLEAN')
        draw_settings_ui(layout, context.scene.polyline_carve_snap)


def menu_func(self, context):
    self.layout.operator(OBJECT_OT_polyline_carve_snap.bl_idname, icon='MOD_BOOLEAN')


classes = (
    PCS_Settings,
    OBJECT_OT_polyline_carve_snap,
    VIEW3D_PT_polyline_carve_snap,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.polyline_carve_snap = PointerProperty(type=PCS_Settings)
    bpy.types.VIEW3D_MT_object.append(menu_func)
    bpy.types.VIEW3D_MT_edit_mesh.append(menu_func)

    icon = resolve_icon()
    for tool in (PCS_Tool_Object, PCS_Tool_Edit):
        tool.bl_icon = icon
        try:
            # в конец списка, отдельно от всех групп инструментов (с разделителем)
            bpy.utils.register_tool(tool, after=None, separator=True, group=False)
        except Exception as e:
            print("Polyline Carve: не удалось зарегистрировать инструмент:", e)


def unregister():
    for tool in (PCS_Tool_Edit, PCS_Tool_Object):
        try:
            bpy.utils.unregister_tool(tool)
        except Exception:
            pass
    bpy.types.VIEW3D_MT_edit_mesh.remove(menu_func)
    bpy.types.VIEW3D_MT_object.remove(menu_func)
    del bpy.types.Scene.polyline_carve_snap
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
