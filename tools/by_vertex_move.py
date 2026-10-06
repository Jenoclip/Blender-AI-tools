bl_info = {
    "name": "Move (Individual Offset)",
    "author": "Anthropic Claude",
    "version": (1, 3, 0),
    "blender": (5, 2, 0),
    "location": "3D Viewport > Toolbar (Object Mode and Edit Mode)",
    "description": "Инструмент перемещения со стрелками: каждая точка/объект смещается "
                   "на вектор смещения, с привязкой вектора к сетке вьюпорта",
    "category": "Object",
}

import math
import bpy
import bmesh
import gpu
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
from mathutils import Vector, Matrix, geometry
from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                       FloatVectorProperty, PointerProperty)
from bpy.types import Operator, PropertyGroup, WorkSpaceTool, GizmoGroup

OP_ID = "view3d.individual_move"
GROUP_ID = "VIEW3D_GGT_individual_move"
AXIS_COLORS = {'X': (0.93, 0.22, 0.32), 'Y': (0.55, 0.86, 0.0), 'Z': (0.16, 0.55, 1.0)}
AXIS_VECS = {'X': Vector((1, 0, 0)), 'Y': Vector((0, 1, 0)), 'Z': Vector((0, 0, 1))}
NAV_EVENTS = {
    'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'WHEELINMOUSE',
    'WHEELOUTMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE', 'NDOF_MOTION',
}
ARROW_PX = 70.0      # длина стрелки в пикселях (при gizmo_scale = 1)
HEAD_PX = 16.0       # длина наконечника
HEAD_R_PX = 6.0      # радиус наконечника
RING_PX = 10.0       # радиус центрального кольца

# общее состояние между гизмо и оператором
STATE = {"pivot": None, "busy": False}

TOOL_IDS = {"object.individual_move_tool", "mesh.individual_move_tool"}


def our_tool_active(context):
    """True, только если в текущем режиме активен именно наш инструмент."""
    try:
        tool = context.workspace.tools.from_space_view3d_mode(context.mode, create=False)
    except Exception:
        return False
    return tool is not None and tool.idname in TOOL_IDS


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
class IM_Settings(PropertyGroup):
    use_snap: BoolProperty(
        name="Snap to Grid",
        description="Привязка к сетке вьюпорта (Ctrl во время перетаскивания инвертирует)",
        default=True,
    )
    snap_mode: EnumProperty(
        name="Snap Mode",
        items=(
            ('OFFSET', "Offset",
             "Вектор смещения кратен размеру сетки, все точки сдвигаются на него одинаково"),
            ('EACH', "Each Point",
             "Итоговая позиция КАЖДОЙ точки/объекта индивидуально привязывается к сетке"),
        ),
        default='OFFSET',
    )
    gizmo_scale: FloatProperty(
        name="Gizmo Scale", description="Размер стрелок",
        default=1.0, min=0.1, max=5.0,
    )


def draw_settings_ui(layout, s):
    col = layout.column(align=True)
    col.prop(s, "use_snap")
    sub = col.column(align=True)
    sub.active = s.use_snap
    sub.prop(s, "snap_mode", text="")
    col.prop(s, "gizmo_scale")


# ----------------------------------------------------------------------------
# Опорная точка для гизмо
# ----------------------------------------------------------------------------
def compute_pivot(context):
    pts = []
    if context.mode == 'EDIT_MESH':
        for obj in context.objects_in_mode_unique_data:
            if obj.type != 'MESH':
                continue
            bm = bmesh.from_edit_mesh(obj.data)
            mw = obj.matrix_world
            pts.extend(mw @ v.co for v in bm.verts if v.select and not v.hide)
    elif context.mode == 'OBJECT':
        pts.extend(o.matrix_world.translation.copy() for o in context.selected_objects)
    if not pts:
        return None
    total = Vector((0.0, 0.0, 0.0))
    for p in pts:
        total += p
    return total / len(pts)


# ----------------------------------------------------------------------------
# Отрисовка ориентира во время перетаскивания
# ----------------------------------------------------------------------------
def draw_overlay(op):
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

        p0 = op.pivot0
        p = op.pivot0 + op.final_delta
        ps = op.pixel_size(p)
        sc = op.settings.gizmo_scale
        L, hl, hr = ARROW_PX * sc * ps, HEAD_PX * sc * ps, HEAD_R_PX * sc * ps

        # ось, на которой идёт перемещение
        if op.axis in AXIS_VECS:
            a = AXIS_VECS[op.axis]
            c = AXIS_COLORS[op.axis]
            draw('LINES', [p0 - a * 1000.0, p0 + a * 1000.0], (c[0], c[1], c[2], 0.35), width=1.5)

        # вектор смещения и начальная точка
        if (p - p0).length > 1e-9:
            draw('LINES', [p0, p], (1.0, 1.0, 1.0, 0.9), width=2.0)
        draw('POINTS', [p0], (1.0, 1.0, 1.0, 0.9), size=6.0)

        # стрелки - двигаются вместе с активной
        for name, a in AXIS_VECS.items():
            col = AXIS_COLORS[name]
            alpha = 1.0 if name == op.axis else 0.6
            color = (col[0], col[1], col[2], alpha)
            tip = p + a * L
            base = p + a * (L - hl)
            draw('LINES', [p, base], color, width=3.0)
            ref = Vector((0, 0, 1)) if abs(a.z) < 0.9 else Vector((1, 0, 0))
            u = a.cross(ref).normalized()
            v = a.cross(u)
            n = 8
            ring = [base + (u * math.cos(2 * math.pi * i / n) +
                            v * math.sin(2 * math.pi * i / n)) * hr for i in range(n)]
            tris = []
            for i in range(n):
                tris += [tip, ring[i], ring[(i + 1) % n]]
            draw('TRIS', tris, color)

        # центральное кольцо (свободное перемещение в плоскости вида)
        rot = op.rv3d.view_rotation
        right = rot @ Vector((1.0, 0.0, 0.0))
        up = rot @ Vector((0.0, 1.0, 0.0))
        r = RING_PX * sc * ps
        n = 32
        ring = [p + (right * math.cos(2 * math.pi * i / n) +
                     up * math.sin(2 * math.pi * i / n)) * r for i in range(n)]
        draw('LINE_LOOP', ring, (1.0, 1.0, 1.0, 1.0 if op.axis == 'VIEW' else 0.7), width=2.0)

        gpu.state.blend_set('NONE')
        gpu.state.line_width_set(1.0)
        gpu.state.point_size_set(1.0)
    except Exception as e:
        print("Individual Move draw error:", e)


# ----------------------------------------------------------------------------
# Оператор
# ----------------------------------------------------------------------------
class VIEW3D_OT_individual_move(Operator):
    bl_idname = OP_ID
    bl_label = "Move (Individual)"
    bl_description = "Смещает каждую выделенную точку/объект на заданный вектор смещения"
    bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}

    axis: EnumProperty(
        name="Axis",
        items=(('X', "X", ""), ('Y', "Y", ""), ('Z', "Z", ""), ('VIEW', "View", "")),
        default='VIEW',
    )
    offset: FloatVectorProperty(name="Offset", size=3, subtype='TRANSLATION',
                                unit='LENGTH', default=(0.0, 0.0, 0.0))
    snap_active: BoolProperty(name="Snap", default=False, options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return context.mode in {'OBJECT', 'EDIT_MESH'}

    # --- сбор данных --------------------------------------------------------
    def collect(self, context):
        self.edit = context.mode == 'EDIT_MESH'
        self.items = []
        positions = []
        if self.edit:
            for obj in context.objects_in_mode_unique_data:
                if obj.type != 'MESH':
                    continue
                bm = bmesh.from_edit_mesh(obj.data)
                mw = obj.matrix_world.copy()
                inv = mw.inverted_safe()
                # храним индексы, а не BMVert: ссылки на элементы становятся
                # недействительными, когда Python-обёртка BMesh уничтожается
                verts = [(v.index, mw @ v.co) for v in bm.verts if v.select and not v.hide]
                if verts:
                    self.items.append((obj, inv, verts))
                    positions.extend(w for _, w in verts)
        else:
            sel = set(context.selected_objects)
            for obj in sel:
                p, skip = obj.parent, False
                while p:
                    if p in sel:
                        skip = True
                        break
                    p = p.parent
                if not skip:
                    m0 = obj.matrix_world.copy()
                    self.items.append((obj, m0))
                    positions.append(m0.translation.copy())
        if not positions:
            return False
        total = Vector((0.0, 0.0, 0.0))
        for p in positions:
            total += p
        self.pivot0 = total / len(positions)
        self.settings = context.scene.individual_move
        self.rv3d = context.region_data
        self.region = context.region
        self.grid = get_grid_size(context.space_data, context.scene)
        self.setup_mask()
        return True

    def setup_mask(self):
        if self.axis in AXIS_VECS:
            self.mask = tuple(i == 'XYZ'.index(self.axis) for i in range(3))
        else:
            mask = [True, True, True]
            if self.rv3d is not None:
                vd = self.rv3d.view_rotation @ Vector((0.0, 0.0, -1.0))
                i = max(range(3), key=lambda k: abs(vd[k]))
                if abs(vd[i]) > 0.999:
                    mask[i] = False
            self.mask = tuple(mask)

    # --- применение смещения ------------------------------------------------
    def apply(self, delta, snap):
        g = self.grid
        mask = self.mask
        delta = delta.copy()
        if snap and self.settings.snap_mode == 'OFFSET':
            for i in range(3):
                if mask[i]:
                    delta[i] = round(delta[i] / g) * g
        each = snap and self.settings.snap_mode == 'EACH'

        def conv(p):
            q = p + delta
            if each:
                for i in range(3):
                    if mask[i]:
                        q[i] = round(q[i] / g) * g
            return q

        if self.edit:
            for obj, inv, verts in self.items:
                bm = bmesh.from_edit_mesh(obj.data)
                bm.verts.ensure_lookup_table()
                for i, w in verts:
                    bm.verts[i].co = inv @ conv(w)
                bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=False)
        else:
            for obj, m0 in self.items:
                m = m0.copy()
                m.translation = conv(m0.translation)
                obj.matrix_world = m
        return delta

    # --- проекция мыши ------------------------------------------------------
    def pixel_size(self, p):
        """Размер одного пикселя экрана в мировых единицах на глубине точки p."""
        pm = self.rv3d.perspective_matrix
        w = (pm @ Vector((p.x, p.y, p.z, 1.0))).w
        return max(abs(w), 1e-6) * 2.0 / (self.region.width * self.rv3d.window_matrix[0][0])

    def raw(self, event):
        coord = Vector((event.mouse_region_x, event.mouse_region_y))
        region, rv3d = self.region, self.rv3d

        if self.axis in AXIS_VECS:
            a = AXIS_VECS[self.axis]
            # Берём экранную проекцию оси и ближайшую к курсору точку на ней.
            # Луч через эту точку пересекает ось ровно там, где она видна под курсором,
            # поэтому стрелка двигается строго за курсором (в т.ч. в перспективе).
            s0 = view3d_utils.location_3d_to_region_2d(region, rv3d, self.pivot0)
            if s0 is None:
                return None
            # Шаг для проекции оси подбирается как ~100 пикселей на глубине опорной точки,
            # поэтому результат не зависит от zoom и расстояния до камеры.
            step = 100.0 * self.pixel_size(self.pivot0)
            d2 = None
            for sign in (1.0, -1.0):  # вторая попытка - если первая точка за камерой
                s1 = view3d_utils.location_3d_to_region_2d(
                    region, rv3d, self.pivot0 + a * (step * sign))
                if s1 is not None:
                    d2 = (s1 - s0) * sign
                    break
            if d2 is None:
                return None
            dl = d2.length_squared
            if dl < 4.0:  # ось направлена почти вдоль взгляда
                return None
            q = s0 + d2 * ((coord - s0).dot(d2) / dl)
            o = view3d_utils.region_2d_to_origin_3d(region, rv3d, q)
            v = view3d_utils.region_2d_to_vector_3d(region, rv3d, q)
            # ближайшая точка оси к лучу (аналитически, без порогов и обрезки по масштабу)
            w0 = self.pivot0 - o
            b = a.dot(v)
            den = 1.0 - b * b
            if den < 1e-10:
                return None
            return (b * v.dot(w0) - a.dot(w0)) / den

        o = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        v = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        n = rv3d.view_rotation @ Vector((0.0, 0.0, 1.0))
        return geometry.intersect_line_plane(o, o + v, self.pivot0, n, False)

    def compute_delta(self, r):
        if self.axis in AXIS_VECS:
            return AXIS_VECS[self.axis] * (r - self.ref)
        return r - self.ref

    # --- жизненный цикл -----------------------------------------------------
    def invoke(self, context, event):
        if context.region_data is None or context.area.type != 'VIEW_3D':
            return {'CANCELLED'}
        if not self.collect(context):
            self.report({'WARNING'}, "Нет выделенной геометрии/объектов")
            return {'CANCELLED'}
        self.delta = Vector((0.0, 0.0, 0.0))
        self.final_delta = Vector((0.0, 0.0, 0.0))
        self.snap_now = False

        r = self.raw(event)
        if r is None:
            r = 0.0 if self.axis in AXIS_VECS else self.pivot0.copy()
        self.ref = r

        STATE["busy"] = True
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            draw_overlay, (self,), 'WINDOW', 'POST_VIEW')
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def cleanup(self, context):
        if getattr(self, "_handle", None) is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None
        context.workspace.status_text_set(None)
        STATE["busy"] = False
        if context.area:
            context.area.tag_redraw()

    def update(self, context, event):
        r = self.raw(event)
        if r is not None:
            self.delta = self.compute_delta(r)
        self.snap_now = self.settings.use_snap != event.ctrl
        self.final_delta = self.apply(self.delta, self.snap_now)
        STATE["pivot"] = self.pivot0 + self.final_delta
        d = self.final_delta
        snap_txt = (f"ON ({self.settings.snap_mode.lower()}, grid {self.grid:.4g})"
                    if self.snap_now else "OFF")
        context.workspace.status_text_set(
            f"Move (Individual) | dX {d.x:.4g}  dY {d.y:.4g}  dZ {d.z:.4g} | "
            f"Snap: {snap_txt} (Ctrl: toggle) | Enter/release: confirm | Esc/RMB: cancel")
        context.area.tag_redraw()

    def modal(self, context, event):
        if context.area is None or context.mode not in {'OBJECT', 'EDIT_MESH'}:
            self.apply(Vector((0.0, 0.0, 0.0)), False)
            self.cleanup(context)
            return {'CANCELLED'}
        if event.type in NAV_EVENTS:
            return {'PASS_THROUGH'}

        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE', 'LEFT_CTRL', 'RIGHT_CTRL'}:
            self.update(context, event)
            return {'RUNNING_MODAL'}

        if (event.type == 'LEFTMOUSE' and event.value == 'RELEASE') or \
                (event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS'):
            self.update(context, event)
            self.offset = self.final_delta
            self.snap_active = self.snap_now
            moved = self.final_delta.length > 1e-12 or \
                (self.snap_now and self.settings.snap_mode == 'EACH')
            self.cleanup(context)
            STATE["pivot"] = compute_pivot(context)
            return {'FINISHED'} if moved else {'CANCELLED'}

        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self.apply(Vector((0.0, 0.0, 0.0)), False)
            self.cleanup(context)
            STATE["pivot"] = compute_pivot(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def execute(self, context):
        """Повтор из панели Adjust Last Operation."""
        if not self.collect(context):
            return {'CANCELLED'}
        self.apply(Vector(self.offset), self.snap_active)
        STATE["pivot"] = compute_pivot(context)
        return {'FINISHED'}


# ----------------------------------------------------------------------------
# Гизмо со стрелками
# ----------------------------------------------------------------------------
class IM_GGT_move(GizmoGroup):
    bl_idname = GROUP_ID
    bl_label = "Move (Individual) Gizmo"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D'}

    @classmethod
    def poll(cls, context):
        # без проверки инструмента группа рисовалась поверх любых других инструментов
        return context.mode in {'OBJECT', 'EDIT_MESH'} and our_tool_active(context)

    def setup(self, context):
        self.arrows = []
        for axis in ('X', 'Y', 'Z'):
            g = self.gizmos.new("GIZMO_GT_arrow_3d")
            props = g.target_set_operator(OP_ID)
            props.axis = axis
            g.draw_style = 'NORMAL'
            g.color = AXIS_COLORS[axis]
            g.alpha = 0.9
            g.color_highlight = (1.0, 1.0, 1.0)
            g.alpha_highlight = 1.0
            self.arrows.append((g, AXIS_VECS[axis]))

        c = self.gizmos.new("GIZMO_GT_move_3d")
        props = c.target_set_operator(OP_ID)
        props.axis = 'VIEW'
        c.draw_style = 'RING_2D'
        c.color = (0.9, 0.9, 0.9)
        c.alpha = 0.8
        c.color_highlight = (1.0, 1.0, 1.0)
        c.alpha_highlight = 1.0
        self.center = c
        self.refresh(context)

    def refresh(self, context):
        if STATE["busy"]:
            return
        STATE["pivot"] = compute_pivot(context)

    def draw_prepare(self, context):
        p = STATE["pivot"]
        hide = p is None
        busy = STATE["busy"]
        scale = context.scene.individual_move.gizmo_scale
        # Во время перетаскивания родные гизмо делаем невидимыми: их собственное
        # смещение складывалось с нашим (из-за этого стрелка бежала быстрее курсора).
        # Вместо них рисуется ориентир из draw_overlay.
        for g, vec in self.arrows:
            g.hide = hide
            g.alpha = 0.0 if busy else 0.9
            g.alpha_highlight = 0.0 if busy else 1.0
            if not hide:
                rot = Vector((0.0, 0.0, 1.0)).rotation_difference(vec).to_matrix().to_4x4()
                g.matrix_basis = Matrix.Translation(p) @ rot
                g.scale_basis = 0.8 * scale
        self.center.hide = hide
        self.center.alpha = 0.0 if busy else 0.8
        self.center.alpha_highlight = 0.0 if busy else 1.0
        if not hide:
            self.center.matrix_basis = Matrix.Translation(p)
            self.center.scale_basis = 0.12 * scale


# ----------------------------------------------------------------------------
# Инструменты
# ----------------------------------------------------------------------------
TOOL_KEYMAP = (
    ("view3d.select",
     {"type": 'LEFTMOUSE', "value": 'PRESS'},
     {"properties": [("deselect_all", True)]}),
    ("view3d.select_box",
     {"type": 'LEFTMOUSE', "value": 'CLICK_DRAG'},
     {"properties": [("mode", 'SET')]}),
)


def _tool_draw_settings(context, layout, tool):
    draw_settings_ui(layout, context.scene.individual_move)


class IM_Tool_Object(WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'OBJECT'
    bl_idname = "object.individual_move_tool"
    bl_label = "Move (Individual)"
    bl_description = "Перемещение: каждый объект смещается на вектор смещения"
    bl_icon = "ops.transform.translate"
    bl_widget = GROUP_ID
    bl_keymap = TOOL_KEYMAP
    draw_settings = _tool_draw_settings


class IM_Tool_Edit(WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'EDIT_MESH'
    bl_idname = "mesh.individual_move_tool"
    bl_label = "Move (Individual)"
    bl_description = "Перемещение: каждая точка смещается на вектор смещения"
    bl_icon = "ops.transform.translate"
    bl_widget = GROUP_ID
    bl_keymap = TOOL_KEYMAP
    draw_settings = _tool_draw_settings


classes = (
    IM_Settings,
    VIEW3D_OT_individual_move,
    IM_GGT_move,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.individual_move = PointerProperty(type=IM_Settings)
    for tool in (IM_Tool_Object, IM_Tool_Edit):
        try:
            bpy.utils.register_tool(tool, after={"builtin.transform"},
                                    separator=False, group=False)
        except Exception as e:
            print("Individual Move: не удалось зарегистрировать инструмент:", e)


def unregister():
    for tool in (IM_Tool_Edit, IM_Tool_Object):
        try:
            bpy.utils.unregister_tool(tool)
        except Exception:
            pass
    del bpy.types.Scene.individual_move
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
