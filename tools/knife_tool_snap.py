bl_info = {
    "name": "Knife (Grid Snap)",
    "author": "Anthropic Claude",
    "version": (1, 1, 0),
    "blender": (5, 2, 0),
    "location": "3D Viewport > Edit Mode > Toolbar / Mesh menu / Sidebar > Tool",
    "description": "Нож для Edit Mode, у которого точки разреза привязываются к сетке",
    "category": "Mesh",
}

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
from mathutils import Vector, geometry
from bpy.props import BoolProperty, FloatProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup, WorkSpaceTool

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
class KS_Settings(PropertyGroup):
    cut_through: BoolProperty(
        name="Cut Through",
        description="Резать также невидимую (скрытую за другими гранями) геометрию",
        default=True,
    )


def draw_settings_ui(layout, s):
    col = layout.column(align=True)
    col.prop(s, "cut_through")


# ----------------------------------------------------------------------------
# Отрисовка
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
        g = op.grid()

        if hover is not None:
            eu = Vector((0, 0, 0))
            ev = Vector((0, 0, 0))
            eu[op.iu] = g
            ev[op.iv] = g
            k = 4
            lines = []
            for i in range(-k, k + 1):
                lines += [hover + eu * i - ev * k, hover + eu * i + ev * k]
                lines += [hover + ev * i - eu * k, hover + ev * i + eu * k]
            draw('LINES', lines, (0.8, 0.8, 0.8, 0.25), width=1.0)

        pts = op.points
        if pts:
            draw('LINE_STRIP', pts, (0.2, 0.9, 1.0, 1.0), width=2.5)
            if hover is not None:
                draw('LINES', [pts[-1], hover], (0.2, 0.9, 1.0, 0.7), width=1.5)
            if len(pts) >= 3:
                draw('LINES', [pts[-1], pts[0]], (0.3, 1.0, 0.3, 0.35), width=1.5)
            draw('POINTS', pts, (1.0, 1.0, 1.0, 1.0), size=8.0)
            draw('POINTS', [pts[0]], (0.3, 1.0, 0.3, 1.0), size=11.0)
        if hover is not None:
            draw('POINTS', [hover], (1.0, 0.9, 0.1, 1.0), size=10.0)

        gpu.state.blend_set('NONE')
        gpu.state.line_width_set(1.0)
        gpu.state.point_size_set(1.0)
    except Exception as e:
        print("Knife Snap draw error:", e)


# ----------------------------------------------------------------------------
# Оператор
# ----------------------------------------------------------------------------
class MESH_OT_knife_snap(Operator):
    bl_idname = "mesh.knife_snap"
    bl_label = "Knife (Grid Snap)"
    bl_description = "Нож с привязкой точек разреза к сетке"
    bl_options = {'REGISTER', 'UNDO'}

    from_tool: BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        return (context.area is not None and context.area.type == 'VIEW_3D'
                and context.mode == 'EDIT_MESH')

    # --- вспомогательное ----------------------------------------------------
    def grid(self):
        return get_grid_size(self.space, self.scene)

    def resolve_plane(self, context):
        if self.axis_mode == 'AUTO':
            vd = self.rv3d.view_rotation @ Vector((0.0, 0.0, -1.0))
            i = max(range(3), key=lambda k: abs(vd[k]))
            self.axis_n = i if abs(vd[i]) > 0.999 else 2
        else:
            self.axis_n = {'X': 0, 'Y': 1, 'Z': 2}[self.axis_mode]
        self.iu, self.iv = [k for k in range(3) if k != self.axis_n]
        self.plane_co = context.scene.cursor.location.copy()

    def snap(self, p):
        g = self.grid()
        q = p.copy()
        for i in range(3):
            q[i] = self.plane_co[i] if i == self.axis_n else round(p[i] / g) * g
        return q

    def mouse_to_plane(self, event):
        coord = (event.mouse_region_x, event.mouse_region_y)
        origin = view3d_utils.region_2d_to_origin_3d(self.region, self.rv3d, coord)
        vec = view3d_utils.region_2d_to_vector_3d(self.region, self.rv3d, coord)
        n = Vector((0.0, 0.0, 0.0))
        n[self.axis_n] = 1.0
        if abs(vec.dot(n)) < 0.02:
            return None
        hit = geometry.intersect_line_plane(origin, origin + vec * 1000.0,
                                            self.plane_co, n, False)
        if hit is None:
            return None
        if self.rv3d.is_perspective and (hit - origin).dot(vec) <= 0:
            return None
        return hit

    def update_hover(self, context, event):
        if not self.points and self.axis_mode == 'AUTO':
            self.resolve_plane(context)
        hit = self.mouse_to_plane(event)
        self.hover = self.snap(hit) if hit is not None else None

    def in_region(self, event):
        return (0 <= event.mouse_region_x < self.region.width and
                0 <= event.mouse_region_y < self.region.height)

    def near_first(self, event):
        if not self.points:
            return False
        s = view3d_utils.location_3d_to_region_2d(self.region, self.rv3d, self.points[0])
        if s is None:
            return False
        return (s - Vector((event.mouse_region_x, event.mouse_region_y))).length < CLOSE_PIXELS

    def add_point_at_hover(self):
        p = self.hover
        if p is None:
            return
        if self.points and (p - self.points[-1]).length < 1e-9:
            return
        self.points.append(p.copy())

    def update_status(self, context):
        txt = ("Knife Snap | LMB: point | Enter/Space: finish cut | click first point: close loop | "
               "Backspace: undo point | X/Y/Z/A: plane | "
               f"Grid: {self.grid():.4g} | Plane normal: "
               f"{'XYZ'[self.axis_n]}{' (auto)' if self.axis_mode == 'AUTO' else ''} | "
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

        self.scene = context.scene
        self.area = context.area
        self.space = context.space_data
        self.region = context.region
        self.rv3d = context.region_data
        self.points = []
        self.hover = None
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

    def modal(self, context, event):
        if context.area is None or context.mode != 'EDIT_MESH':
            self.cleanup(context)
            return {'CANCELLED'}
        context.area.tag_redraw()

        if event.type in NAV_EVENTS or \
                (event.type.startswith('NUMPAD_') and event.type != 'NUMPAD_ENTER'):
            return {'PASS_THROUGH'}

        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            self.update_hover(context, event)
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if not self.in_region(event):
                return {'PASS_THROUGH'}
            self.update_hover(context, event)
            if len(self.points) >= 3 and self.near_first(event):
                return self.finish(context, closed=True)
            self.add_point_at_hover()
            self.update_status(context)
            return {'RUNNING_MODAL'}

        if event.value == 'PRESS':
            if event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'}:
                return self.finish(context, closed=False)
            if event.type in {'ESC', 'RIGHTMOUSE'}:
                self.cleanup(context)
                return {'CANCELLED'}
            if event.type == 'BACK_SPACE' or (event.type == 'Z' and event.ctrl):
                if self.points:
                    self.points.pop()
                self.update_status(context)
                return {'RUNNING_MODAL'}
            if event.type in {'X', 'Y', 'Z', 'A'}:
                if self.points:
                    self.report({'INFO'}, "Смена плоскости возможна только пока нет точек")
                else:
                    self.axis_mode = 'AUTO' if event.type == 'A' else event.type
                    self.resolve_plane(context)
                    self.update_hover(context, event)
                    self.update_status(context)
                return {'RUNNING_MODAL'}

        return {'RUNNING_MODAL'}

    def finish(self, context, closed):
        if len(self.points) < 2:
            self.report({'WARNING'}, "Нужно минимум 2 точки")
            return {'RUNNING_MODAL'}
        self.cleanup(context)
        return {'FINISHED'} if self.cut(context, closed) else {'CANCELLED'}

    # --- разрез -------------------------------------------------------------
    def cut(self, context, closed):
        """Создаёт временную полилинию-кривую и вызывает mesh.knife_project."""
        s = context.scene.knife_snap_settings
        view_layer = context.view_layer

        curve = bpy.data.curves.new("KnifeSnapCutter", 'CURVE')
        curve.dimensions = '3D'
        spline = curve.splines.new('POLY')
        spline.points.add(len(self.points) - 1)
        for sp, p in zip(spline.points, self.points):
            sp.co = (p.x, p.y, p.z, 1.0)
        spline.use_cyclic_u = bool(closed and len(self.points) >= 3)

        cutter = bpy.data.objects.new("KnifeSnapCutter", curve)
        context.scene.collection.objects.link(cutter)

        # запоминаем выделение объектов вне Edit Mode, чтобы они не стали "ножами"
        stash = [o for o in view_layer.objects if o.select_get() and o.mode != 'EDIT']
        ok = False
        try:
            for o in stash:
                o.select_set(False)
            cutter.select_set(True)
            view_layer.update()

            with context.temp_override(area=self.area, region=self.region):
                bpy.ops.mesh.knife_project(cut_through=s.cut_through)
            ok = True
        except Exception as e:
            self.report({'ERROR'}, f"Не удалось выполнить разрез: {e}")
        finally:
            bpy.data.objects.remove(cutter, do_unlink=True)
            bpy.data.curves.remove(curve)
            for o in stash:
                try:
                    o.select_set(True)
                except Exception:
                    pass
        return ok

# ----------------------------------------------------------------------------
# Инструмент, панель, меню
# ----------------------------------------------------------------------------
class KS_Tool(WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'EDIT_MESH'
    bl_idname = "mesh.knife_snap_tool"
    bl_label = "Knife (Grid Snap)"
    bl_description = "Нож, у которого точки разреза привязываются к сетке"
    bl_icon = "ops.mesh.knife_tool"
    bl_widget = None
    bl_keymap = (
        ("mesh.knife_snap",
         {"type": 'LEFTMOUSE', "value": 'PRESS'},
         {"properties": [("from_tool", True)]}),
    )

    def draw_settings(context, layout, tool):
        draw_settings_ui(layout, context.scene.knife_snap_settings)


class VIEW3D_PT_knife_snap(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tool"
    bl_label = "Knife (Grid Snap)"

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def draw(self, context):
        layout = self.layout
        layout.operator(MESH_OT_knife_snap.bl_idname, icon='MOD_BOOLEAN')
        draw_settings_ui(layout, context.scene.knife_snap_settings)


def menu_func(self, context):
    self.layout.operator(MESH_OT_knife_snap.bl_idname)


classes = (
    KS_Settings,
    MESH_OT_knife_snap,
    VIEW3D_PT_knife_snap,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.knife_snap_settings = PointerProperty(type=KS_Settings)
    bpy.types.VIEW3D_MT_edit_mesh.append(menu_func)
    try:
        bpy.utils.register_tool(KS_Tool, after={"builtin.knife"},
                                separator=False, group=False)
    except Exception as e:
        print("Knife Snap: не удалось зарегистрировать инструмент:", e)


def unregister():
    try:
        bpy.utils.unregister_tool(KS_Tool)
    except Exception:
        pass
    bpy.types.VIEW3D_MT_edit_mesh.remove(menu_func)
    del bpy.types.Scene.knife_snap_settings
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
