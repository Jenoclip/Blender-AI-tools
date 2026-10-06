bl_info = {
    "name": "Array Copy Tool",
    "author": "Anthropic Claude",
    "version": (1, 0, 0),
    "blender": (5, 2, 0),
    "location": "3D Viewport > Toolbar (Object Mode) > Array Copy",
    "description": "Gizmo-driven object copy tool: drag the gizmo to set the offset step, "
                   "rotate it to set the copy direction. Supports per-copy rotation and "
                   "scale steps and puts the result into a collection named after the source.",
    "category": "Object",
}

import math
import time

import bpy
import gpu
from bpy.props import (
    BoolProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
)
from bpy.types import GizmoGroup, Operator, PropertyGroup, WorkSpaceTool
from gpu_extras.batch import batch_for_shader
from mathutils import Euler, Matrix, Vector

TOOL_ID = "object.array_copy_tool"
GIZMO_GROUP_ID = "VIEW3D_GGT_array_copy"
COLL_KEY = "array_copy_source"   # custom property on the created collection
OBJ_KEY = "array_copy_of"        # custom property on created copies
COMMIT_DELAY = 0.15              # seconds after the gizmo is released before copies are built

AXIS_NAMES = ('X', 'Y', 'Z')
AXIS_COLORS = ((1.0, 0.2, 0.32), (0.54, 0.86, 0.0), (0.16, 0.55, 1.0))
# rotate gizmo local +Z onto X / Y / Z
AXIS_MATS = (
    Matrix.Rotation(math.radians(90.0), 4, 'Y'),
    Matrix.Rotation(math.radians(-90.0), 4, 'X'),
    Matrix.Identity(4),
)


# ----------------------------------------------------------------------------
# Shared parameters (used by both the Scene settings and the operator)
# ----------------------------------------------------------------------------

class ArrayCopyParams:
    step: FloatVectorProperty(
        name="Step",
        description="Offset between copies, in the local axes of the direction frame",
        size=3, subtype='TRANSLATION', unit='LENGTH', default=(0.0, 0.0, 0.0),
    )
    direction: FloatVectorProperty(
        name="Direction",
        description="Orientation of the offset frame (direction in which copies are placed)",
        size=3, subtype='EULER', default=(0.0, 0.0, 0.0),
    )
    snap_to_grid: BoolProperty(
        name="Snap to Grid",
        description="Snap the offset step to the viewport grid size",
        default=False,
    )
    snap_to_angle: BoolProperty(
        name="Snap to Angle",
        description="Snap direction rotation to the given angle increment",
        default=False,
    )
    snap_angle: FloatProperty(
        name="Angle Increment",
        description="Angle multiple used for rotation snapping",
        subtype='ANGLE', unit='ROTATION',
        min=math.radians(0.1), max=math.radians(180.0), default=math.radians(15.0),
    )
    rot_step: FloatVectorProperty(
        name="Rotation Step",
        description="Every next copy is additionally rotated by this XYZ step (world axes)",
        size=3, subtype='EULER', default=(0.0, 0.0, 0.0),
    )
    rot_pivot: FloatVectorProperty(
        name="Rotation Pivot",
        description="Pivot of the step rotation, offset from the origin of each copy (world axes)",
        size=3, subtype='TRANSLATION', unit='LENGTH', default=(0.0, 0.0, 0.0),
    )
    scale_step: FloatVectorProperty(
        name="Scale Step",
        description="Every next copy is scaled by this XYZ factor relative to the previous one",
        size=3, subtype='XYZ', default=(1.0, 1.0, 1.0),
    )
    scale_center: FloatVectorProperty(
        name="Scale Center",
        description="Center of the step scaling, offset from the origin of each copy (world axes)",
        size=3, subtype='TRANSLATION', unit='LENGTH', default=(0.0, 0.0, 0.0),
    )
    count: IntProperty(
        name="Copies",
        description="Number of new copies (the source object is not counted)",
        min=1, soft_max=100, default=2,
    )
    linked_data: BoolProperty(
        name="Linked Data",
        description="Share object data (mesh, curve...) between copies instead of duplicating it",
        default=False,
    )


PARAM_KEYS = (
    "step", "direction", "snap_to_grid", "snap_to_angle",
    "snap_angle", "rot_step", "rot_pivot", "scale_step", "scale_center",
    "count", "linked_data",
)


class ArrayCopySettings(ArrayCopyParams, PropertyGroup):
    pass


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _find_view3d_space(context):
    sd = context.space_data
    if sd is not None and sd.type == 'VIEW_3D':
        return sd
    screen = context.screen
    if screen:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                return area.spaces.active
    return None


def grid_step(context, params):
    """Snapping step in Blender units, derived from the viewport grid settings."""
    sd = _find_view3d_space(context)
    grid_scale = sd.overlay.grid_scale if sd is not None else 1.0
    unit = context.scene.unit_settings
    base = 1.0
    if unit.system == 'IMPERIAL':
        base = 0.3048
    if unit.system != 'NONE':
        base = base / max(unit.scale_length, 1e-9)
    else:
        base = 1.0
    size = grid_scale * base
    return max(size, 1e-9)


def tool_is_active(context):
    """True only if Array Copy is the active tool of the *current* workspace."""
    try:
        ws = context.workspace
        if ws is None:
            return False
        tool = ws.tools.from_space_view3d_mode('OBJECT', create=False)
        return tool is not None and tool.idname == TOOL_ID
    except Exception:
        return False


def pixel_size(rv3d, region, p):
    """World size of one pixel at point p."""
    pm = rv3d.perspective_matrix
    w = pm[3][0] * p[0] + pm[3][1] * p[1] + pm[3][2] * p[2] + pm[3][3]
    return abs(w) * 2.0 / (max(region.width, 1) * rv3d.window_matrix[0][0])


def copy_transforms(matrix_world, p, count):
    """Yield (location, rotation quaternion, scale) for every new copy."""
    loc0, rot0, scl0 = matrix_world.decompose()
    step = Vector(p.step)
    r_dir = Euler(tuple(p.direction), 'XYZ').to_matrix()
    rs = tuple(p.rot_step)
    pivot = Vector(p.rot_pivot)
    ss = tuple(p.scale_step)
    sc = Vector(p.scale_center)

    for i in range(1, count + 1):
        origin = loc0 + r_dir @ (step * i)

        # step rotation around pivot
        r_i = Euler((rs[0] * i, rs[1] * i, rs[2] * i), 'XYZ').to_matrix()
        loc = origin + pivot - r_i @ pivot
        rot = r_i.to_quaternion() @ rot0

        # step scale around center
        s = Vector((ss[0] ** i, ss[1] ** i, ss[2] ** i))
        center = origin + sc
        d = loc - center
        loc = center + Vector((d.x * s.x, d.y * s.y, d.z * s.z))
        scl = Vector((scl0.x * s.x, scl0.y * s.y, scl0.z * s.z))
        yield loc, rot, scl


def _remove_object(ob):
    data = ob.data
    bpy.data.objects.remove(ob, do_unlink=True)
    if data is not None and data.users == 0:
        try:
            bpy.data.batch_remove({data})
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Operator: builds the copies and the collection
# ----------------------------------------------------------------------------

class OBJECT_OT_array_copy_tool_apply(ArrayCopyParams, Operator):
    bl_idname = "object.array_copy_tool_apply"
    bl_label = "Array Copy"
    bl_description = "Create copies of the active object using the Array Copy tool parameters"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and context.active_object is not None

    def execute(self, context):
        src = context.active_object
        scene = context.scene

        if self.snap_to_grid:
            g = grid_step(context, self)
            self.step = [round(c / g) * g for c in self.step]
        if self.snap_to_angle:
            a = self.snap_angle
            self.direction = [round(c / a) * a for c in self.direction]

        # --- collection: reuse the one made by a previous run, otherwise create it
        coll = None
        for c in src.users_collection:
            if c.get(COLL_KEY) == src.name:
                coll = c
                break

        if coll is not None:
            for ob in list(coll.objects):
                if ob is not src and ob.get(OBJ_KEY) == src.name:
                    _remove_object(ob)
            # bases of the view layer must be rebuilt before anything touches them again
            context.view_layer.update()
        else:
            owners = list(src.users_collection)
            parent = owners[0] if owners else scene.collection
            coll = bpy.data.collections.new(src.name)
            coll[COLL_KEY] = src.name
            parent.children.link(coll)
            for c in owners:
                c.objects.unlink(src)
            coll.objects.link(src)

        # --- copies
        for loc, rot, scl in copy_transforms(src.matrix_world, self, self.count):
            new = src.copy()
            if src.data is not None and not self.linked_data:
                new.data = src.data.copy()
            new.name = src.name
            new[OBJ_KEY] = src.name
            coll.objects.link(new)
            new.matrix_world = Matrix.LocRotScale(loc, rot, scl)

        # --- keep the source selected so the gizmo stays on it.
        # View layer must be synced first, otherwise iterating its objects can
        # hit freed bases (this caused a crash on redo).
        context.view_layer.update()
        for ob in list(coll.objects):
            if ob is not src:
                try:
                    ob.select_set(False)
                except RuntimeError:
                    pass
        for ob in list(context.selected_objects):
            if ob is not src:
                try:
                    ob.select_set(False)
                except RuntimeError:
                    pass
        src.select_set(True)
        context.view_layer.objects.active = src

        # --- sync settings back so the gizmo matches the redo panel
        # Keep the other parameters for the next use, but do NOT keep the gizmo
        # offset/direction: the gizmo returns to the object after the copies are made.
        s = scene.array_copy_tool
        for key in PARAM_KEYS:
            if key in ("step", "direction"):
                continue
            setattr(s, key, getattr(self, key))
        s.step = (0.0, 0.0, 0.0)
        s.direction = (0.0, 0.0, 0.0)
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.prop(self, "count")

        box = layout.box()
        box.label(text="Position")
        box.prop(self, "step")
        box.prop(self, "direction")
        box.prop(self, "snap_to_grid")
        box.prop(self, "snap_to_angle")
        sub = box.column()
        sub.active = self.snap_to_angle
        sub.prop(self, "snap_angle")

        box = layout.box()
        box.label(text="Rotation per copy")
        box.prop(self, "rot_step")
        box.prop(self, "rot_pivot")

        box = layout.box()
        box.label(text="Scale per copy")
        box.prop(self, "scale_step")
        box.prop(self, "scale_center")

        layout.prop(self, "linked_data")


# ----------------------------------------------------------------------------
# Deferred commit: after the gizmo stops changing, run the operator
# ----------------------------------------------------------------------------

_pending = {"t": 0.0, "ctx": None}
_groups = []   # live gizmo group instances (used to detect an active drag)


def _gizmo_busy():
    """True while the user is still holding / dragging one of the gizmos."""
    busy = False
    for g in list(_groups):
        try:
            for gz in (*g.arrows, *g.dials):
                if getattr(gz, "is_modal", False):
                    busy = True
        except Exception:
            _groups.remove(g)
    return busy


def _is_trivial(s):
    return (Vector(s.step).length < 1e-9
            and all(abs(v) < 1e-9 for v in s.rot_step)
            and all(abs(v - 1.0) < 1e-9 for v in s.scale_step))


def _commit_timer():
    if _gizmo_busy():
        return 0.1                      # still dragging: wait for release
    remaining = COMMIT_DELAY - (time.monotonic() - _pending["t"])
    if remaining > 0.0:
        return remaining
    ctx = _pending["ctx"]
    _pending["ctx"] = None
    if ctx is None:
        return None
    window, area, region = ctx
    try:
        scene = window.scene
        s = scene.array_copy_tool
        if _is_trivial(s):
            return None                 # nothing to copy (e.g. plain click on the gizmo)
        kwargs = {}
        for key in PARAM_KEYS:
            v = getattr(s, key)
            kwargs[key] = tuple(v) if hasattr(v, "__len__") else v
        with bpy.context.temp_override(window=window, area=area, region=region):
            if bpy.ops.object.array_copy_tool_apply.poll():
                bpy.ops.object.array_copy_tool_apply('EXEC_DEFAULT', True, **kwargs)
                area.tag_redraw()
    except Exception as e:  # context may be gone, tool changed, etc.
        print("Array Copy Tool: commit failed:", e)
    return None


def _changed(context):
    _pending["t"] = time.monotonic()
    _pending["ctx"] = (context.window, context.area, context.region)
    if context.area:
        context.area.tag_redraw()
    if not bpy.app.timers.is_registered(_commit_timer):
        bpy.app.timers.register(_commit_timer, first_interval=COMMIT_DELAY)


# ----------------------------------------------------------------------------
# Gizmo group
# ----------------------------------------------------------------------------

_drag = {}


def _arrow_get():
    return 0.0


def _dial_get():
    return 0.0


def _make_arrow_set(axis):
    def _set(value):
        if "step" not in _drag:
            return
        ctx = bpy.context
        s = ctx.scene.array_copy_tool
        step = _drag["step"].copy()
        step[axis] += value * _drag["scale"]
        if s.snap_to_grid:
            g = grid_step(ctx, s)
            step[axis] = round(step[axis] / g) * g
        s.step = step
        _changed(ctx)
    return _set


def _make_dial_set(axis):
    def _set(value):
        if "dir_m" not in _drag:
            return
        ctx = bpy.context
        s = ctx.scene.array_copy_tool
        if s.snap_to_angle:
            inc = max(s.snap_angle, 1e-6)
            value = round(value / inc) * inc
        m = _drag["dir_m"] @ Matrix.Rotation(value, 3, AXIS_NAMES[axis])
        s.direction = m.to_euler('XYZ', _drag["dir_e"])
        _changed(ctx)
    return _set


class VIEW3D_GGT_array_copy(GizmoGroup):
    bl_idname = GIZMO_GROUP_ID
    bl_label = "Array Copy Gizmo"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'PERSISTENT', 'SCALE'}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (context.mode == 'OBJECT' and ob is not None and ob.select_get()
                and tool_is_active(context))

    def setup(self, context):
        self._scale = 1.0
        self.arrows = []
        self.dials = []
        _groups.append(self)
        for axis in range(3):
            col = AXIS_COLORS[axis]

            dial = self.gizmos.new("GIZMO_GT_dial_3d")
            dial.draw_options = {'CLIP'}
            dial.color = col
            dial.alpha = 0.7
            dial.color_highlight = col
            dial.alpha_highlight = 1.0
            dial.line_width = 3.0
            dial.scale_basis = 0.8
            dial.use_draw_modal = True
            dial.target_set_handler("offset", get=_dial_get, set=_make_dial_set(axis))
            self.dials.append(dial)

            arrow = self.gizmos.new("GIZMO_GT_arrow_3d")
            arrow.color = col
            arrow.alpha = 0.9
            arrow.color_highlight = col
            arrow.alpha_highlight = 1.0
            arrow.use_draw_modal = True
            arrow.target_set_handler("offset", get=_arrow_get, set=_make_arrow_set(axis))
            self.arrows.append(arrow)

    def invoke_prepare(self, context, gizmo):
        s = context.scene.array_copy_tool
        _drag.clear()
        _drag["step"] = Vector(s.step)
        _drag["dir_e"] = Euler(tuple(s.direction), 'XYZ')
        _drag["dir_m"] = _drag["dir_e"].to_matrix()
        _drag["scale"] = self._scale

    def draw_prepare(self, context):
        ob = context.active_object
        rv3d = context.region_data
        region = context.region
        visible = (ob is not None and rv3d is not None and region is not None
                   and tool_is_active(context))
        for gz in (*self.arrows, *self.dials):
            gz.hide = not visible
        if not visible:
            return
        s = context.scene.array_copy_tool
        r_dir = Euler(tuple(s.direction), 'XYZ').to_matrix()
        center = ob.matrix_world.translation + r_dir @ Vector(s.step)

        origin = ob.matrix_world.translation
        px = context.preferences.view.gizmo_size * 0.7

        # arrows: follow the step offset
        size = pixel_size(rv3d, region, center) * px
        self._scale = size if size > 1e-9 else 1.0
        arrow_base = (Matrix.Translation(center) @ r_dir.to_4x4()
                      @ Matrix.Diagonal((self._scale, self._scale, self._scale, 1.0)))

        # rotation rings: always fixed at the origin of the source object
        dsize = pixel_size(rv3d, region, origin) * px
        dsize = dsize if dsize > 1e-9 else 1.0
        dial_base = (Matrix.Translation(origin) @ r_dir.to_4x4()
                     @ Matrix.Diagonal((dsize, dsize, dsize, 1.0)))

        for axis in range(3):
            self.arrows[axis].matrix_basis = arrow_base @ AXIS_MATS[axis]
            self.dials[axis].matrix_basis = dial_base @ AXIS_MATS[axis]


# ----------------------------------------------------------------------------
# Viewport preview of the copy positions
# ----------------------------------------------------------------------------

_draw_handle = None


def _draw_preview():
    ctx = bpy.context
    try:
        if ctx.mode != 'OBJECT':
            return
        if not tool_is_active(ctx):
            return
        ob = ctx.active_object
        rv3d = ctx.region_data
        region = ctx.region
        if ob is None or not ob.select_get() or rv3d is None or region is None:
            return

        s = ctx.scene.array_copy_tool
        origin = ob.matrix_world.translation
        r_dir = Euler(tuple(s.direction), 'XYZ').to_matrix()
        center = origin + r_dir @ Vector(s.step)

        pts = [origin, center]
        for loc, _rot, _scl in copy_transforms(ob.matrix_world, s, s.count):
            k = pixel_size(rv3d, region, loc) * 6.0
            for v in (Vector((k, 0, 0)), Vector((0, k, 0)), Vector((0, 0, k))):
                pts.append(loc - v)
                pts.append(loc + v)

        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINES', {"pos": [tuple(p) for p in pts]})
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(2.0)
        shader.bind()
        shader.uniform_float("color", (1.0, 0.75, 0.2, 0.9))
        batch.draw(shader)
        gpu.state.line_width_set(1.0)
        gpu.state.blend_set('NONE')
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Tool
# ----------------------------------------------------------------------------

class ArrayCopyTool(WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'OBJECT'
    bl_idname = TOOL_ID
    bl_label = "Array Copy"
    bl_description = (
        "Copy the selected object with a gizmo: move the gizmo to set the offset step, "
        "rotate it to set the direction"
    )
    bl_icon = "ops.transform.transform"
    bl_widget = GIZMO_GROUP_ID
    bl_keymap = (
        ("view3d.select",
         {"type": 'LEFTMOUSE', "value": 'CLICK'},
         {"properties": [("deselect_all", True)]}),
        ("view3d.select_box",
         {"type": 'LEFTMOUSE', "value": 'CLICK_DRAG'},
         {"properties": [("mode", 'SET')]}),
    )

    @staticmethod
    def draw_settings(context, layout, tool):
        s = context.scene.array_copy_tool

        row = layout.row(align=True)
        row.prop(s, "snap_to_grid", toggle=True)

        layout.separator()

        row = layout.row(align=True)
        row.prop(s, "snap_to_angle", toggle=True)
        sub = row.row(align=True)
        sub.active = s.snap_to_angle
        sub.prop(s, "snap_angle", text="")


# ----------------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------------

classes = (
    ArrayCopySettings,
    OBJECT_OT_array_copy_tool_apply,
    VIEW3D_GGT_array_copy,
)


def register():
    global _draw_handle
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.array_copy_tool = PointerProperty(type=ArrayCopySettings)
    _draw_handle = bpy.types.SpaceView3D.draw_handler_add(_draw_preview, (), 'WINDOW', 'POST_VIEW')
    bpy.utils.register_tool(ArrayCopyTool, after=None, separator=False, group=False)


def unregister():
    global _draw_handle
    if bpy.app.timers.is_registered(_commit_timer):
        bpy.app.timers.unregister(_commit_timer)
    _groups.clear()
    bpy.utils.unregister_tool(ArrayCopyTool)
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        _draw_handle = None
    del bpy.types.Scene.array_copy_tool
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
