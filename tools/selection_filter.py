bl_info = {
    "name": "Selection Filter",
    "author": "Anthropic Claude",
    "version": (1, 0, 0),
    "blender": (5, 2, 0),
    "location": "3D Viewport > Edit Mode > Sidebar (N) > Filter / Select menu",
    "description": "Оставляет выделенными только элементы (вершины, рёбра или грани), "
                   "подходящие под заданные параметры; опционально - наоборот",
    "category": "Mesh",
}

import math

import bmesh
import bpy
from bpy.props import (BoolProperty, BoolVectorProperty, EnumProperty, FloatProperty,
                       FloatVectorProperty, PointerProperty)
from bpy.types import Operator, Panel, PropertyGroup
from mathutils import Vector
from mathutils.bvhtree import BVHTree

MODE_LABELS = {'VERT': "вершины", 'EDGE': "рёбра", 'FACE': "грани"}


def element_mode(context):
    """С какими элементами работает фильтр - зависит от режима выделения."""
    m = context.tool_settings.mesh_select_mode
    if m[2]:
        return 'FACE'
    if m[1]:
        return 'EDGE'
    return 'VERT'


# ----------------------------------------------------------------------------
# Настройки
# ----------------------------------------------------------------------------
class SF_Settings(PropertyGroup):
    invert: BoolProperty(
        name="Invert",
        description="Инвертировать: остаются только элементы, которые НЕ подходят под параметры",
        default=False,
    )
    space: EnumProperty(
        name="Space",
        description="В каких координатах заданы значения (координаты, направления, нормали)",
        items=(
            ('GLOBAL', "Global", "Мировые координаты"),
            ('LOCAL', "Local", "Локальные координаты каждого объекта"),
        ),
        default='GLOBAL',
    )

    # --- координаты (все типы элементов) ---
    use_coords: BoolProperty(name="Coordinates", default=False,
                             description="Фильтр по совпадению отдельных составляющих координат")
    coord_axes: BoolVectorProperty(
        name="Axes", size=3, default=(False, False, True),
        description="Какие составляющие координат должны совпасть (X, Y, Z)")
    coord_mode: EnumProperty(
        name="Match",
        items=(
            ('CENTER', "Center", "Сравнивается центр элемента (для вершины - она сама)"),
            ('ALL', "All Vertices", "Совпасть должны все вершины элемента"),
            ('ANY', "Any Vertex", "Достаточно совпадения хотя бы одной вершины элемента"),
        ),
        default='CENTER',
    )
    coord_active: BoolProperty(
        name="From Active", default=False,
        description="Брать значения координат у активного элемента")
    coord_value: FloatVectorProperty(
        name="Value", size=3, default=(0.0, 0.0, 0.0), subtype='TRANSLATION', unit='LENGTH',
        description="Значения координат для сравнения")
    coord_tol: FloatProperty(
        name="Tolerance", default=1e-4, min=0.0, precision=6, subtype='DISTANCE', unit='LENGTH',
        description="Допустимое отличие координаты")

    # --- направление ребра ---
    use_direction: BoolProperty(name="Edge Direction", default=False,
                                description="Фильтр рёбер по направлению")
    dir_active: BoolProperty(name="From Active Edge", default=False,
                             description="Брать направление у активного ребра")
    dir_value: FloatVectorProperty(
        name="Direction", size=3, default=(0.0, 0.0, 1.0), subtype='DIRECTION',
        description="Направление для сравнения")
    dir_angle: FloatProperty(
        name="Angle", default=math.radians(1.0), min=0.0, max=math.radians(90.0),
        subtype='ANGLE', unit='ROTATION', description="Допустимое отклонение по углу")
    dir_both: BoolProperty(
        name="Both Directions", default=True,
        description="Считать противоположные направления одинаковыми")

    # --- нормаль грани ---
    use_normal: BoolProperty(name="Face Normal", default=False,
                             description="Фильтр граней по направлению нормали")
    normal_active: BoolProperty(name="From Active Face", default=False,
                                description="Брать нормаль активной грани")
    normal_value: FloatVectorProperty(
        name="Normal", size=3, default=(0.0, 0.0, 1.0), subtype='DIRECTION',
        description="Нормаль для сравнения")
    normal_angle: FloatProperty(
        name="Angle", default=math.radians(1.0), min=0.0, max=math.radians(180.0),
        subtype='ANGLE', unit='ROTATION', description="Допустимое отклонение по углу")

    # --- материал грани ---
    use_material: BoolProperty(name="Material", default=False,
                               description="Фильтр граней по материалу")
    material_active: BoolProperty(name="From Active Face", default=False,
                                  description="Брать материал активной грани")
    material: PointerProperty(
        name="Material", type=bpy.types.Material,
        description="Материал для сравнения (пусто - грани без материала)")

    # --- компланарность ---
    use_coplanar: BoolProperty(
        name="Coplanar", default=False,
        description="Фильтр граней, лежащих в одной плоскости с активной гранью")
    coplanar_dist: FloatProperty(
        name="Distance", default=1e-4, min=0.0, precision=6, subtype='DISTANCE', unit='LENGTH',
        description="Допустимое расстояние центра грани до плоскости активной грани")
    coplanar_angle: FloatProperty(
        name="Angle", default=math.radians(1.0), min=0.0, max=math.radians(90.0),
        subtype='ANGLE', unit='ROTATION', description="Допустимое отличие нормалей по углу")
    coplanar_flip: BoolProperty(
        name="Ignore Normal Sign", default=True,
        description="Считать компланарными и грани, развёрнутые нормалью в обратную сторону")

    # --- сторона нормали ---
    use_side: BoolProperty(name="Normal Side", default=False,
                           description="Фильтр граней: нормаль направлена наружу или внутрь объекта")
    side: EnumProperty(
        name="Side",
        items=(
            ('OUTWARD', "Outward", "Нормаль направлена от объекта (наружу)"),
            ('INWARD', "Inward", "Нормаль направлена внутрь объекта"),
        ),
        default='OUTWARD',
    )
    side_method: EnumProperty(
        name="Method",
        items=(
            ('RAYCAST', "Raycast",
             "Точно, в том числе для вогнутых форм: чётность пересечений луча вдоль нормали "
             "с мешем (нужен замкнутый меш)"),
            ('CENTER', "Center", "Быстро: по направлению от центра объёма (bounding box) к грани"),
        ),
        default='RAYCAST',
    )


# ----------------------------------------------------------------------------
# Вспомогательное
# ----------------------------------------------------------------------------
def angle_ok(a, b, tol, both):
    la, lb = a.length, b.length
    if la < 1e-12 or lb < 1e-12:
        return False
    d = a.dot(b) / (la * lb)
    if both:
        d = abs(d)
    return d >= math.cos(tol) - 1e-9


def active_point(bm, hist):
    """Точка активного элемента (локальные координаты): вершина, центр ребра или грани."""
    if isinstance(hist, bmesh.types.BMVert):
        return hist.co.copy()
    if isinstance(hist, bmesh.types.BMEdge):
        return (hist.verts[0].co + hist.verts[1].co) * 0.5
    if isinstance(hist, bmesh.types.BMFace):
        return hist.calc_center_median()
    if bm.faces.active is not None:
        return bm.faces.active.calc_center_median()
    return None


def active_face(bm, hist):
    if bm.faces.active is not None:
        return bm.faces.active
    return hist if isinstance(hist, bmesh.types.BMFace) else None


# ----------------------------------------------------------------------------
# Оператор фильтрации
# ----------------------------------------------------------------------------
class MESH_OT_filter_selection(Operator):
    bl_idname = "mesh.filter_selection"
    bl_label = "Filter Selection"
    bl_description = ("Оставляет выделенными только элементы, подходящие под параметры фильтра "
                      "(с галочкой Invert - только не подходящие)")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and context.edit_object is not None

    def enabled_checks(self, s, mode):
        checks = set()
        if s.use_coords:
            checks.add('coords')
        if mode == 'EDGE' and s.use_direction:
            checks.add('direction')
        if mode == 'FACE':
            if s.use_normal:
                checks.add('normal')
            if s.use_material:
                checks.add('material')
            if s.use_coplanar:
                checks.add('coplanar')
            if s.use_side:
                checks.add('side')
        return checks

    def build_refs(self, context, s, mode, obj, checks):
        """Эталонные значения (из полей или из активного элемента). None - ошибка."""
        ref = {}
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        bm.normal_update()
        mw = obj.matrix_world
        n_mat = mw.to_3x3().inverted_safe().transposed()
        hist = bm.select_history.active
        local = s.space == 'LOCAL'

        if 'coords' in checks:
            if s.coord_active:
                pt = active_point(bm, hist)
                if pt is None:
                    self.report({'ERROR'}, "Нет активного элемента для координат")
                    return None
                ref['coords'] = pt if local else mw @ pt
            else:
                ref['coords'] = Vector(s.coord_value)

        if 'direction' in checks:
            if s.dir_active:
                if not isinstance(hist, bmesh.types.BMEdge):
                    self.report({'ERROR'}, "Нет активного ребра (выберите ребро последним)")
                    return None
                d = hist.verts[1].co - hist.verts[0].co
                ref['direction'] = d if local else mw.to_3x3() @ d
            else:
                ref['direction'] = Vector(s.dir_value)
            if ref['direction'].length < 1e-9:
                self.report({'ERROR'}, "Направление не должно быть нулевым")
                return None

        if 'normal' in checks:
            if s.normal_active:
                f = active_face(bm, hist)
                if f is None:
                    self.report({'ERROR'}, "Нет активной грани для нормали")
                    return None
                ref['normal'] = f.normal.copy() if local else (n_mat @ f.normal).normalized()
            else:
                ref['normal'] = Vector(s.normal_value)
            if ref['normal'].length < 1e-9:
                self.report({'ERROR'}, "Нормаль не должна быть нулевой")
                return None

        if 'material' in checks:
            if s.material_active:
                f = active_face(bm, hist)
                if f is None:
                    self.report({'ERROR'}, "Нет активной грани для материала")
                    return None
                slots = obj.material_slots
                idx = f.material_index
                ref['material'] = slots[idx].material if idx < len(slots) else None
            else:
                ref['material'] = s.material

        if 'coplanar' in checks:
            f = active_face(bm, hist)
            if f is None:
                self.report({'ERROR'}, "Для компланарности нужна активная грань")
                return None
            ref['plane'] = (mw @ f.calc_center_median(), (n_mat @ f.normal).normalized())
        return ref

    @staticmethod
    def coords_ok(s, pts, center, mw, local, ref):
        points = pts if s.coord_mode in {'ALL', 'ANY'} else [center]
        if not local:
            points = [mw @ p for p in points]
        for axis in range(3):
            if not s.coord_axes[axis]:
                continue
            hits = [abs(p[axis] - ref[axis]) <= s.coord_tol + 1e-12 for p in points]
            if s.coord_mode == 'ANY':
                if not any(hits):
                    return False
            elif not all(hits):
                return False
        return True

    def filter_object(self, obj, s, mode, checks, ref):
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        bm.normal_update()
        mw = obj.matrix_world
        m3 = mw.to_3x3()
        n_mat = m3.inverted_safe().transposed()
        local = s.space == 'LOCAL'
        slots = obj.material_slots

        # данные для проверки стороны нормали
        tree, bb_center, diag = None, Vector((0.0, 0.0, 0.0)), 1.0
        if 'side' in checks:
            cos_ = [v.co for v in bm.verts]
            if cos_:
                lo = Vector((min(c.x for c in cos_), min(c.y for c in cos_), min(c.z for c in cos_)))
                hi = Vector((max(c.x for c in cos_), max(c.y for c in cos_), max(c.z for c in cos_)))
                bb_center = (lo + hi) * 0.5
                diag = max((hi - lo).length, 1e-6)
            if s.side_method == 'RAYCAST':
                tree = BVHTree.FromBMesh(bm)

        if mode == 'VERT':
            elems = [v for v in bm.verts if v.select and not v.hide]
        elif mode == 'EDGE':
            elems = [e for e in bm.edges if e.select and not e.hide]
        else:
            elems = [f for f in bm.faces if f.select and not f.hide]

        failing = []
        for e in elems:
            if mode == 'VERT':
                pts, center = [e.co], e.co
            elif mode == 'EDGE':
                pts = [e.verts[0].co, e.verts[1].co]
                center = (pts[0] + pts[1]) * 0.5
            else:
                pts = [v.co for v in e.verts]
                center = e.calc_center_median()

            ok = True
            if ok and 'coords' in checks:
                ok = self.coords_ok(s, pts, center, mw, local, ref['coords'])

            if ok and 'direction' in checks:
                d = pts[1] - pts[0]
                if not local:
                    d = m3 @ d
                ok = angle_ok(d, ref['direction'], s.dir_angle, s.dir_both)

            if ok and 'normal' in checks:
                n = e.normal.copy() if local else (n_mat @ e.normal).normalized()
                ok = angle_ok(n, ref['normal'], s.normal_angle, False)

            if ok and 'material' in checks:
                idx = e.material_index
                mat = slots[idx].material if idx < len(slots) else None
                ok = mat == ref['material']

            if ok and 'coplanar' in checks:
                pc, pn = ref['plane']
                n_w = (n_mat @ e.normal).normalized()
                c_w = mw @ center
                ok = (angle_ok(n_w, pn, s.coplanar_angle, s.coplanar_flip)
                      and abs((c_w - pc).dot(pn)) <= s.coplanar_dist + 1e-12)

            if ok and 'side' in checks:
                n = e.normal
                if tree is not None:
                    eps = diag * 1e-5
                    origin = center + n * eps
                    count = 0
                    for _ in range(64):
                        loc, _nor, _idx, _dist = tree.ray_cast(origin, n)
                        if loc is None:
                            break
                        count += 1
                        origin = loc + n * eps
                    outward = (count % 2 == 0)
                else:
                    outward = n.dot(center - bb_center) > 0.0
                ok = outward if s.side == 'OUTWARD' else not outward

            if s.invert:
                ok = not ok
            if not ok:
                failing.append(e)

        for e in failing:
            e.select_set(False)
        bm.select_flush_mode()
        bm.select_history.validate()
        bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
        return len(elems) - len(failing), len(failing)

    def execute(self, context):
        s = context.scene.selection_filter
        mode = element_mode(context)
        checks = self.enabled_checks(s, mode)
        if not checks:
            self.report({'INFO'}, f"Не включено ни одного фильтра для режима «{MODE_LABELS[mode]}»")
            return {'CANCELLED'}
        objs = [o for o in context.objects_in_mode_unique_data if o.type == 'MESH']
        if not objs:
            return {'CANCELLED'}

        ref = self.build_refs(context, s, mode, context.edit_object, checks)
        if ref is None:
            return {'CANCELLED'}

        kept = removed = 0
        for obj in objs:
            k, r = self.filter_object(obj, s, mode, checks, ref)
            kept += k
            removed += r
        self.report({'INFO'}, f"Осталось выделено: {kept}, снято: {removed} ({MODE_LABELS[mode]})")
        return {'FINISHED'}


class MESH_OT_filter_selection_set_vector(Operator):
    bl_idname = "mesh.filter_selection_set_vector"
    bl_label = "Set Axis"
    bl_description = "Задать направление вдоль оси"
    bl_options = {'INTERNAL', 'UNDO'}

    target: EnumProperty(items=(('NORMAL', "Normal", ""), ('DIRECTION', "Direction", "")))
    vector: FloatVectorProperty(size=3)

    def execute(self, context):
        s = context.scene.selection_filter
        if self.target == 'NORMAL':
            s.normal_value = self.vector
            s.normal_active = False
        else:
            s.dir_value = self.vector
            s.dir_active = False
        return {'FINISHED'}


# ----------------------------------------------------------------------------
# Интерфейс
# ----------------------------------------------------------------------------
def axis_buttons(layout, target):
    row = layout.row(align=True)
    for label, vec in (("+X", (1, 0, 0)), ("-X", (-1, 0, 0)), ("+Y", (0, 1, 0)),
                       ("-Y", (0, -1, 0)), ("+Z", (0, 0, 1)), ("-Z", (0, 0, -1))):
        op = row.operator(MESH_OT_filter_selection_set_vector.bl_idname, text=label)
        op.target = target
        op.vector = vec


class SF_PT_main(Panel):
    bl_idname = "VIEW3D_PT_selection_filter"
    bl_label = "Selection Filter"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Filter"

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def draw(self, context):
        s = context.scene.selection_filter
        layout = self.layout
        layout.label(text=f"Элементы: {MODE_LABELS[element_mode(context)]}", icon='RESTRICT_SELECT_OFF')
        layout.prop(s, "space", expand=True)
        layout.prop(s, "invert", icon='ARROW_LEFTRIGHT')
        layout.operator(MESH_OT_filter_selection.bl_idname, icon='FILTER')


class _SubPanel(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Filter"
    bl_parent_id = SF_PT_main.bl_idname
    bl_options = {'DEFAULT_CLOSED'}
    modes = {'VERT', 'EDGE', 'FACE'}
    flag = ""

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and element_mode(context) in cls.modes

    def draw_header(self, context):
        self.layout.prop(context.scene.selection_filter, self.flag, text="")


class SF_PT_coords(_SubPanel):
    bl_label = "Coordinates"
    flag = "use_coords"

    def draw(self, context):
        s = context.scene.selection_filter
        col = self.layout.column(align=True)
        col.active = s.use_coords
        col.prop(s, "coord_axes", text="Axes", toggle=True)
        col.prop(s, "coord_mode", text="")
        col.prop(s, "coord_active")
        sub = col.column(align=True)
        sub.active = s.use_coords and not s.coord_active
        sub.prop(s, "coord_value", text="")
        col.prop(s, "coord_tol")


class SF_PT_direction(_SubPanel):
    bl_label = "Edge Direction"
    flag = "use_direction"
    modes = {'EDGE'}

    def draw(self, context):
        s = context.scene.selection_filter
        col = self.layout.column(align=True)
        col.active = s.use_direction
        col.prop(s, "dir_active")
        sub = col.column(align=True)
        sub.active = s.use_direction and not s.dir_active
        sub.prop(s, "dir_value", text="")
        axis_buttons(sub, 'DIRECTION')
        col.prop(s, "dir_angle")
        col.prop(s, "dir_both")


class SF_PT_normal(_SubPanel):
    bl_label = "Face Normal"
    flag = "use_normal"
    modes = {'FACE'}

    def draw(self, context):
        s = context.scene.selection_filter
        col = self.layout.column(align=True)
        col.active = s.use_normal
        col.prop(s, "normal_active")
        sub = col.column(align=True)
        sub.active = s.use_normal and not s.normal_active
        sub.prop(s, "normal_value", text="")
        axis_buttons(sub, 'NORMAL')
        col.prop(s, "normal_angle")


class SF_PT_material(_SubPanel):
    bl_label = "Material"
    flag = "use_material"
    modes = {'FACE'}

    def draw(self, context):
        s = context.scene.selection_filter
        col = self.layout.column(align=True)
        col.active = s.use_material
        col.prop(s, "material_active")
        sub = col.column(align=True)
        sub.active = s.use_material and not s.material_active
        sub.prop(s, "material", text="")


class SF_PT_coplanar(_SubPanel):
    bl_label = "Coplanar (with active face)"
    flag = "use_coplanar"
    modes = {'FACE'}

    def draw(self, context):
        s = context.scene.selection_filter
        col = self.layout.column(align=True)
        col.active = s.use_coplanar
        col.prop(s, "coplanar_dist")
        col.prop(s, "coplanar_angle")
        col.prop(s, "coplanar_flip")


class SF_PT_side(_SubPanel):
    bl_label = "Normal Side"
    flag = "use_side"
    modes = {'FACE'}

    def draw(self, context):
        s = context.scene.selection_filter
        col = self.layout.column(align=True)
        col.active = s.use_side
        col.prop(s, "side", expand=True)
        col.prop(s, "side_method", text="")


def menu_func(self, context):
    self.layout.separator()
    self.layout.operator(MESH_OT_filter_selection.bl_idname, icon='FILTER')


classes = (
    SF_Settings,
    MESH_OT_filter_selection,
    MESH_OT_filter_selection_set_vector,
    SF_PT_main,
    SF_PT_coords,
    SF_PT_direction,
    SF_PT_normal,
    SF_PT_material,
    SF_PT_coplanar,
    SF_PT_side,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.selection_filter = PointerProperty(type=SF_Settings)
    bpy.types.VIEW3D_MT_select_edit_mesh.append(menu_func)


def unregister():
    bpy.types.VIEW3D_MT_select_edit_mesh.remove(menu_func)
    del bpy.types.Scene.selection_filter
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
