bl_info = {
    "name": "Snap Vertices to Grid Size",
    "author": "Google Gemini",
    "version": (1, 0),
    "blender": (4, 0, 0),
    "location": "Edit Mode > Vertex Menu > Snap to Grid Size",
    "description": "Притягивает выделенные вершины к ближайшему шагу сетки с учётом масштаба и юнитов",
    "category": "Mesh",
}

import bpy
import bmesh
from mathutils import Vector

class MESH_OT_snap_verts_to_grid(bpy.types.Operator):
    """Притянуть выделенные вершины к глобальной сетке с учётом её масштаба"""
    bl_idname = "mesh.snap_verts_to_grid"
    bl_label = "Snap to Grid Size"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and context.active_object is not None

    def execute(self, context):
        obj = context.active_object
        mw = obj.matrix_world  # Матрица перевода из локальных в мировые координаты
        mwi = mw.inverted()    # Обратная матрица (из мировых в локальные)

        # 1. Получаем настройки масштаба юнитов сцены
        unit_settings = context.scene.unit_settings
        unit_scale = unit_settings.scale_length

        # 2. Получаем настройки масштаба сетки из оверлея 3D Viewport
        grid_scale = 1.0
        if context.space_data and context.space_data.type == 'VIEW_3D':
            grid_scale = context.space_data.overlay.grid_scale
        else:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    grid_scale = area.spaces.active.overlay.grid_scale
                    break

        # Итоговый шаг сетки
        grid_step = unit_scale * grid_scale

        if grid_step <= 0:
            self.report({'WARNING'}, "Не удалось определить размер сетки.")
            return {'CANCELLED'}

        # 3. Модификация геометрии
        bm = bmesh.from_edit_mesh(obj.data)
        selected_verts = [v for v in bm.verts if v.select]
        
        if not selected_verts:
            self.report({'INFO'}, "Нет выделенных вершин.")
            return {'CANCELLED'}

        for v in selected_verts:
            # Локальные координаты -> Мировые
            world_pos = mw @ v.co

            # Округление до ближайшего шага сетки
            snap_x = round(world_pos.x / grid_step) * grid_step
            snap_y = round(world_pos.y / grid_step) * grid_step
            snap_z = round(world_pos.z / grid_step) * grid_step

            # Мировые координаты -> Локальные
            v.co = mwi @ Vector((snap_x, snap_y, snap_z))

        bmesh.update_edit_mesh(obj.data)
        
        self.report({'INFO'}, f"Выровнено вершин: {len(selected_verts)} (Шаг сетки: {grid_step:.4f} м)")
        return {'FINISHED'}


def menu_func(self, context):
    self.layout.separator()
    self.layout.operator(MESH_OT_snap_verts_to_grid.bl_idname, icon='GRID')

def register():
    bpy.utils.register_class(MESH_OT_snap_verts_to_grid)
    bpy.types.VIEW3D_MT_edit_mesh_vertices.append(menu_func)

def unregister():
    bpy.types.VIEW3D_MT_edit_mesh_vertices.remove(menu_func)
    bpy.utils.unregister_class(MESH_OT_snap_verts_to_grid)

if __name__ == "__main__":
    register()