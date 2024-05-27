import taichi as ti
import taichi.math as tm
# @ti.func
# def point_to_line_distance(point, line_start, line_end):
#     # 将点和线段端点转换为Taichi向量
#     p = ti.Vector([point[0], point[1]])
#     l1 = ti.Vector([line_start[0], line_start[1]])
#     l2 = ti.Vector([line_end[0], line_end[1]])
#
#     # 计算线段的方向向量
#     d = l2 - l1
#
#     # 当方向向量长度接近0时（即线段长度几乎为0），直接返回点到线段任一端点的距离
#     # if ti.abs(d) < ti.f32(1e-6):  # 这里设定一个小于1e-6的阈值作为近似零判断
#     #     return tm.normalize(p - l1)
#
#     # 计算垂直于线段方向向量的向量
#     v = ti.Vector([-d.y, d.x])
#     print(v)
#
#     # 计算垂足在直线上的投影坐标
#     projection = l1 + (p - l1).dot(v) * d / d.dot(d)
#     ppp = (p - l1).dot(d) / d.dot(d)
#     print('projection', ppp)
#
#     # 检查垂足是否在线段内部
#     is_inside_segment = (l1.x <= projection.x <= l2.x) and \
#                        (l1.y <= projection.y <= l2.y)
#
#     # 如果垂足在线段上，则返回点到垂足的距离；否则返回点到线段端点的最小距离
#     return tm.length(p - projection)
#
# # 使用示例（需要在ti.init()之后调用）
# @ti.kernel
# def run_example():
#     x = 1.011
#     print(ti.cast(x, ti.i32))
#     point = ti.Vector([3.0, 5.0])
#     line_start = ti.Vector([0.0, 3.0])
#     line_end = ti.Vector([6.0, 3.0])
#     dist = point_to_line_distance(point, line_start, line_end)
#     print(dist)
#
# ti.init()
# run_example()

ti.init(arch=ti.cpu)
S = ti.root.dense(ti.i, 10).dynamic(ti.j, 1024, chunk_size=32)
x = ti.field(int)
S.place(x)
@ti.kernel
def add_data():
    for i in range(10):
        for j in range(i):
            x[i].append(j)
        print(x[i].length())  # will print i

    x.deactivate()
    for i in range(10):
        # x[i].deactivate()
        print(x[i].length())  # will print 0

add_data()