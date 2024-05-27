import taichi as ti

ti.init(arch=ti.cpu)

S = ti.root.dynamic(ti.i, 1024, chunk_size=32)
x = ti.field(int)
S.place(x)

@ti.kernel
def add_data():
    for i in range(1000):
        x.append(i)
        # print(x.length())

# add_data()

@ti.kernel
def clear_data():
    x.deactivate()
    print(x.length())  # will print 0

clear_data()